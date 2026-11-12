"""Reversibly collect legacy normal runs into flat data/runs/<run_id> records.

Use --apply explicitly. Original files move into an ignored migration backup
only after canonical copies pass SHA-256 checks. No training or inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from scripts.dataset.digit_dataset_config import PROJECT_ROOT
from utilities.digit_run_artifacts import load_run, save_run_record, write_json, write_run_readme

RUN_ID = re.compile(r'(\d{8}_\d{6}_[0-9a-f]{8})')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root=PROJECT_ROOT):
    records = {}
    for folder, kind in [('models', 'checkpoint.pt'), ('validation_outputs', 'output.pt')]:
        for path in sorted((Path(root) / 'data' / folder).glob('*.pt')):
            match = RUN_ID.search(path.name)
            if not match:
                raise ValueError(f'Unrecognized artifact name; inspect before migration: {path}')
            record = records.setdefault(match[1], {})
            if kind in record:
                raise ValueError(f'Ambiguous duplicate run ID: {match[1]}')
            record[kind] = path
    return records


def migrate(root=PROJECT_ROOT, resume_backup=None):
    root = Path(root).resolve()
    records = inventory(root)
    if not records:
        raise ValueError("No legacy artifacts found; nothing to migrate")
    for run_id in records:
        if (root / 'data/runs' / run_id).exists() and resume_backup is None:
            raise FileExistsError(f'Canonical record already exists: {run_id}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    backup = Path(resume_backup).resolve() if resume_backup else root / 'data/tmp/migrations' / stamp
    if resume_backup:
        assert backup.is_dir() and backup.parent == root / 'data/tmp/migrations'
        assert not (backup / 'manifest.json').exists(), 'Only a pre-archive copy failure can resume this way'
        assert all(Path(p).is_file() for paths in records.values() for p in paths.values())
    else:
        backup.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'copying', 'root': str(root), 'backup': str(backup),
                'files': [], 'run_ids': list(records), 'unassigned_predictions': [],
                'rollback': 'Restore each original_backup to original_path; remove only migration-created canonical run directories after verifying restored hashes. Index backups are listed separately. No original serialized records were rewritten.'}
    mapping = {}

    def copy_verified(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if resume_backup and digest(source) == digest(destination):
                return digest(source)
            raise FileExistsError(destination)
        shutil.copy2(source, destination)
        assert digest(source) == digest(destination), destination
        return digest(source)

    notes = list((root / 'data/runs/experiments/notes').glob('*.md'))
    for run_id, paths in records.items():
        directory = root / 'data/runs' / run_id
        directory.mkdir(parents=True, exist_ok=resume_backup is not None)
        artifact = None
        if 'output.pt' in paths:
            artifact = load_run(paths['output.pt'], root=root)
            assert artifact['run_id'] == run_id
        for kind, source in paths.items():
            destination = directory / kind
            sha = copy_verified(source, destination)
            mapping[str(source)] = str(destination)
            manifest['files'].append({'original_path': str(source), 'canonical_path': str(destination),
                                     'sha256': sha, 'original_backup': str(backup / 'originals' / source.relative_to(root))})
        if artifact is not None:
            save_run_record(directory, artifact,
                            status='completed' if 'checkpoint.pt' in paths else 'missing_checkpoint')
        else:
            write_json(directory / 'record.json', {'run_id': run_id,
                                                  'note': 'No paired validation record; no completion claim.'})
        group = Path(artifact['sweep_directory']) if artifact and artifact.get('sweep_directory') else None
        if group and (group / 'sweep.log').is_file():
            # Keep only this trial's bounded START/DONE segment when available.
            text = (group / 'sweep.log').read_text()
            marker = 'START ' + run_id
            if marker in text:
                segment = text.split(marker, 1)[1]
                segment = marker + segment.split('\nSTART ', 1)[0]
                end = segment.find('\nDONE ' + run_id)
                if end >= 0:
                    segment = segment[:segment.find('\n', end + 1) if '\n' in segment[end+1:] else len(segment)]
                (directory / 'run.log').write_text(segment.rstrip() + '\n')
        for note in notes:
            text = note.read_text()
            if run_id in text or (group and group.name in text):
                copy_verified(note, directory / note.name)
        if group:
            for summary in group.glob('error_investigations/*/summary.json'):
                if json.loads(summary.read_text()).get('run_id') == run_id:
                    for source in summary.parent.iterdir():
                        if source.is_file():
                            target = directory / source.name
                            if target.exists():
                                if digest(source) == digest(target):
                                    continue
                                target = directory / (summary.parent.name + '_' + source.name)
                            copy_verified(source, target)
        write_run_readme(directory, artifact, status=('checkpoint_only_incomplete' if artifact is None else 'completed' if 'checkpoint.pt' in paths else 'missing_checkpoint'))
    # Flat group indexes remain shared; update their active links after backup.
    manifest['index_backups'] = []
    def replace_links(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [replace_links(v) for v in value]
        if isinstance(value, dict):
            return {k: replace_links(v) for k, v in value.items()}
        return value
    for group in (root / 'data/runs/experiments').glob('bespoke_lr_sweep_*'):
        for source in [*group.glob('*.json'), *group.glob('*.csv'), *group.glob('*/*.json'), *group.glob('*/*.csv')]:
            if source.suffix == '.json':
                value = json.loads(source.read_text());updated = replace_links(value)
                if updated == value:
                    continue
                content = json.dumps(updated, indent=2, allow_nan=False) + '\n'
            else:
                content = source.read_text()
                for old, new in mapping.items():
                    content = content.replace(old, new)
                if content == source.read_text():
                    continue
            original = backup / 'indexes' / source.relative_to(root)
            sha = copy_verified(source, original)
            manifest['index_backups'].append({'path': str(source), 'backup': str(original), 'sha256': sha})
            source.write_text(content)
    # Archive originals only after all canonical copies have passed validation.
    manifest['status'] = 'copied_verified_archiving_originals'
    write_json(backup / 'manifest.json', manifest)
    for row in manifest['files']:
        assert digest(row['canonical_path']) == row['sha256']
        original = Path(row['original_path']);target = Path(row['original_backup'])
        assert digest(original) == row['sha256'], f'Original changed during migration: {original}'
        target.parent.mkdir(parents=True, exist_ok=True)
        original.rename(target)
        assert digest(target) == row['sha256']
    # An unlabelled submission CSV cannot honestly be assigned to one checkpoint.
    for source in sorted((root / 'data/predictions').glob('*.csv')):
        target = root / 'data/runs/unassigned' / source.name
        sha = copy_verified(source, target)
        archived = backup / 'originals' / source.relative_to(root)
        archived.parent.mkdir(parents=True, exist_ok=True);source.rename(archived)
        manifest['unassigned_predictions'].append({'path': str(target), 'sha256': sha,
                                                  'original_backup': str(archived),
                                                  'reason': 'No recorded run ID/checkpoint association; not guessed.'})
    for folder in ('models', 'validation_outputs', 'predictions'):
        path = root / 'data' / folder
        if path.is_dir():
            metadata = path / '.DS_Store'
            if metadata.is_file():
                archived = backup / 'originals' / metadata.relative_to(root)
                archived.parent.mkdir(parents=True, exist_ok=True)
                sha = digest(metadata);metadata.rename(archived)
                manifest.setdefault('folder_metadata_backups', []).append({'path': str(archived), 'sha256': sha})
            if not any(path.iterdir()):
                path.rmdir()
    manifest['status'] = 'completed'
    write_json(backup / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--resume-migration', type=Path, help='Resume a verified pre-archive copy failure in its existing backup directory')
    args = parser.parse_args()
    if args.apply:
        result = migrate(resume_backup=args.resume_migration)
        print(json.dumps({'runs': len(result['run_ids']), 'artifacts': len(result['files']),
                          'manifest': str(Path(result['backup']) / 'manifest.json')}, indent=2))
    else:
        print(json.dumps({k: {n: str(p) for n, p in v.items()} for k, v in inventory().items()}, indent=2))


if __name__ == '__main__':
    main()
