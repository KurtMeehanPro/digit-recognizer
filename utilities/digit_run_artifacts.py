"""Per-run storage and compatibility readers for normal Digit Recognizer runs.

Raw inputs and multi-run comparisons stay shared. Historical serialized records
are never rewritten: readers resolve the adjacent canonical checkpoint instead.
"""
import json
from pathlib import Path

import torch

from scripts.dataset.digit_dataset_config import PROJECT_ROOT


def run_directory(run_id, root=PROJECT_ROOT):
    if not run_id or Path(run_id).name != run_id or run_id in {'.', '..', 'unassigned', 'experiments'}:
        raise ValueError('A run ID must be one directory name')
    return Path(root) / 'data/runs' / run_id


def allocate_run(run_id, root=PROJECT_ROOT):
    """Reserve a fresh record; never reuse a partially populated run directory."""
    directory = run_directory(run_id, root)
    directory.mkdir(parents=True, exist_ok=False)
    write_run_readme(directory, status='allocated')
    return directory / 'checkpoint.pt', directory / 'output.pt'


def validation_path(path):
    path = Path(path)
    if path.name in {'unassigned', 'experiments'} or path.parent.name in {'unassigned', 'experiments'}:
        raise ValueError('This is a reserved container, not a run')
    return path / 'output.pt' if path.is_dir() else path


def current_experiment_path(path, root=PROJECT_ROOT):
    """Resolve the old shared-group prefix without rewriting historical tensors."""
    path = Path(path)
    try:
        relative = path.relative_to(Path(root) / 'data/experiments')
    except ValueError:
        return path
    return Path(root) / 'data/runs/experiments' / relative


def load_run(path, root=PROJECT_ROOT):
    """Read canonical directories/files or legacy files without changing bytes."""
    path = validation_path(path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    saved = dict(saved)
    adjacent = path.resolve().parent / 'checkpoint.pt'
    if adjacent.is_file():
        saved['checkpoint_path'] = str(adjacent)
    if saved.get('sweep_directory'):
        saved['sweep_directory'] = str(current_experiment_path(saved['sweep_directory'], root))
    saved.setdefault('model_config', {'model_name': 'lenet'})
    return saved


def iter_validation_paths(root=PROJECT_ROOT):
    """Discover each record once, preferring canonical files over legacy aliases."""
    root = Path(root)
    seen = set()
    paths = [*sorted(root.glob('data/runs/*/output.pt')),
             *sorted(root.glob('data/validation_outputs/*.pt'))]
    for path in paths:
        if path.parent.name in {'unassigned', 'experiments'}:
            continue
        saved = load_run(path, root=root)
        identity = saved.get('run_id', str(path.resolve()))
        if identity not in seen:
            seen.add(identity)
            yield path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_run_record(directory, artifact, *, status=None):
    """Export readable settings/history beside the byte-preserved tensor artifact."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'config.json', {
        'model_config': artifact.get('model_config', {'model_name': 'lenet'}),
        'training_config': artifact.get('training_config', {}),
        'seed': artifact.get('seed'),
        'validation_fraction': artifact.get('validation_fraction'),
        'environment': artifact.get('environment', {}),
    })
    history = artifact.get('history', [])
    write_json(directory / 'history.json', history)
    write_json(directory / 'record.json', {
        'run_id': artifact['run_id'],
        'experiment_group': artifact.get('sweep_directory'),
    })
    write_run_readme(directory, artifact, status=status)


def write_run_readme(directory, artifact=None, status=None):
    """Write a short file guide from facts already present in one run record."""
    directory = Path(directory)
    readme_path = directory / 'README.md'
    current = readme_path.read_text() if readme_path.is_file() else ''
    human_notes = current.split('\n## Notes\n', 1)[1].strip() if '\n## Notes\n' in current else ''
    if artifact is None and (directory / 'output.pt').is_file():
        artifact = load_run(directory)
    artifact = artifact or {}
    labels = {'completed': 'Completed', 'allocated': 'Prepared; no result saved yet',
              'checkpoint_only_incomplete': 'Incomplete: checkpoint only',
              'missing_checkpoint': 'Incomplete: checkpoint missing',
              'interrupted_or_failed': 'Interrupted or failed'}
    known_status = next((line for line in current.splitlines() if line.startswith('**Status:** ')), None)
    lines = [f"# Run {directory.name}", '']
    if status is not None:
        lines += ['**Status:** ' + labels.get(status, status.replace('_', ' ')) + '.', '']
    elif known_status:
        lines += [known_status, '']

    if (directory / 'output.pt').is_file():
        count = len(artifact['labels']) if 'labels' in artifact else None
        text = '`output.pt` contains held-out validation logits (raw class scores), true labels and available run metadata.'
        if count is not None:
            text += f" {count:,} validation examples"
            if artifact.get('seed') is not None:
                text += f"; split seed {artifact['seed']}"
            if artifact.get('validation_fraction') is not None:
                text += f"; validation fraction {artifact['validation_fraction']:.0%}"
            text += '.'
        epoch = artifact.get('best_epoch')
        if epoch is not None:
            text += f" Saved checkpoint/output selection: minimum validation loss, epoch {epoch}."
        lines += [text, '']
    else:
        lines += ['No paired output has been saved for this record.', '']
    descriptions = {
        'checkpoint.pt': 'Saved model weights.',
        'output.pt': 'Validation logits, labels and run metadata.',
        'config.json': 'Available model and training settings.',
        'history.json': 'Per-epoch metrics in JSON.',
        'record.json': 'Run ID, experiment group and any retained error/note.',
        'run.log': 'Captured training progress.',
        'predictions.csv': 'Separately generated prediction rows.',
        'summary.json': 'Saved error-analysis summary.',
        'per_digit_metrics.csv': 'Per-digit validation metrics.',
        'confusion_matrix.png': 'Validation confusion matrix.',
        'all_digit_misses.png': 'Compact overview of validation mistakes.',
    }
    history_path = directory / 'history.json'
    no_history = history_path.is_file() and not json.loads(history_path.read_text())
    if no_history:
        descriptions['history.json'] = 'No per-epoch history was retained (empty array).'
    lines += ['| File | Contents |', '| --- | --- |']
    listed = set()
    for name, meaning in descriptions.items():
        if (directory / name).is_file():
            lines.append(f'| [{name}]({name}) | {meaning} |')
            listed.add(name)
    notes = sorted(p for p in directory.glob('*.md') if p.name not in {'README.md'})
    if notes:
        lines += ['', '**Related notes:** ' + ', '.join(f'[{p.name}]({p.name})' for p in notes) + '.']
        listed.update(p.name for p in notes)
    extra = [p for p in directory.iterdir() if p.is_file() and p.name not in listed | {'README.md'}]
    if extra:
        from collections import Counter
        counts = Counter(p.suffix or '(no extension)' for p in extra)
        lines += ['', 'Additional files in this folder: ' + ', '.join(f'{n} {suffix}' for suffix, n in sorted(counts.items())) + '.']
    missing = []
    if not (directory / 'run.log').is_file():
        missing.append('captured training log')
    if not history_path.is_file() or no_history:
        missing.append('per-epoch history')
    if missing:
        lines += ['', 'Unavailable in this record: ' + ', '.join(missing) + '.']
    if human_notes:
        lines += ['', '## Notes', '', human_notes]
    readme_path.write_text('\n'.join(lines) + '\n')
