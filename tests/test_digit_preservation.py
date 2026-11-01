"""Regression checks for immutable artifacts versus editable notebook displays."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from scripts.utilities.bespoke_lr_sweep import capture_protected_files, verify_protected_files


class PreservationTests(unittest.TestCase):
    def make_files(self, root):
        for name in ('data/models/old_best.pt','data/validation_outputs/old_validation.pt',
                     'notebooks/02_lenet_style_cnn.ipynb','notebooks/03_bespoke_upsampling_cnn.ipynb',
                     'docs/evaluations/notes.md'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original')

    def test_benign_notebook03_autosave_and_notes_edit_do_not_fail(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);self.make_files(root);fingerprints=capture_protected_files(root)
            (root/'notebooks/03_bespoke_upsampling_cnn.ipynb').write_text('{"cells":[{"outputs":["epoch 3"]}]}')
            (root/'docs/evaluations/notes.md').write_text('Updated presentation notes')
            verify_protected_files(fingerprints)

    def test_prior_weight_validation_or_notebook02_mutation_still_fails(self):
        for name in ('data/models/old_best.pt','data/validation_outputs/old_validation.pt',
                     'notebooks/02_lenet_style_cnn.ipynb'):
            with self.subTest(name=name),TemporaryDirectory() as folder:
                root=Path(folder);self.make_files(root);fingerprints=capture_protected_files(root)
                (root/name).write_bytes(b'changed')
                with self.assertRaisesRegex(AssertionError,'Protected file changed'):
                    verify_protected_files(fingerprints)

    def test_captured_model_source_snapshot_remains_protected(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'source_snapshots/model.py';path.parent.mkdir();path.write_text('original model')
            fingerprints={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
            path.write_text('changed model')
            with self.assertRaisesRegex(AssertionError,'Protected file changed'):
                verify_protected_files(fingerprints)


if __name__=='__main__':unittest.main()
