"""Prepared conventional-notebook controls and isolation; no training."""
from pathlib import Path
from tempfile import TemporaryDirectory
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch
from scripts.model import digit_experiment_helpers as experiments
from scripts.utilities import bespoke_lr_sweep as sweep


class ConventionalNotebookTests(unittest.TestCase):
    def test_conventional_model_forwarded_without_training(self):
        class FakeProcess:
            stdout=['Learning-rate testing results: /tmp/conventional-test\n']
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def wait(self):return 0
        with patch.object(sweep,'active_sweep_directories',return_value=[]),patch('subprocess.Popen',return_value=FakeProcess()) as launch:
            experiments.run_bespoke_sweep(epochs=3,learning_rates=(1e-6,1e-5,1e-4,1e-3),warmup_fraction=.1,weight_decay=0,model_name='bespoke28')
        command=launch.call_args.args[0]
        self.assertEqual(command[command.index('--model')+1],'bespoke28')
        self.assertNotIn('--resume-directory',command)

    def test_model_checkpoint_names_are_distinct(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            old=sweep.run_paths('relu','same_id',root,model_name='bespoke')
            new=sweep.run_paths('relu','same_id',root,model_name='bespoke28')
            self.assertNotEqual(old,new)
            self.assertTrue(new[0].name.startswith('bespoke28_relu_'))

    def test_cross_model_resume_rejected_before_mutating_old_group(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);directory=root/'data/experiments/old';directory.mkdir(parents=True)
            status=directory/'status.json';status.write_text(json.dumps({'model_name':'bespoke','status':'completed'}));before=status.read_bytes()
            with patch.object(sweep,'PROJECT_ROOT',root),patch.object(sweep,'MODEL_NAME','bespoke'),patch.object(sweep,'active_sweep_directories',return_value=[]),patch.object(sweep,'execute_sweep',side_effect=AssertionError('No training')) as train,patch.object(sys,'argv',['test','--run','--model','bespoke28','--resume-directory',str(directory)]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):sweep.main()
                train.assert_not_called()
            self.assertEqual(status.read_bytes(),before)
            self.assertFalse((directory/'sweep.log').exists())


if __name__=='__main__':unittest.main()
