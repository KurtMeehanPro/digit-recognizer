"""Verify notebook delegation and legacy imports without launching training."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts.model import digit_experiment_helpers as experiments
from scripts.model import digit_model_helpers as helpers
from scripts.model import digit_model_definitions as definitions


class NotebookDelegationTests(unittest.TestCase):
    def test_training_controls_are_forwarded_to_existing_runner(self):
        directory = Path('/tmp/controlled-sweep-test')
        class FakeProcess:
            stdout = [f'SWEEP_DIRECTORY {directory}\n']
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def wait(self): return 0
        with patch('scripts.utilities.bespoke_lr_sweep.active_sweep_directories', return_value=[]), patch('subprocess.Popen', return_value=FakeProcess()) as launch:
            selected = experiments.run_bespoke_sweep(epochs=3, learning_rates=(1e-6,1e-5,1e-4,1e-3), warmup_fraction=.1, weight_decay=0, sweep_directory=directory)
        command=launch.call_args.args[0]
        self.assertEqual(command[0],sys.executable)
        for flag, expected in [('--epochs','3'),('--warmup-fraction','0.1'),('--weight-decay','0')]:
            self.assertEqual(command[command.index(flag)+1],expected)
        start=command.index('--learning-rates')+1
        self.assertEqual(command[start:start+4],['1e-06','1e-05','0.0001','0.001'])
        self.assertEqual(selected,directory)

    def test_running_sweep_blocks_notebook_launch(self):
        with patch('scripts.utilities.bespoke_lr_sweep.active_sweep_directories', return_value=[Path('/tmp/active')]), patch('subprocess.Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'no duplicate'):
                experiments.run_bespoke_sweep(epochs=3, learning_rates=(1e-6,), warmup_fraction=.1, weight_decay=0)
            launch.assert_not_called()

    def test_legacy_helper_exports_remain_compatible(self):
        self.assertIs(helpers.LeNet,definitions.LeNet)
        self.assertIs(helpers.BespokeCNN,definitions.BespokeCNN)
        self.assertIsInstance(helpers.build_model('lenet'),definitions.LeNet)
        self.assertIsInstance(helpers.build_model('bespoke'),definitions.BespokeCNN)


if __name__ == '__main__': unittest.main()
