"""Check optional warmup and sweep isolation without dataset training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.model.digit_model_helpers import linear_warmup_rate, seed_training, train_model, build_model
from scripts.utilities.bespoke_lr_sweep import order_fingerprint, run_paths, state_fingerprint, unique_run_id


class WarmupTests(unittest.TestCase):
    def test_step_endpoints_and_hold(self):
        self.assertEqual(linear_warmup_rate(.001, 0, 158), 0)
        self.assertEqual(linear_warmup_rate(.001, 1, 158), .001 / 158)
        self.assertEqual(linear_warmup_rate(.001, 158, 158), .001)
        self.assertEqual(linear_warmup_rate(.001, 1575, 158), .001)
        self.assertEqual(linear_warmup_rate(.001, 1, 0), .001)
        for args in ((0, 1, 2), (.1, -1, 2), (.1, 1, -2)):
            with self.assertRaises(ValueError): linear_warmup_rate(*args)

    def test_schedule_uses_global_update_count(self):
        rates = []
        original = torch.optim.Adam
        class RecordingAdam(original):
            def step(self, *args, **kwargs):
                rates.append(self.param_groups[0]['lr'])
                return super().step(*args, **kwargs)
        seed_training(42)
        loader = DataLoader(TensorDataset(torch.randn(8, 2), torch.arange(8) % 2), batch_size=2)
        with tempfile.TemporaryDirectory() as folder, patch('scripts.model.digit_model_helpers.get_training_device', return_value=torch.device('cpu')), patch('torch.optim.Adam', RecordingAdam):
            model, history, *_ = train_model(nn.Linear(2, 2), loader, loader, nn.CrossEntropyLoss(reduction='none'), 3, Path(folder)/'best.pt', learning_rate=.001, warmup_fraction=.5, weight_decay=0)
        self.assertEqual(len(rates), 12)
        np.testing.assert_allclose(rates, [.001 * min(step/6, 1) for step in range(1,13)])
        self.assertEqual(history[-1]['optimizer_steps_completed'], 12)
        self.assertEqual(history[1]['learning_rate_first'], rates[4])

    def test_matching_initialization_and_order_without_consuming_generator(self):
        fingerprints=[]
        for activation in ('relu','tanh'):
            seed_training(42)
            fingerprints.append(state_fingerprint(build_model('bespoke',activation=activation)))
        self.assertEqual(*fingerprints)
        loader=DataLoader(TensorDataset(torch.arange(20)),batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42))
        before=loader.generator.get_state().clone()
        first=order_fingerprint(loader)
        self.assertTrue(torch.equal(before,loader.generator.get_state()))
        self.assertEqual(first,order_fingerprint(loader))
        same=DataLoader(TensorDataset(torch.arange(20)),batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42))
        self.assertEqual(first,order_fingerprint(same))

    def test_unique_pair_paths_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);identifier=unique_run_id()
            first=run_paths('relu',identifier,root)
            second=run_paths('tanh',identifier,root)
            self.assertNotEqual(first,second)
            self.assertNotEqual(identifier,unique_run_id())
            first[0].parent.mkdir(parents=True);first[0].write_bytes(b'old')
            with self.assertRaises(FileExistsError):run_paths('relu',identifier,root)


if __name__ == '__main__':unittest.main()
