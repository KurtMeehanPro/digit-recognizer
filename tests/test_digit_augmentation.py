"""Rotation preparation tests: synthetic CPU inputs only, no training."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS
from scripts.model import digit_model_helpers as models
from scripts.model import digit_experiment_helpers as experiments

class AugmentationTests(unittest.TestCase):
    def test_rotation_reproducible_shape_dtype_range_and_separate_rng(self):
        image=torch.zeros(1,28,28);image[:,5:24,9:13]=1
        data=TensorDataset(image[None],torch.tensor([3]))
        a=models.RandomRotationDataset(data,10,42);b=models.RandomRotationDataset(data,10,42)
        state=torch.get_rng_state().clone()
        for _ in range(3):
            x,y=a[0];z,w=b[0]
            self.assertTrue(torch.equal(x,z));self.assertEqual(y,w)
            self.assertEqual(x.shape,(1,28,28));self.assertEqual(x.dtype,torch.float32)
            self.assertTrue(torch.isfinite(x).all());self.assertGreaterEqual(x.min().item(),0);self.assertLessEqual(x.max().item(),1)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(data[0][0],image))
        self.assertIs(models.training_dataset(data,0,42),data)
        with self.assertRaises(ValueError):models.training_dataset(data,float('nan'))

    def test_train_only_validation_and_order_unchanged(self):
        rng=np.random.default_rng(4)
        frame=pd.DataFrame(rng.integers(0,256,(100,784)),columns=PIXEL_COLUMNS)
        frame['label']=np.tile(np.arange(10),10)
        baseline,val=models.build_data_loaders(frame)
        rotated,rotval=models.build_data_loaders(frame,rotation_degrees=5,augmentation_seed=42)
        self.assertIsInstance(rotated.dataset,models.RandomRotationDataset)
        self.assertIsInstance(rotval.dataset,TensorDataset)
        for a,b in zip(val.dataset.tensors,rotval.dataset.tensors):self.assertTrue(torch.equal(a,b))
        for a,b in zip(baseline.dataset.tensors,rotated.dataset.dataset.tensors):self.assertTrue(torch.equal(a,b))
        self.assertEqual(list(baseline.sampler),list(rotated.sampler))

    def test_proposed_trial_configuration_forwarding_without_launch(self):
        with patch.object(experiments,'run_bespoke_sweep',side_effect=['/tmp/a','/tmp/b','/tmp/c']) as run:
            selected=experiments.run_rotation_experiment(rotation_ranges=(0,5,10),activation='tanh')
        self.assertEqual(list(selected),[0,5,10]);self.assertEqual(len(set(selected.values())),3)
        for degrees,call in zip((0,5,10),run.call_args_list):
            kw=call.kwargs
            self.assertEqual(kw['rotation_degrees'],degrees);self.assertEqual(kw['activations'],('tanh',))
            self.assertEqual(kw['learning_rates'],(.0001,));self.assertEqual(kw['weight_decay'],0)
            self.assertEqual(kw['augmentation_seed'],42);self.assertEqual(kw['model_name'],'bespoke28')

    def test_cli_forwarding_without_process(self):
        class Fake:
            stdout=['Learning-rate testing results: /tmp/mock\n']
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def wait(self):return 0
        with patch('scripts.utilities.bespoke_lr_sweep.active_sweep_directories',return_value=[]),patch('subprocess.Popen',return_value=Fake()) as process:
            experiments.run_bespoke_sweep(epochs=3,learning_rates=(.001,),warmup_fraction=.1,weight_decay=0,model_name='bespoke28',activations=('relu',),rotation_degrees=5,augmentation_seed=42)
        command=process.call_args.args[0]
        for option,value in [('--rotation-degrees','5'),('--augmentation-seed','42'),('--activations','relu')]:self.assertEqual(command[command.index(option)+1],value)

    def test_fixed_eleven_views_index_labels_center_and_determinism(self):
        angles=(0,-5,5,-10,10,-22,22,-45,45,-90,90)
        images=torch.zeros(2,1,28,28);images[0,0,5,13]=1;images[1,0,12:16,12:16]=1
        originals=TensorDataset(images,torch.tensor([3,8]))
        views=models.FixedRotationDataset(originals,angles)
        self.assertEqual(len(views),22)
        for index in range(22):
            a,label=views[index];b,other=views[index]
            self.assertTrue(torch.equal(a,b));self.assertEqual(label,originals[index//11][1])
            self.assertEqual(a.shape,(1,28,28));self.assertEqual(a.dtype,torch.float32)
            self.assertGreaterEqual(a.min().item(),0);self.assertLessEqual(a.max().item(),1)
        self.assertTrue(torch.equal(views[0][0],images[0]))
        self.assertTrue(torch.equal(views[11][0],images[1]))
        # Right-angle rotation about center maps pixel (5,13) to (14,5).
        self.assertTrue(torch.allclose(views[10][0],torch.rot90(images[0],1,(-2,-1)),atol=2e-6))
        with self.assertRaises(IndexError):views[22]
        with self.assertRaises(ValueError):models.FixedRotationDataset(originals,(0,0))
        config=models.rotation_configuration(angles=angles)
        self.assertEqual(config['views_per_original'],11);self.assertEqual(config['fill'],0)
        self.assertTrue(config['split_before_augmentation']);self.assertTrue(config['training_only'])

    def test_fixed_split_isolation_and_configuration_forwarding(self):
        frame=pd.DataFrame(np.zeros((100,784)),columns=PIXEL_COLUMNS);frame['label']=np.tile(np.arange(10),10)
        base,val=models.build_data_loaders(frame)
        views,augval=models.build_data_loaders(frame,rotation_angles=(0,-90,90))
        self.assertEqual(len(views.dataset),3*len(base.dataset))
        for a,b in zip(val.dataset.tensors,augval.dataset.tensors):self.assertTrue(torch.equal(a,b))
        for a,b in zip(base.dataset.tensors,views.dataset.dataset.tensors):self.assertTrue(torch.equal(a,b))
        class Fake:
            stdout=['Learning-rate testing results: /tmp/mock\n']
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def wait(self):return 0
        with patch('scripts.utilities.bespoke_lr_sweep.active_sweep_directories',return_value=[]),patch('subprocess.Popen',return_value=Fake()) as process:
            experiments.run_bespoke_sweep(epochs=3,learning_rates=(.0001,),warmup_fraction=.1,weight_decay=0,model_name='bespoke28',activations=('tanh',),rotation_angles=(0,-90,90))
        command=process.call_args.args[0];i=command.index('--rotation-angles')
        self.assertEqual(command[i+1:i+4],['0','-90','90'])

    def test_prepared_notebook_syntax_and_metadata(self):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/05_bespoke_conventional_performance_cnn.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),'<notebook05>','exec')
        config=nb['metadata']['digit_experiment']['configuration']
        self.assertEqual(config['rotation_angles'],[0,-5,5,-10,10,-22,22,-45,45,-90,90]);self.assertEqual(config['augmentation_protocol'],'fixed_lazy_views')
        self.assertEqual(config['activation_order'],['tanh','relu'])
        self.assertEqual(config['relu_rates'],[.0005,.00075,.001,.00125,.0015])
        self.assertEqual(nb['metadata']['digit_experiment']['state'],'prepared_not_run')
        self.assertEqual(config['weight_decay'],0)

if __name__=='__main__':unittest.main()
