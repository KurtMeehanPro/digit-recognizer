"""Notebook05 serial ordering and explicit artifact reuse, without training."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from scripts.model import digit_experiment_helpers as experiments
from scripts.model import digit_model_helpers as models

class AugmentedNotebookPlanTests(unittest.TestCase):
    def test_matching_saved_runs_reused_in_tanh_then_relu_order(self):
        angles=(0,-5,5,-10,10,-22,22,-45,45,-90,90)
        plan=[('tanh',.0001),('relu',.0005),('relu',.00075),('relu',.001),('relu',.00125),('relu',.0015)]
        with TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'weights.pt';checkpoint.touch()
            rows=[];artifacts={}
            for index,(activation,rate) in enumerate(reversed(plan)):
                path=str(root/f'{index}.pt')
                row={'activation':activation,'target_learning_rate':rate,'run_id':str(index),'val_accuracy':.98,'validation_outputs':path}
                rows.append(row)
                artifacts[path]={'model_config':{'model_name':'bespoke28','activation':activation},'checkpoint_path':str(checkpoint),
                    'training_config':{'epochs':3,'seed':42,'batch_size':64,'validation_fraction':.2,'optimizer':'Adam','weight_decay':0,
                    'augmentation':models.rotation_configuration(0,42,angles),'checkpoint_selection':'lowest_validation_loss','schedule':{'fraction':.1}}}
            (root/'status.json').write_text(json.dumps({'completed_runs':rows}))
            with patch.object(experiments.torch,'load',side_effect=lambda path,**kw:artifacts[path]),patch.object(experiments,'run_bespoke_sweep',side_effect=AssertionError('No training expected')) as train,contextlib.redirect_stdout(io.StringIO()):
                selected=experiments.run_augmented_comparison(tanh_rate=.0001,relu_rates=(.0005,.00075,.001,.00125,.0015),rotation_angles=angles,saved_groups=(root,))
                train.assert_not_called()
            self.assertEqual([(v['activation'],v['target_learning_rate']) for v in selected],plan)
            self.assertTrue(all(v['reused'] for v in selected))

    def test_fresh_ignores_saved_groups_and_forwards_serial_configuration(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'status.json').write_text(json.dumps({'completed_runs':[{'run_id':'fake'}]}))
            with patch.object(experiments,'run_bespoke_sweep',return_value=root) as train:
                selected=experiments.run_augmented_comparison(tanh_rate=.0001,relu_rates=(.0005,.0015),rotation_angles=(0,-90,90),fresh=True,saved_groups=('/missing',))
            self.assertEqual([(c.kwargs['activations'],c.kwargs['learning_rates']) for c in train.call_args_list],[(('tanh',),(.0001,)),(('relu',),(.0005,)),(('relu',),(.0015,))])
            self.assertTrue(all(c.kwargs['rotation_angles']==(0,-90,90) for c in train.call_args_list))
            self.assertTrue(all(not row['reused'] for row in selected))

    def test_ten_epoch_followup_prepared_after_comparison(self):
        import math
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'notebooks/05_bespoke_conventional_performance_cnn.ipynb').read_text())
        sources=[''.join(cell['source']) for cell in nb['cells']]
        compare=next(i for i,text in enumerate(sources) if 'SELECTED_RUNS = experiments.run_augmented_comparison' in text)
        follow=next(i for i,text in enumerate(sources) if 'FOLLOWUP_DIRECTORY = experiments.run_bespoke_sweep' in text)
        self.assertGreater(follow,compare)
        self.assertIn('epochs=FOLLOWUP_EPOCHS',sources[follow])
        c=nb['metadata']['digit_experiment']['configuration']['ten_epoch_followup']
        self.assertEqual(c['epochs'],10);self.assertTrue(c['fresh_initialization'])
        self.assertEqual(c['warmup_steps'],math.ceil(.1*10*5775))
        self.assertEqual(c['total_optimizer_steps'],57750)
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),'<05>','exec')
