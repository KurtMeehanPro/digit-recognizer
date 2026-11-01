"""Independent treatments and unchanged defaults; tiny synthetic CPU checks."""
import copy,contextlib,io,json,sys,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from scripts.model import digit_model_helpers as h
from scripts.model import digit_experiment_helpers as e
from scripts.utilities import bespoke_lr_sweep as sweep

class RegularizationTests(unittest.TestCase):
    def test_zero_defaults_exact_original_adam_update(self):
        torch.manual_seed(9);a=torch.nn.Linear(2,2);b=copy.deepcopy(a)
        oa=h.build_training_optimizer(a,.001);ob=torch.optim.Adam(b.parameters(),lr=.001,weight_decay=0)
        x=torch.tensor([[.3,.7]]);y=torch.tensor([1])
        for model,opt in [(a,oa),(b,ob)]:
            torch.nn.functional.cross_entropy(model(x),y).backward();opt.step()
        for av,bv in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(av,bv))

    def test_l2_penalty_gradient_and_bias_exclusion(self):
        m=torch.nn.Linear(2,1);m.weight.data.fill_(2);m.bias.data.fill_(7)
        penalty=h.explicit_l2_penalty(m,.1);self.assertAlmostEqual(penalty.item(),.4,places=6)
        penalty.backward();self.assertTrue(torch.allclose(m.weight.grad,torch.full_like(m.weight,.2)))
        self.assertIsNone(m.bias.grad)

    def test_decoupled_decay_independent_of_explicit_l2(self):
        m=torch.nn.Linear(2,1);m.weight.data.fill_(2);m.bias.data.fill_(7)
        opt=h.build_training_optimizer(m,.01,decoupled_weight_decay=.1)
        self.assertIsInstance(opt,torch.optim.AdamW)
        for p in m.parameters():p.grad=torch.zeros_like(p)
        opt.step()
        self.assertTrue(torch.allclose(m.weight,torch.full_like(m.weight,2*(1-.01*.1))))
        self.assertEqual(m.bias.item(),7)
        for value in [-1,float('nan')]:
            with self.assertRaises(ValueError):h.regularization_configuration(value,0)

    def test_total_objective_separate_from_data_and_validation_loss(self):
        torch.manual_seed(3);m=torch.nn.Linear(2,2)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([0,1])),batch_size=2)
        with TemporaryDirectory() as folder,patch.object(h,'get_training_device',return_value=torch.device('cpu')),contextlib.redirect_stdout(io.StringIO()):
            _,history,_,_,_=h.train_model(m,loader,loader,torch.nn.CrossEntropyLoss(reduction='none'),1,Path(folder)/'m.pt',l2_strength=.01,decoupled_weight_decay=.01)
        row=history[0];self.assertGreater(row['train_l2_penalty'],0)
        self.assertAlmostEqual(row['train_objective'],row['train_loss']+row['train_l2_penalty'])
        with torch.no_grad():expected=torch.nn.functional.cross_entropy(m(loader.dataset.tensors[0]),loader.dataset.tensors[1]).item()
        self.assertAlmostEqual(row['val_loss'],expected,places=6)

    def test_cli_forwarding_and_mismatched_resume_rejected(self):
        class Fake:
            stdout=['Learning-rate testing results: /tmp/mock\n']
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def wait(self):return 0
        with patch.object(sweep,'active_sweep_directories',return_value=[]),patch('subprocess.Popen',return_value=Fake()) as process:
            e.run_bespoke_sweep(epochs=10,learning_rates=(.00075,),warmup_fraction=.1,weight_decay=0,l2_strength=1e-5,decoupled_weight_decay=.001)
        command=process.call_args.args[0]
        self.assertEqual(command[command.index('--l2-strength')+1],'1e-05')
        self.assertEqual(command[command.index('--decoupled-weight-decay')+1],'0.001')
        with TemporaryDirectory() as folder:
            root=Path(folder);d=root/'data/experiments/old';d.mkdir(parents=True);p=d/'status.json';p.write_text(json.dumps({'model_name':'bespoke28','status':'completed'}));before=p.read_bytes()
            with patch.object(sweep,'PROJECT_ROOT',root),patch.object(sweep,'active_sweep_directories',return_value=[]),patch.object(sys,'argv',['test','--run','--model','bespoke28','--resume-directory',str(d),'--l2-strength','1e-5']),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):sweep.main()
            self.assertEqual(p.read_bytes(),before)

    def test_notebook_preserves_first_three_and_appends_independent_arms(self):
        root=Path(__file__).resolve().parents[1];nb=json.loads((root/'notebooks/05_bespoke_conventional_performance_cnn.ipynb').read_text())
        config=nb['metadata']['digit_experiment']['configuration']
        self.assertEqual(config['relu_rates'],[.0005,.00075,.001,.00125,.0015])
        self.assertEqual(config['ten_epoch_followup']['epochs'],10)
        arms=nb['metadata']['digit_experiment']['configuration']['regularization_arms']
        self.assertEqual([(x['l2_strength'],x['decoupled_weight_decay']) for x in arms],[(1e-5,0),(0,.001),(1e-5,.001)])
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<05>','exec')
