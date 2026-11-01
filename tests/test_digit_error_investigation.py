"""Saved-validation alignment and honest galleries, no model execution."""
import hashlib,json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS
from scripts.model import digit_model_metrics as m

class ErrorInvestigationTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(8);frame=pd.DataFrame(rng.integers(0,256,(100,784)),columns=PIXEL_COLUMNS);frame['label']=np.tile(np.arange(10),10)
        _,indices=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,stratify=frame.label.to_numpy())
        images=torch.tensor(frame.iloc[indices][PIXEL_COLUMNS].to_numpy(),dtype=torch.float32).reshape(-1,1,28,28)/255
        labels=torch.tensor(frame.iloc[indices].label.to_numpy())
        artifact={'dataset_sha256':hashlib.sha256(frame[['label',*PIXEL_COLUMNS]].to_numpy().tobytes()).hexdigest(),'validation_fraction':.2,'seed':42,'labels':labels,'validation_sha256':hashlib.sha256(images.numpy().tobytes()+labels.numpy().tobytes()).hexdigest(),'logits':torch.zeros(len(labels),10)}
        return frame,artifact,indices,images
    def test_exact_source_indices_and_upright_alignment(self):
        frame,a,expected,original=self.fixture();images,labels,indices=m.aligned_saved_validation(a,frame)
        self.assertTrue(np.array_equal(indices,expected));self.assertTrue(torch.equal(images,original))
        self.assertTrue(torch.equal(labels,a['labels']));self.assertEqual(len(set(indices)),len(indices))
        a['labels']=a['labels'].roll(1)
        with self.assertRaises(ValueError):m.aligned_saved_validation(a,frame)
    def test_dataset_change_or_nonfinite_predictions_rejected(self):
        frame,a,_,_=self.fixture();a['logits'][0,0]=float('nan')
        with self.assertRaises(ValueError):m.aligned_saved_validation(a,frame)
        frame.loc[0,'pixel0']+=1
        with self.assertRaises(ValueError):m.aligned_saved_validation(a,frame)
    def test_zero_support_no_miss_and_confident_wrong_counts(self):
        logits=torch.tensor([[9.,0,0,0,0,0,0,0,0,0],[0.,9,0,0,0,0,0,0,0,0]])
        labels=torch.tensor([0,2]);matrix,metrics,predictions,errors,pairs,bins=m.build_error_tables(logits,labels,np.array([7,12]))
        self.assertEqual(int(matrix.sum()),2);self.assertEqual(metrics.loc[9,'Support'],0)
        self.assertEqual(metrics.loc[9,'F1'],0);self.assertEqual(metrics.loc[0,'Errors'],0)
        self.assertEqual(errors.validation_index.tolist(),[1]);self.assertEqual(errors.csv_line_number.tolist(),[14])
        self.assertEqual(pairs.iloc[0].to_dict(),{'true_digit':2,'predicted_digit':1,'errors':1})
        self.assertEqual(int(bins['count'].sum()),2);self.assertEqual(int(bins.errors.sum()),1)
        fig=m.plot_error_gallery(torch.zeros(2,1,28,28),errors.iloc[:0],'No misses')
        self.assertEqual(sum(len(ax.images) for ax in fig.axes),0)
        import matplotlib.pyplot as plt
        plt.close(fig)
        with self.assertRaises(ValueError):m.plot_error_gallery(torch.zeros(2,1,28,28),pd.concat([errors,errors]),'Duplicates')
    def test_appended_notebook_preserves_existing_cells(self):
        root=Path(__file__).resolve().parents[1];nb=json.loads((root/'notebooks/05_bespoke_conventional_performance_cnn.ipynb').read_text())
        last=nb['cells'][-1];source=''.join(last['source'])
        self.assertIn('20261030_143937_9823b3b6',source);self.assertIn('max_per_digit=8',source)
        compile(source,'<saved-errors>','exec')
