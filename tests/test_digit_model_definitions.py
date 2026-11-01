"""CPU-only forward and compatibility checks; no model training."""
import io
import unittest
import torch
from torch import nn
from scripts.model import digit_model_definitions as definitions
from scripts.model.digit_model_helpers import build_model


class Bespoke28Tests(unittest.TestCase):
    def test_shape_schedule_for_both_activations_without_resize(self):
        for activation in ('relu','tanh'):
            with self.subTest(activation=activation):
                model=build_model('bespoke28',activation=activation)
                observed=[];handles=[]
                for stage in model.stages:
                    for layer in (stage[0],stage[2],stage[4]):
                        handles.append(layer.register_forward_hook(lambda module,inputs,output:observed.append(tuple(output.shape[1:]))))
                with torch.no_grad():output=model(torch.zeros(2,1,28,28))
                for handle in handles:handle.remove()
                expected=[]
                for channels,size,pool in ((16,28,14),(32,14,7),(64,7,4),(128,4,2),(256,2,1)):
                    expected.extend([(channels,size,size),(channels,size,size),(channels,pool,pool)])
                self.assertEqual(observed,expected)
                self.assertEqual(tuple(output.shape),(2,10))
                self.assertTrue(torch.isfinite(output).all())
                self.assertFalse(any(isinstance(layer,nn.Upsample) for layer in model.modules()))

    def test_state_dict_roundtrip_and_factory_compatibility(self):
        model=definitions.BespokeCNN28(activation='tanh');buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        clone=build_model('bespoke28',activation='tanh');clone.load_state_dict(torch.load(buffer,weights_only=True))
        images=torch.zeros(1,1,28,28)
        with torch.no_grad():self.assertTrue(torch.equal(model(images),clone(images)))
        self.assertIsInstance(build_model(),definitions.LeNet)
        self.assertIsInstance(build_model('bespoke'),definitions.BespokeCNN)

    def test_invalid_activation_and_input_are_rejected(self):
        with self.assertRaises(ValueError):definitions.BespokeCNN28(activation='invalid')
        with self.assertRaises(ValueError):definitions.BespokeCNN28()(torch.zeros(1,1,224,224))


if __name__=='__main__':unittest.main()
