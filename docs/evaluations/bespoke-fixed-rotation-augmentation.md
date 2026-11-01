# Bespoke Conventional CNN — fixed rotation augmentation

**Status / date:** Tanh and ReLU background tests complete, October 30, 2026; notebook 05 remains prepared for tanh and unexecuted.

**Change / controls:** Conventional `bespoke28`; tanh Adam LR 1e-4, ReLU Adam LR 1e-3; zero weight decay, seed 42, batch 64, three epochs on MPS. Split before augmentation: 33,600 training originals; 8,400 upright validation examples unchanged. Eleven lazy fixed views per original: 0°, ±5°, ±10°, ±22°, ±45°, ±90°. Rotation about image center; bilinear interpolation, black fill 0.0, unchanged 28×28 frame. Labels retained; large rotations may crop strokes or create ambiguity. No random angle sampling or eleven stored datasets.

**Budget:** 369,600 logical examples/epoch; 5,775 updates/epoch, 17,325 total (11× baseline). Linear warm-up for 1,733 updates, then hold. Completed in 149.6 seconds (tanh), 149.7 seconds (ReLU).

| Setup | Selected epoch | Validation loss | Accuracy |
| --- | ---: | ---: | ---: |
| Saved tanh baseline, no augmentation | 3 | 0.097029 | 97.26% |
| Tanh, eleven fixed views | 3 | 0.045203 | 98.68% |
| Saved ReLU baseline, no augmentation | 3 | 0.066391 | 98.10% |
| ReLU, eleven fixed views | 2 | 0.046218 | 98.76% |

**Finding:** Tanh improved by 1.42 percentage points; ReLU by 0.67. Selected ReLU classified 8,296/8,400 correctly versus tanh's 8,289 (seven images). Tanh had slightly lower selected loss. ReLU epoch 3 reached 99.00% accuracy but loss rose to 0.046300; the existing lowest-loss rule restored epoch 2, so the saved checkpoint/logits represent 98.76%, not 99.00%. Full epoch metrics remain in history. These single-seed results combine augmentation with 11× more updates and do not isolate augmentation's contribution. No Kaggle score or weight decay test.

**Run IDs:** Tanh `20261030_142314_039b2794`, baseline `20261030_131420_eaeeb7fc`; ReLU `20261030_143050_495b4a16`, baseline `20261030_131433_21578689`. Both baselines reused, not retrained.

**ReLU artifacts:** `data/experiments/bespoke_lr_sweep_20261030_143049_e5322aac/`, paired `bespoke28_relu_20261030_143050_495b4a16` checkpoint/validation outputs. Same expanded sampler order, initial weights, angles and upright validation were verified against augmented tanh. Notebook 05 remains configured for tanh.

**Tanh artifacts / next:** `data/experiments/bespoke_lr_sweep_20261030_142313_4cfd332a/` contains status, summary, histories/config references, plots, source snapshots and preservation verification. Paired `bespoke28_tanh_20261030_142314_039b2794` weights/logits remain under `data/models/` and `data/validation_outputs/`. Select this group in notebook 05's `SWEEP_DIRECTORY` and run only Results to inspect without training. Notebooks 03/04/05 and prior experiment records were hash-verified unchanged during execution. Review before further experiments; notebook 06 remains an idea.

[Follow-up: five-rate augmented ReLU comparison](bespoke-augmented-relu-finer-rates.md).
