# Augmented conventional ReLU — ten epochs

**Status / date:** Complete background run, October 30, 2026. Notebook 05 prepared, not executed.

**Change / controls:** Fresh `bespoke28` ReLU initialization, Adam LR 0.00075, ten epochs, zero weight decay, seed 42, batch 64, original split and upright validation unchanged. Same eleven lazy fixed training views: 0°, ±5°, ±10°, ±22°, ±45°, ±90°, bilinear/black fill. 5,775 updates/epoch; 57,750 total. Ten-percent warm-up is now 5,775 updates, then hold; the prior three-epoch run used 1,733. This is not continuation of its weights or schedule.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.650395 | 76.63% | 0.059417 | 98.31% |
| 2 | 0.089395 | 97.43% | 0.051858 | 98.61% |
| 3 | 0.058404 | 98.34% | 0.039912 | 98.99% |
| 4 | 0.045539 | 98.69% | 0.045115 | 99.04% |
| 5 | 0.037977 | 98.91% | 0.044929 | 98.93% |
| 6 | 0.034095 | 99.03% | 0.059560 | 99.00% |
| 7 | 0.031352 | 99.12% | 0.052888 | 98.87% |
| 8 | 0.026778 | 99.25% | 0.069440 | 98.23% |
| 9 | 0.026175 | 99.26% | 0.069751 | 98.48% |
| 10 | 0.027075 | 99.26% | 0.043935 | 99.04% |

**Selection / finding:** Minimum-loss checkpoint: epoch 3, loss 0.039912, accuracy 98.99%. Peak observed accuracy: 99.04% at epochs 4 and 10. Saved weights/logits correspond to minimum loss; other peak-epoch weights are not guaranteed retained. All later validation losses exceeded epoch 3 while training loss generally fell, an overfitting pattern; epoch 10 recovered but did not beat the minimum. The earlier three-epoch run remains better at 99.14%/loss 0.033286, with a different warm-up schedule. One seed, no Kaggle score.

**Run / time:** `20261030_145530_5b033b37`; 497.3 seconds.

**Artifacts / next:** `data/experiments/bespoke_lr_sweep_20261030_145529_03c5f864/` contains status/summary, full configuration/history references, ten-epoch plots, source snapshots and preservation receipt; paired weights/logits stay in `data/models/` and `data/validation_outputs/`. Initialization/order/split matched the earlier run's starting state; notebooks 03/04/05 and prior records were hash-verified unchanged during execution. To inspect without training, set notebook 05's `FOLLOWUP_DIRECTORY` to this group and execute only its final Results cell. Review before further experiments.
