# Augmented conventional ReLU — finer learning rates

**Status / date:** Four new background runs complete; center reused. October 30, 2026.

**Change / controls:** ReLU `bespoke28`; targets ±50%/±25% around 0.001. Same eleven lazy fixed views (0°, ±5°, ±10°, ±22°, ±45°, ±90°), training only; upright validation unchanged. Seed 42, split 80/20, batch 64, Adam zero weight decay, three epochs, MPS. Bilinear/black fill; 369,600 examples per epoch, 17,325 updates, 1,733 warm-up then hold. Initialization/order/split fingerprints matched all five runs.

| Target LR | Selected accuracy | Selected loss | Selected epoch | Peak observed accuracy | Peak epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0005 | 99.04% | 0.041263 | 3 | 99.04% | 3 |
| 0.00075 | 99.14% | 0.033286 | 3 | 99.14% | 3 |
| 0.001 | 98.76% | 0.046218 | 2 | 99.00% | 3 |
| 0.00125 | 98.73% | 0.052139 | 3 | 98.73% | 3 |
| 0.0015 | 98.56% | 0.054145 | 2 | 98.71% | 3 |

**Finding:** Lowest selected loss: 0.00075; highest selected accuracy: 0.00075. Peak observed accuracy: 99.14% at 0.00075, epoch 3. Selection remains minimum validation loss; weights/logits for other peak epochs are not guaranteed retained. One seed and repeatedly used validation set; no Kaggle score.

**Run IDs:** 0.0005: `20261030_143705_1077c38e`; 0.00075: `20261030_143937_9823b3b6`; 0.001: `20261030_143050_495b4a16` (reused); 0.00125: `20261030_144207_b32afc08`; 0.0015: `20261030_144437_09ca63e5`.

**Artifacts / next:** `data/experiments/bespoke_lr_sweep_20261030_143704_01fa13bf/` has the four new runs. `five_rate_comparison/` contains all five summaries/curves and reuse/preservation provenance; the original center group is unchanged. Weights and validation outputs remain paired under `data/models/` and `data/validation_outputs/`. Notebook 05 is prepared, unexecuted: tanh first, then this ReLU comparison; matching results are reused by default, with a fresh-run option. Notebook 03/04 and prior artifacts unchanged; the authorized notebook 05 edit is separately hashed. Review before further runs.

[Fresh ten-epoch follow-up](bespoke-augmented-relu-ten-epochs.md).
