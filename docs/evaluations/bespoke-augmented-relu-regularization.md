# Augmented conventional ReLU — initial regularization trials

**Status / date:** Three fresh serial background runs complete, October 30, 2026. Notebook 05 sections 1–3 preserved, sections 4–6 prepared; notebook not executed.

**Controls / treatments:** Same ReLU `bespoke28`, LR 0.00075, eleven fixed training views, upright validation, seed 42/split/batch 64, ten epochs and 5,775/57,750-step warm-up. Initial untuned L2 λ=1e-5; decoupled WD=1e-3, reused unchanged in the combined arm. L2 objective is mean training CE + `(λ/2) × sum(weight²)`; Adam for L2-only. WD-only and combined use AdamW, with independent explicit L2 in the combined arm. Both treatments include named weight tensors and exclude biases; legacy coupled Adam weight_decay remains zero. No coefficient sweep.

| Setup | Selected accuracy | Selected CE | Epoch | Peak accuracy | Peak epoch(s) | Peak CE | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10-epoch zero-regularization baseline | 98.99% | 0.039912 | 3 | 99.04% | 4,10 | 0.045115,0.043935 | 497.3 |
| 3-epoch saved winner (different warmup) | 99.14% | 0.033286 | 3 | 99.14% | 3 | 0.033286 | 149.8 |
| L2 only | 98.98% | 0.036779 | 5 | 99.14% | 9 | 0.058072 | 526.5 |
| WD only | 99.04% | 0.043292 | 3 | 99.08% | 4,9 | 0.046667,0.069954 | 498.0 |
| L2 + WD | 99.04% | 0.037559 | 3 | 99.13% | 10 | 0.051299 | 526.4 |

**Finding:** None improved on the saved three-epoch winner. Against the ten-epoch baseline, L2-only and combined reduced selected loss; WD-only and combined gained four correct validation images. L2-only selected one fewer correct image. These initial choices show mixed metric changes, not a tuned optimum.

**Interpretation:** Selection remains minimum unpenalized validation CE. Peak-epoch weights are not guaranteed retained. Training history distinguishes data CE (`train_loss`), L2 penalty, and total penalized objective; AdamW decay is not added to that objective. Compare primarily against the ten-epoch zero-regularization baseline; the 99.14% three-epoch winner has a different warm-up. Initial strengths, one seed, repeatedly used validation set; no Kaggle score.

**Artifacts:**

- L2 only: `bespoke_lr_sweep_20261030_155745_89b6b79f`, run `20261030_155746_1c51dea6`.
- WD only: `bespoke_lr_sweep_20261030_160634_94b77997`, run `20261030_160635_069dfacc`.
- L2 + WD: `bespoke_lr_sweep_20261030_161454_abcc19e7`, run `20261030_161456_8bcf0f4e`.

Paired weights/logits remain under `data/models/` and `data/validation_outputs/`; each group holds history/config/source snapshots and a verification receipt. The combined group's `regularization_comparison/` contains all five comparison rows, full new histories and preservation hashes. Initialization, data order and validation matched the ten-epoch baseline; prior artifacts/notebooks were unchanged. Review before further experiments.
