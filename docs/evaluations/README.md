# Curated model results

The [Model implementation choices](../model_implementation_choices.md) describes the
architectures and notebook roles. These are selected findings, not a run log.

| Notebook / setup | Selected epoch | Validation loss | Validation accuracy | Kaggle accuracy |
| --- | ---: | ---: | ---: | ---: |
| [02 — LeNet-style CNN](../../notebooks/02_lenet_style_cnn.ipynb) | 8 | 0.0609 | 98.24% | 98.10% |
| [03 — Upsampling CNN, tanh](../../notebooks/03_bespoke_upsampling_cnn.ipynb) | 10 | 0.0629 | 98.23% | Not submitted |
| [04 — Conventional CNN, ReLU](../../notebooks/04_bespoke_conventional_cnn.ipynb) | 3 | 0.066391 | 98.10% | Not submitted |
| [05 — Conventional CNN — Performance](../../notebooks/05_bespoke_conventional_cnn_performance.ipynb), ReLU + fixed rotations | 3 | 0.033286 | **99.14%** | Not submitted |

The best saved conventional run used LR 0.00075, eleven fixed training views
(0°, ±5°, ±10°, ±22°, ±45°, ±90°), 10% linear warm-up and no regularization.
Its upright validation result was 8,328/8,400 correct. Longer training and the
initial L2/decoupled-weight-decay trials did not improve on that saved winner.

Checkpoints are selected by lowest unpenalized validation cross-entropy, which
can differ from the epoch with highest accuracy. Experiments use the same
seed 42 stratified 80/20 split, but architectures and training budgets differ;
augmentation expands the training budget elevenfold. These single-seed results
and repeated validation tuning do not establish a general improvement or an
unbiased final test score. Only the LeNet result above has a Kaggle score.

Notebook 05 includes captured comparisons, per-digit metrics and saved-winner
error galleries. Remaining errors mix unusual handwriting with recognizable
digits; model probabilities are not calibrated certainty.

Detailed run notes, full histories, datasets and checkpoints are local records
under ignored `data/`. Saved-output refresh cells require those local artifacts;
they are not bundled with the public repository. Public notebook outputs and
this page retain the selected findings. New results are promoted here selectively.
