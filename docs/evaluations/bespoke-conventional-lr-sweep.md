# Bespoke Conventional CNN — first learning-rate test

**Status:** All 8 background runs complete; notebooks 03/04 and their saved outputs were unchanged.
**Date:** October 30, 2026.

**Change / controls:** New `bespoke28` model: 28-input, five blocks, two SAME stride-1 convolutions each, kernels 7/6/5/4/3, channels 16/32/64/128/256, ceil-mode max pooling after each, flatten 256 → linear ten logits. ReLU/tanh serial; seed 42, stratified 80/20 split, batch 64, Adam with zero weight decay, MPS, three epochs. Warm-up reaches target at update 158 of 1,575, then holds.

| Activation | Target LR | Selected epoch | Validation loss | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| relu | 1e-06 | 3 | 2.302255 | 10.36% |
| tanh | 1e-06 | 3 | 2.246662 | 21.56% |
| relu | 1e-05 | 3 | 1.535456 | 36.64% |
| tanh | 1e-05 | 3 | 0.855930 | 68.94% |
| relu | 0.0001 | 3 | 0.170796 | 94.73% |
| tanh | 0.0001 | 3 | 0.097029 | 97.26% |
| relu | 0.001 | 3 | 0.066391 | 98.10% |
| tanh | 0.001 | 3 | 0.134066 | 96.23% |

**Finding:** ReLU at 1e-3 led this test (8,240/8,400 correct); tanh favored 1e-4. ReLU learned under the conventional architecture, unlike the tested upsampling runs. Multiple architectural features changed, so this does not isolate the proposed two-feature-map bottleneck or establish general superiority. One seed, no Kaggle score.

**Time:** Each three-epoch run took 12.8–13.4 seconds. These are measured experiment timings, not a same-model speed benchmark.

**Next:** Inspect saved results without retraining by selecting this group in notebook 04’s Results cell. The notebook itself was not executed.

**Artifacts:** `data/experiments/bespoke_lr_sweep_20261030_131312_866ea4fa/` holds status, summary CSV, full run IDs/config/history references, comparison plots, source snapshots, and notebook/prior-group preservation hashes. Paired checkpoints and validation artifacts have the distinct `bespoke28_` prefix in `data/models/` and `data/validation_outputs/`. Matching initialization/order/split fingerprints and all eight artifact pairs were verified.

[Architecture card](assets/bespoke-conventional-architecture.png) · [Notebook 03 hypothesis](bespoke-relu-finer-rates.md).

**Experiment direction:**

- Notebook 03 remains the upsampling track; notebook 04 remains the conventional track. Notebook 05 now tests conventional tanh with fixed rotation augmentation and no weight decay; its [background result is recorded separately](bespoke-fixed-rotation-augmentation.md). Earlier runs remain separate.
- Notebook 05 has six experiments: tanh augmentation, the ReLU learning-rate search, the best-settings ten-epoch run, then L2-only, decoupled weight-decay-only and combined trials. The [initial regularization results](bespoke-augmented-relu-regularization.md) are recorded separately; strengths are not tuned optima.

**Future Kaggle presentation:** Show the final best architecture, its model card and a brief approach, with a GitHub “Want to see more” link for experiment details. Keep the historical experiment sequence in the repository. Presentation idea only; no Kaggle notebook created or published.
