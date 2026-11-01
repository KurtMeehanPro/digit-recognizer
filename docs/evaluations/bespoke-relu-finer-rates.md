# Bespoke CNN — ReLU finer-rate diagnostic

**Status:** Complete, ReLU only; October 30, 2026. No notebook changes or tanh runs.

**Change / controls:** Targets 2.5e-6, 6.25e-6, 7.5e-6, serial MPS, three epochs each. Original 224-upsample model; seed 42, stratified 80/20 split, batch 64, Adam with zero weight decay. Linear warm-up reaches target at update 158 of 1,575, then holds.

| Target LR | Selected epoch | Validation loss | Accuracy | Predicted classes (8,400 images) |
| --- | ---: | ---: | ---: | --- |
| 2.5e-06 | 3 | 2.302411 | 9.85% | 6: 8,400 |
| 6.25e-06 | 3 | 2.301181 | 11.15% | 1: 8,400 |
| 7.5e-06 | 3 | 2.301178 | 11.15% | 1: 8,400 |

**Finding:** Each selected checkpoint predicts one class for every validation image. These finer rates did not resolve collapse within this three-epoch procedure. The cause remains unestablished; no gradient diagnostics or extra inference were performed.

**Run IDs:** `20261030_123454_bf8ab8bc`; `20261030_124412_2ea6965b`; `20261030_125328_ff484a1b`.

**Artifacts:** `data/experiments/bespoke_lr_sweep_20261030_123453_a9f92e74/` holds status, logs, summary, plots, source snapshots, the launch record, and saved-logit class distributions. Paired weights/validation files remain in `data/models/` and `data/validation_outputs/`.

**Preservation / next:** Notebook 03 matched its before/after SHA-256; the earlier four-rate group was unchanged. Full hashes are in `probe.json`. Review before selecting another experiment; no further run was started.

**Working hypothesis / direction:**

- The two feature maps immediately after the 224 resize may form an early signal/information bottleneck that interacts differently with ReLU and tanh. Tanh learned in the tested upsampling runs while ReLU collapsed; this does not establish why, or prove that weights or information were destroyed.
- The finer three-epoch learning-rate tests did not rescue ReLU: selected checkpoints predicted only digit 6 or only digit 1 across all validation images. Rates above 0.001 remain untested for the upsampling model; needing a higher rate is an open hypothesis, not an established explanation.
- Park notebook 03 for now and focus on notebook 04's conventional architecture. Its initial background results are [recorded separately](bespoke-conventional-lr-sweep.md); no bottleneck-isolation test has been performed.
