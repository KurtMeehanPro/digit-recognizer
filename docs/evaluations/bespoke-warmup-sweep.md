# Bespoke CNN — three-epoch warm-up sweep

**Status:** All 8 runs completed. A final notebook-byte check failed afterward; status was recovered from validated artifacts, with the original error retained.
**Date:** October 30, 2026 (America/New_York run timestamps).

**Change / controls:** Unchanged 224-upsample model; seed 42, stratified 80/20 split, batch 64, Adam, zero weight decay, MPS. ReLU then tanh ran serially at targets `1e-6 → 1e-5 → 1e-4 → 1e-3`, three epochs each. Linear warm-up reaches target at update 158 of 1,575, then holds; no decay.

| Activation | Target LR | Selected epoch | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| relu | 1e-06 | 3 | 2.303033 | 9.94% |
| tanh | 1e-06 | 3 | 2.302255 | 9.94% |
| relu | 1e-05 | 3 | 2.301181 | 11.15% |
| tanh | 1e-05 | 3 | 0.810183 | 72.81% |
| relu | 0.0001 | 3 | 2.301183 | 11.15% |
| tanh | 0.0001 | 3 | 0.110873 | 96.89% |
| relu | 0.001 | 3 | 2.301210 | 11.15% |
| tanh | 0.001 | 2 | 0.111401 | 96.86% |

**Finding:** Tanh at 1e-4 (8,139/8,400 correct) led 1e-3 by only three validation images. ReLU remained near chance. Initialization, data order, and validation fingerprints matched. One seed does not establish a general winner or warm-up benefit.

**Next:** Inspect notebook 03’s final Results cell; no retraining is needed. The protection check now allows notebook 03 autosaves/presentation edits while still protecting prior artifacts, notebook 02, and captured model/training source snapshots.

**Artifacts:** `data/experiments/bespoke_lr_sweep_20261030_031845_3f06eb0f/` contains status, logs, summary CSV, comparison plots, source snapshots, and recovery provenance. Paired files remain in `data/models/` and `data/validation_outputs/`; exact filenames/run IDs are in `summary.csv` and `status.json`.

**Provenance:** `status_before_bookkeeping_recovery.json` preserves the original failed wrapper status/error; `completion_recovery` records validation and checksums. All eight paired artifacts were verified unchanged. The earlier interrupted attempts remain saved. Original partial-run documentation is retained in Git at `2301b74:docs/evaluations/bespoke-warmup-sweep.md`.

[Architecture](assets/bespoke-224-architecture.png) · [First-run comparison](bespoke-first-run.md).
