# Bespoke CNN — three-epoch warm-up sweep

**Status:** Stopped by user; 3/8 runs completed. Owned Python PID 11474 and its
caffeinate helper were confirmed exited. No automatic restart.
**Date:** October 30, 2026 (America/New_York run timestamps).

**Change / controls:** Keep the 224-upsample model, seed 42, stratified 80/20 split,
batch 64, Adam, zero weight decay, and MPS. Run ReLU then tanh serially at targets
`1e-6 → 1e-5 → 1e-4 → 1e-3`, three epochs each. Linear warm-up covers 158 of 1,575
updates: first update uses target/158; update 158 reaches target; then hold.

| Activation / target LR | Status | Selected epoch | Validation loss | Validation accuracy |
| --- | --- | ---: | ---: | ---: |
| ReLU / 1e-6 | Complete | 3 | 2.303033 | 9.94% |
| tanh / 1e-6 | Complete | 3 | 2.302255 | 9.94% |
| ReLU / 1e-5 | Complete | 3 | 2.301181 | 11.15% |
| tanh / 1e-5 | Partial: 2/3 epochs | — | 0.9807* | 65.95%* |

\*Rounded progress-log values, not a completed run's selected result. Targets
1e-4 and 1e-3 were not run. Read exact completed values from saved histories.

**Finding:** Completed runs matched initialization, data order, and validation
fingerprints. Tanh improved at 1e-5, but this incomplete sweep cannot establish
the best learning rate or a warm-up benefit.

**Next:** User tests from notebook 03. Its selected directory can resume missing
configurations; partial runs restart with fresh IDs because optimizer state was
not saved. Completed configurations are kept. No agent training remains running.

**Run IDs:** ReLU 1e-6 `20261030_033200_d3f00f63`;
tanh 1e-6 `20261030_034120_0102dfb8`;
ReLU 1e-5 `20261030_035045_c888c9c9`;
partial tanh 1e-5 `20261030_040011_24992242`.
The earlier interrupted ReLU attempt `20261030_031846_d6571c35` was retained.

**Artifacts:** `data/experiments/bespoke_lr_sweep_20261030_031845_3f06eb0f/` holds
status, logs, summary CSV, comparison plots, source snapshots, and the authorized
presentation-change manifest. Paired weights/validation files use
`data/models/bespoke_{activation}_{run_id}_best.pt` and
`data/validation_outputs/bespoke_{activation}_{run_id}_validation.pt`.

**Provenance:** Original detailed recovery/refactor record is retained in Git at
`2301b74:docs/evaluations/bespoke-warmup-sweep.md`.
[Architecture](assets/bespoke-224-architecture.png) ·
[First-run comparison](bespoke-first-run.md).
