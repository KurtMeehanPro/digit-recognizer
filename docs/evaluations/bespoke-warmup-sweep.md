# Three-epoch bespoke learning-rate sweep: partial run record

The agent-owned background sweep was **stopped at the user's request** so the user
could test from notebook 03. No additional training was launched afterward. Three
of the eight planned runs completed. The partial tanh run is not a completed
three-epoch result, and this record does not rank unrun learning rates.

## Fixed design and controls

The existing eight-stage 224-upsample architecture is unchanged. See the
[architecture card](assets/bespoke-224-architecture.png) and the
[first-run record](bespoke-first-run.md). Each stage has two stride-one SAME
convolutions and ceil-mode max pooling, ending at 1 x 1 x 256, then linear ten
logits. Both ReLU and tanh are configured serially, with target rates
`1e-6, 1e-5, 1e-4, 1e-3` in low-to-high order.

Each run uses seed 42, stratified 80/20 validation (33,600 / 8,400), batch size 64,
Adam with explicit weight_decay=0, and three epochs on MPS. There are 1,575
optimizer updates. Linear warm-up covers ceil(10% x 1,575) = 158 updates: update 1
uses target/158, update 158 reaches target, and all later updates hold target.
There is no subsequent learning-rate decay. Matched initial-weight, first-epoch
order, and validation fingerprints were verified across all three completed runs.
The first-epoch order probe does not consume the training loader's generator.

## Completed results

Best checkpoints are selected by lowest validation loss. Values are read from
saved metadata/logits, not estimated from plots.

| Activation | Target LR | Best epoch | Validation loss | Validation accuracy | Correct / 8,400 | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relu | 1e-06 | 3 | 2.303032866887 | 9.940476% | 835 | 559.6 |
| tanh | 1e-06 | 3 | 2.302255438850 | 9.940476% | 835 | 565.0 |
| relu | 1e-05 | 3 | 2.301181160609 | 11.154762% | 937 | 565.3 |

The interrupted tanh `1e-5` run reached epoch 2 with printed validation loss
**0.9807** and accuracy **65.95%**. Those are rounded progress-log values, not a
completed run's final selected artifact. Both activations at `1e-4` and `1e-3`
were not run in this sweep. These partial results do not establish which target
rate is best. Earlier ten-epoch baseline results remain a separate experiment.

## Saved artifacts and recovery

Experiment directory, relative to the repository:
`data/experiments/bespoke_lr_sweep_20261030_031845_3f06eb0f/`

It contains `status.json`, cumulative `sweep.log`, detached `launcher.log`,
`summary.csv`, the actual comparison PNGs, source snapshots, and an integrity
manifest. Model weights and validation-output files use the existing paired
run-ID convention in ignored `data/models/` and `data/validation_outputs/`.

Completed run IDs:

- `20261030_033200_d3f00f63` — relu, target 1e-06.
  Weights: `data/models/bespoke_relu_20261030_033200_d3f00f63_best.pt`.
  Validation/history/config: `data/validation_outputs/bespoke_relu_20261030_033200_d3f00f63_validation.pt`.

- `20261030_034120_0102dfb8` — tanh, target 1e-06.
  Weights: `data/models/bespoke_tanh_20261030_034120_0102dfb8_best.pt`.
  Validation/history/config: `data/validation_outputs/bespoke_tanh_20261030_034120_0102dfb8_validation.pt`.

- `20261030_035045_c888c9c9` — relu, target 1e-05.
  Weights: `data/models/bespoke_relu_20261030_035045_c888c9c9_best.pt`.
  Validation/history/config: `data/validation_outputs/bespoke_relu_20261030_035045_c888c9c9_validation.pt`.

Partial run IDs are retained rather than overwritten:

- `20261030_031846_d6571c35`: original ReLU `1e-6` attempt, tool-session process
  ended after two reported epochs during a transport disconnect. Its absence was
  confirmed before restarting under a fresh ID.
- `20261030_040011_24992242`: tanh `1e-5`, stopped after two reported epochs by the
  user's explicit request. Its partial checkpoint/log remain local.

No optimizer/RNG checkpoint was saved for exact mid-run continuation. Resuming
this directory keeps completed configurations and restarts incomplete ones from
matched initialization using fresh IDs. Changing controls while resuming is
rejected. Choose a new directory for a deliberately different experiment.

The owned Python PID 11474 was verified before SIGINT and confirmed exited, along
with its caffeinate helper. Notebook kernels were not stopped. The status records
`stopped_by_user` and `no_automatic_restart`.

## Refactor and notebook handover

- [Model definitions](../../scripts/model/digit_model_definitions.py) now contain
  LeNet and BespokeCNN. Existing helper imports remain compatible. The exact class
  syntax trees and rationale docstrings were preserved; saved LeNet/ReLU/tanh
  checkpoints produced identical CPU outputs before and after extraction.
- [Training helpers](../../scripts/model/digit_model_helpers.py) retain training,
  including optional warm-up. [Sweep utility](../../scripts/utilities/bespoke_lr_sweep.py)
  retains the existing run artifacts and prevents overlapping managed sweeps.
- [Notebook orchestration/reporting helpers](../../scripts/model/digit_experiment_helpers.py)
  keep notebook 03 concise. Its training call explicitly passes epochs, target
  rates, warmup_fraction, and weight_decay. Its one final Results call reads saved
  outputs; historical ten-epoch reports are labeled separately.
- Notebook 02 and its saved outputs remain byte-for-byte unchanged. Notebook 03's
  prior progress, numerical results, and seven historical figures were preserved
  and relocated; the architecture card is a separate display cell. The full
  pre-reorganization notebook is backed up with this local experiment.

The running serial process had already imported its model/training functions; it
never reloaded them or spawned per-run workers. Its original source snapshots and
hashes therefore describe what executed despite the on-disk behavior-preserving
extraction. An explicit manifest records the authorized notebook reorganization.
The sweep was later stopped by the user before its end-of-sweep byte-hash check;
no claim is made that the changed presentation bytes are identical.

The Pillow architecture card was rendered locally following the approved layout
after the supported Library download failed. Every displayed convolution/pooling
shape was checked against a real CPU forward pass. This is a display change, not
a new architecture or an additional experiment. No new dependencies, concurrency,
benchmark, smaller-input model, remote push, or publication was performed.
