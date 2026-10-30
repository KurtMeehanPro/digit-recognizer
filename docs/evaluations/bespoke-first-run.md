# Bespoke CNN: first completed activation comparison

This record preserves the first completed bespoke experiment: one ReLU run and
one tanh run, each for ten epochs. It was written after both runs completed and
before any next experiment. No training was run to prepare this document.

## Evidence and saved revision

The source of measured values is the existing local validation-output artifacts:
their full histories, configuration, environment, best-epoch logits, labels, and
paired checkpoints. Notebook 03 contains captured outputs of these same runs.
Both artifact label tensors match in order; each contains 8,400 labels and
8,400 x 10 best-epoch logits. Checkpoints each contain sixteen convolution weight
tensors and a classifier weight tensor of shape (10, 256).

- [Captured notebook](../../notebooks/03_bespoke_cnn.ipynb)
- [Model and training helpers](../../scripts/model/digit_model_helpers.py)
- [Evaluation helpers](../../scripts/model/digit_model_metrics.py)
- Saved experiment commit: `d0556cefdf0d8c187f88d4ed8afe4f3f15644d5b`.
- Main merge preserving it: `c10d42ce2ac5bd0d052e203b596df75f60dd63ee`.

Those commits capture the completed outputs and subsequent display improvements;
they are not separate training runs or a source snapshot made at training start.
Run IDs were generated using America/New_York local time on October 29, 2026
(the corresponding UTC dates are October 30). Checkpoint state dictionaries do
not embed a complete source-code snapshot. Architecture attribution below is
supported by the saved notebook model representation, configuration, checkpoint
tensor shapes, and checked current source.

## Architecture that actually ran

![Implemented first-run bespoke CNN architecture](assets/bespoke-224-architecture.png)

The locally rendered Pillow card shows the implemented architecture, not an unrun smaller-input proposal.

Normalized input is N x 1 x 28 x 28, with pixel values divided by 255. The model
resizes directly to 224 x 224 using bilinear interpolation with
`align_corners=False`. The three conceptual doublings (28 -> 56 -> 112 -> 224)
are implemented as one resize, not three interpolation operations.

Every stage contains two convolutions with the listed kernel/channel count,
stride 1, SAME zero padding, and the selected activation after each convolution.
Every stage then uses max pooling, kernel 2, stride 2, `ceil_mode=True`.
Shapes omit the batch dimension and use channels x height x width.

| Stage | Kernel for both convolutions | Both convolution outputs | After max pooling |
| --- | --- | --- | --- |
| 1 | 10 x 10 | 2 x 224 x 224 | 2 x 112 x 112 |
| 2 | 9 x 9 | 4 x 112 x 112 | 4 x 56 x 56 |
| 3 | 8 x 8 | 8 x 56 x 56 | 8 x 28 x 28 |
| 4 | 7 x 7 | 16 x 28 x 28 | 16 x 14 x 14 |
| 5 | 6 x 6 | 32 x 14 x 14 | 32 x 7 x 7 |
| 6 | 5 x 5 | 64 x 7 x 7 | 64 x 4 x 4 |
| 7 | 4 x 4 | 128 x 4 x 4 | 128 x 2 x 2 |
| 8 | 3 x 3 | 256 x 2 x 2 | 256 x 1 x 1 |

Flatten the final 256 values, then `Linear(256, 10)` produces raw logits. There
is no model softmax, global average pooling, batch normalization, dropout, or
residual connection. Even kernels require asymmetric SAME padding; PyTorch
reported that a padded copy may be created. Late large kernels use substantial
zero padding.

The rationale was to provide more spatial room for learned processing along a
longer descent while channels grow, potentially helping unusual handwriting.
Interpolation adds no new image information. This experiment does not prove that
upsampling helps, nor does it compare upsampling methods or input resolutions.

## Matched experiment controls

| Control | Both variants |
| --- | --- |
| Activation | ReLU versus tanh; all other configured architecture settings held fixed |
| Seed | 42 for Python, NumPy, PyTorch, and the training-loader generator |
| Split | Stratified 80/20: 33,600 training and 8,400 validation examples |
| Validation order | Fixed, no shuffle |
| Batch size | 64: 525 training batches and 132 validation batches per epoch |
| Optimizer | Adam, learning rate 0.001; no warmup or learning-rate scheduler |
| Budget | 10 epochs per activation, ReLU first, then tanh |
| Loss | Cross-entropy, reduction="none"; mean for backpropagation, summed losses / examples for reporting |
| Initialization | Same PyTorch default initialization policy, reseeded before constructing each model |
| Selection | Restore checkpoint with the lowest validation loss |
| Device | Saved environment: mps:0; notebook prints mps |
| Versions | Python 3.14.8; torch 2.14.0; NumPy 2.5.3; pandas 3.0.5; scikit-learn 1.9.1 |

The notebook rebuilds seeded loaders and reseeds model construction for each
activation. Identical initialization and batch order were checked synthetically
during implementation; this is one completed seed per variant, not a repeated
multi-seed study. Seeds do not guarantee equivalence across devices/versions.

## Measured results

Best checkpoints are selected by validation loss, not accuracy. Values below are
read from saved artifacts; the notebook's four-decimal/two-percent display rounds
them. Correct counts are calculated from saved best-epoch logits and labels.

| Variant / checkpoint | Epoch | Training loss | Training accuracy | Validation loss | Validation accuracy | Correct validation images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relu — selected | 9 | 2.301349815187 | 11.151786% | 2.3011729499271936 | 11.154762% | 937 |
| relu — final | 10 | 2.3013351926349457 | 11.151786% | 2.3011772160303026 | 11.154762% | not retained separately |
| tanh — selected | 10 | 0.05953778586261684 | 98.258929% | 0.0628951889631294 | 98.226190% | 8251 |

ReLU's selected epoch 9 predicts **digit 1 for all 8,400 validation images**;
937 are actually ones. This is established from the saved predictions, not
inferred solely from the near-chance loss. It has no correct validation examples
of the other digits. The cause is not established: no gradient-magnitude,
activation-distribution, or initialization-sensitivity study was performed.

Tanh's best and final checkpoint is epoch 10: **8,251 / 8,400 correct (98.23%)**.
Its validation loss fluctuated during training; the final selected loss is
0.0628951889631294. Of the actual digit threes, ReLU misclassified all 870 and
tanh misclassified nine. Displaying fewer examples changes neither count nor
metrics.

For context, the documented LeNet baseline reached **98.24% validation accuracy**
on the same split. The bespoke tanh result is close, slightly lower at displayed
precision; it does not demonstrate an improvement. Neither bespoke variant has a
Kaggle test score. This single comparison does not establish a general activation
ranking or the benefit of upsampling.

### Complete epoch histories

These are copied from the artifacts, not reconstructed from rounded notebook
printouts. Accuracy is a fraction in these tables.

#### relu

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.3019675840650287 | 2.3016844849359424 | 0.11080357142857143 | 0.11154761904761905 |
| 2 | 2.3015875720977785 | 2.3013951537722632 | 0.11151785714285714 | 0.11154761904761905 |
| 3 | 2.3014765362512497 | 2.3012052422478084 | 0.11098214285714286 | 0.11154761904761905 |
| 4 | 2.3014759912944975 | 2.301181845892043 | 0.11151785714285714 | 0.11154761904761905 |
| 5 | 2.3013328783852716 | 2.3013944594065348 | 0.11151785714285714 | 0.11154761904761905 |
| 6 | 2.301421764010475 | 2.301187379019601 | 0.11151785714285714 | 0.11154761904761905 |
| 7 | 2.301388393583752 | 2.3011879076276505 | 0.11151785714285714 | 0.11154761904761905 |
| 8 | 2.301338085447039 | 2.3011944970630465 | 0.11151785714285714 | 0.11154761904761905 |
| 9 | 2.301349815187 | 2.3011729499271936 | 0.11151785714285714 | 0.11154761904761905 |
| 10 | 2.3013351926349457 | 2.3011772160303026 | 0.11151785714285714 | 0.11154761904761905 |

#### tanh

| Epoch | Train loss | Validation loss | Train accuracy | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6719942153422606 | 0.18221369397072565 | 0.768154761904762 | 0.9483333333333334 |
| 2 | 0.1481765230549943 | 0.1146249605502401 | 0.9568154761904762 | 0.9678571428571429 |
| 3 | 0.10363582560703868 | 0.07049356550687835 | 0.9704464285714286 | 0.98 |
| 4 | 0.08653108562475868 | 0.11180608808284714 | 0.9747916666666666 | 0.9678571428571429 |
| 5 | 0.08381070747971535 | 0.07583351768553256 | 0.9755357142857143 | 0.9763095238095238 |
| 6 | 0.10742001162664522 | 0.10330551326274871 | 0.9698214285714286 | 0.9717857142857143 |
| 7 | 0.07955476826056838 | 0.06868040509521961 | 0.9772619047619048 | 0.98 |
| 8 | 0.06988676020831225 | 0.0923422196436496 | 0.9786309523809524 | 0.9735714285714285 |
| 9 | 0.06296055465759266 | 0.07393710506104288 | 0.9813392857142857 | 0.9786904761904762 |
| 10 | 0.05953778586261684 | 0.0628951889631294 | 0.9825892857142857 | 0.9822619047619048 |

## Preserved local artifacts and fingerprints

Paths are relative to the existing repository root. These files remain in the
ignored data directory. They were read, not moved, overwritten, or deleted.
The captured notebook outputs are also retained in Git. Fingerprints below
identify the actual files independently of filenames.

### relu
- Run ID: `20261029_220327_ba95689c`.
- Weights: `data/models/bespoke_relu_20261029_220327_ba95689c_best.pt`.
- Weights SHA-256: `c4c7e6e29e744c90cdfc1e378be09f1ff05411d49591a1f112cfec4e995b87ff`.
- Validation outputs/history/config: `data/validation_outputs/bespoke_relu_20261029_220327_ba95689c_validation.pt`.
- Validation-output SHA-256: `6b3c283eaadcfc06593f58d2590475e804e1e168af7fe389b71a18b339b38624`.

### tanh
- Run ID: `20261029_223450_f9e9ac62`.
- Weights: `data/models/bespoke_tanh_20261029_223450_f9e9ac62_best.pt`.
- Weights SHA-256: `ab8628ae291f23d49f1680b3e587254c1c5ee618b56f3b5856c6e9d08c58dc3b`.
- Validation outputs/history/config: `data/validation_outputs/bespoke_tanh_20261029_223450_f9e9ac62_validation.pt`.
- Validation-output SHA-256: `6cb37f2daea6f442e08e536d9f43aa0bc17f577eeb14b95d74701964f415621e`.

Both artifacts contain the same recorded fingerprints:
- Dataset fingerprint: `9a191170902781e41923084ada8d7fcd08642446b4049ccae36d95508d5196f6`.
- Validation fingerprint: `a6369e3c7f4d69d3dc9d32cb0b21e187357cc125887bcb9aa2fd118d4bcc7713`.

The dataset fingerprint hashes the NumPy array bytes of label plus pixel columns
in row order. The validation fingerprint hashes normalized float32 image tensor
bytes followed by int64 label tensor bytes. These are pipeline fingerprints, not
checksums of the original CSV files.

## What changed after the run, and what did not

The current model still matches the eight-stage 224-resize architecture recorded
above. Verified subsequent edits affect presentation or results access:

- Accuracy panels gained a contrasting dashed 10% chance reference; loss panels
  have no percentage reference.
- Shared axes compare ReLU and tanh; individual learning curves remain available.
- Explicit Markdown activation headings distinguish result groups.
- Actual-digit-three galleries are capped at ten displayed images per activation;
  ReLU shows ten of 870 errors, tanh shows all nine errors.
- Results-only cells reload plotting modules and use saved histories/logits. They
  do not retrain; module-qualified calls avoid stale imported function aliases.
- Documentation was corrected to describe completed rather than pending runs.

These changes did not alter learned weights, numerical histories, splits, model
architecture, optimizer settings, or selected checkpoints. Regenerated graph and
gallery outputs were checked against the real saved artifacts. The original
pre-display-refresh notebook was also backed up locally, but the canonical saved
artifacts and captured completed metrics are the durable experiment record.

The earlier ten-stage diagram with 512/1,024-channel stages and global average
pooling was a proposal superseded before this run; it was not the trained model.
A future smaller-input/no-224-resize or 28-input architecture remains a proposal:
it has **not been implemented or run**. No new architecture run is hidden behind
the display changes. Any future model change should be a separately labeled run
with new artifacts so this comparison remains intact.

Lower learning rates, warmup, and learning-rate decay are untested optimization
hypotheses. None was applied to either recorded run. No speed optimization,
parallel activation run, new seed study, or next training experiment was executed
to produce this record.
