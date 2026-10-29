# Model evaluations

This page records completed model evaluations for the Digit Recognizer project.
The LeNet-style CNN is our baseline; future models can be compared using the same
validation split. Planned approaches are listed in the
[model implementation options](../model_implementation_choices.md).

## Results summary

### Validation results

| Model | Best epoch | Validation loss | Validation accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| LeNet-style CNN | 8 | 0.0609 | 98.24% | 0.9822 |

The checkpoint was selected by lowest validation loss, rather than by highest
validation accuracy.

### Kaggle test result

**Accuracy: 98.10%** (Kaggle score: **0.98100**).

The first successful submission appeared at **rank 688** on the rolling
[Digit Recognizer leaderboard](https://www.kaggle.com/competitions/digit-recognizer/leaderboard)
on October 29, 2026. Rank changes as other participants submit.
The leaderboard stated that scoring used all 28,000 test images.

The model was trained locally using
[notebook 2](../../notebooks/02_lenet_style_cnn.ipynb). The
[prediction utility](../../scripts/utilities/predict_digits.py) loaded its saved
weights and generated the CSV uploaded to Kaggle. A Kaggle-hosted submission
notebook was not used for this result.

Kaggle does not provide the test labels needed to calculate our own per-digit
metrics. The remaining analysis uses the held-out validation set.

## Per-digit validation results

Count refers to actual examples of the digit. Incorrectly predicted is the number
of those examples assigned another digit. Precision, recall, F1, and accuracy use
a one-versus-rest interpretation for each digit. Accuracy includes true negatives;
recall is the fraction of actual examples of that digit classified correctly.

| Digit | Count | Correctly predicted | Incorrectly predicted | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 827 | 820 | 7 | 98.20% | 99.15% | 98.68% | 99.74% |
| 1 | 937 | 932 | 5 | 98.31% | 99.47% | 98.89% | 99.75% |
| 2 | 835 | 815 | 20 | 98.91% | 97.60% | 98.25% | 99.65% |
| 3 | 870 | 843 | 27 | 97.91% | 96.90% | 97.40% | 99.46% |
| 4 | 814 | 808 | 6 | 98.54% | 99.26% | 98.90% | 99.79% |
| 5 | 759 | 737 | 22 | 97.88% | 97.10% | 97.49% | 99.55% |
| 6 | 827 | 821 | 6 | 98.68% | 99.27% | 98.98% | 99.80% |
| 7 | 880 | 874 | 6 | 97.54% | 99.32% | 98.42% | 99.67% |
| 8 | 813 | 791 | 22 | 98.26% | 97.29% | 97.78% | 99.57% |
| 9 | 838 | 811 | 27 | 98.18% | 96.78% | 97.48% | 99.50% |

## Findings

- Digit 3 had the lowest F1, at 97.40%, with 27 of 870 actual threes misclassified.
- Digit 9 had the lowest recall, at 96.78%, with 27 of 838 actual nines misclassified.
- Digit 6 had the highest F1, at 98.98%.
- Training loss continued falling after epoch 8, but neither epoch 9 nor epoch 10
  improved validation loss. We restored epoch 8 rather than the final weights.
- The validation accuracy of 98.24% and Kaggle accuracy of 98.10% were close in this
  experiment. Further model comparisons should use validation results for selection.
- The misclassified-three gallery includes ambiguous handwriting and digits that
  visually resemble other classes. Visual guesses about why the model erred are
  hypotheses, not explanations established by the model's predictions alone.

## Evaluation approach

- Split the 42,000 labeled images into 33,600 training and 8,400 validation images.
- Use a stratified 80/20 split with seed 42 and keep validation order fixed.
- Scale pixel intensities from 0–255 to 0–1; reshape each image to one 28 × 28 channel.
- Train the LeNet-style CNN from random initialization, using tanh and average pooling.
- Use Adam with learning rate 0.001, batch size 64, and ten epochs on Apple MPS.
- Request individual cross-entropy losses; use their mean for backpropagation and
  their sum divided by the dataset size for epoch loss reporting.
- Restore the checkpoint with the lowest validation loss.
- Keep the best epoch's validation logits and labels aligned for later inspection.

The model has 61,706 learned values in ten weight and bias tensors. Training uses
only the training subset; validation does not update model parameters. The test
images are used for submission predictions.

## Reproducibility check

We compared two separately saved training runs using
[compare_training_runs.py](../../scripts/utilities/compare_training_runs.py):

- All model weight and bias tensors matched exactly.
- All 84,000 validation logits (8,400 images × ten scores) matched exactly.
- Validation labels matched in order.
- Maximum absolute logit difference was zero; no predicted digits changed.
- Both runs selected epoch 8.

This establishes exact agreement for the compared runs. A fixed seed alone does
not guarantee identical results across every future run, hardware configuration,
or library version.

## Detailed outputs and saved artifacts

[Notebook 2](../../notebooks/02_lenet_style_cnn.ipynb) contains learning curves, a
confusion matrix, the per-digit table sorted by F1, a gallery of misclassified
threes, and one validation image's logits and probabilities.

Local artifacts are excluded from Git:

- `data/models/`: best model weights for each run.
- `data/validation_outputs/`: logits, labels, best epoch, seed, validation fraction,
  run identifier, and matching checkpoint path.
- `data/predictions/`: the predictions CSV generated by the standalone utility.

Model and validation-output filenames share a run identifier so the two files can
be paired. PyTorch files in these folders have different contents: model parameters
in one, validation results and run information in the other.

## Further notes

These results use one validation split. We have not yet evaluated the bespoke CNN
or other planned architectures. Comparing them on this same split will give us a
baseline for assessing changes, though repeated tuning can also overfit validation.
