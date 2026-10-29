# Digit Recognizer

## Overview

An image-classification project for the
[Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/overview),
using MNIST data to identify handwritten digits from 0 through 9.

The project will explore the provided labeled training data, develop and evaluate
image-classification models using a training/validation split, and prepare
predictions for Kaggle's unlabeled test images. Submissions are evaluated using
classification accuracy: the proportion of test images classified correctly.

## Results Summary

The LeNet-style CNN achieved **98.24% validation accuracy** and **98.10% Kaggle
accuracy**. See [model evaluations](docs/evaluations/README.md) for the training
setup, per-digit metrics, and reproducibility check.

## Documentation

- **Project README** — The project's top-level README, the document you're reading right now.
- [Model implementation options](docs/model_implementation_choices.md) — Planned model approaches.
- [Model evaluations](docs/evaluations/README.md) — Completed experiments, validation results, and Kaggle results.

## Layout

- `data/` is reserved for competition data and generated artifacts; the entire directory is excluded from Git.
- `docs/` — See the [Documentation](#documentation) section above.
- `notebooks/` is reserved for numbered exploration and modeling notebooks.
- `scripts/dataset/` — Dataset configuration, loading helpers, and sanity checks.
- `scripts/model/` — Model architecture, training helpers, and evaluation metrics.
- `scripts/utilities/` — Standalone tools for prediction and comparing training runs.
- `tests/` is reserved for behavioral tests of project functionality.

## Setup

Use **Python 3.14.8** with [requirements.txt](requirements.txt).

Install the packages:

```bash
python -m pip install -r requirements.txt
```

### Kaggle Authentication

Authenticate with Kaggle before downloading competition data. Generate an API token in your [Kaggle account settings](https://www.kaggle.com/settings/api) and save the token in `~/.kaggle/access_token`, as described in the [KaggleHub authentication instructions](https://github.com/Kaggle/kagglehub#authenticate).

Accept the [competition rules](https://www.kaggle.com/competitions/digit-recognizer/rules) on Kaggle.

## Dataset sanity check

From the project root, run:

```bash
python scripts/dataset/digit_dataset_sanity.py
```

The script downloads missing competition files to `data/raw/` and loads the data.
It checks:

- Training and test column names and order.
- Pixel values are present, finite whole numbers from 0 through 255.
- Training labels are present, finite whole numbers from 0 through 9.

It prints a success message after each passing check, then the training and test
shapes and training label counts. A failed check stops the script with an error.

## Saved-model utilities

Run these from the project root with the Python environment active:

```bash
python scripts/utilities/predict_digits.py
python scripts/utilities/compare_training_runs.py
```

Prediction loads saved LeNet weights and writes a predictions CSV. Run comparison
checks the two latest paired model checkpoints and validation outputs for exact
equality. Both tools accept command-line arguments; use `--help` for options.
