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

TBD

## Documentation

- **Project README** — The project's top-level README, the document you're reading right now.

## Layout

- `data/` is reserved for competition data and generated artifacts; the entire directory is excluded from Git.
- `docs/` — See the [Documentation](#documentation) section above.
- `notebooks/` is reserved for numbered exploration and modeling notebooks.
- `scripts/` is reserved for reusable project code.
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

TBD
