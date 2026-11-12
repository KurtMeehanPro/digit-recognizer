"""Validate requirements specific to the Digit Recognizer dataset."""

import numpy as np

from scripts.dataset.digit_dataset_config import EXPECTED_COLUMNS, PIXEL_COLUMNS


# These validation checks enforce requirements specific to the Digit Recognizer dataset.

def validate_dataset_columns(datasets) -> None:
    """Check column names and order only, without inspecting cell values or row counts.

    Train must start with a column named exactly 'label', followed by pixel0–pixel783.
    Test must contain only those pixel columns.
    Sample submission is not checked.
    """
    # Check train starts with 'label', followed by every pixel column in order.
    if list(datasets["train"].columns) != EXPECTED_COLUMNS["train"]:
        raise ValueError("Unexpected train columns: expected label followed by pixel0–pixel783.")

    # Check test contains only pixel0 through pixel783 in order.
    if list(datasets["test"].columns) != EXPECTED_COLUMNS["test"]:
        raise ValueError("Unexpected test columns: expected pixel0–pixel783 in order.")


def validate_image_values(datasets) -> None:
    """Check train/test pixels are nonmissing, finite integers from 0 through 255."""
    for name in ("train", "test"):
        pixels = datasets[name][PIXEL_COLUMNS].to_numpy()

        # Reject nonnumeric columns before applying NumPy arithmetic.
        if pixels.dtype.kind not in "iuf":
            raise ValueError(f"Unexpected {name} pixels: values must be numeric.")

        # Missing values and infinities are not valid pixel intensities.
        if not np.isfinite(pixels).all():
            raise ValueError(f"Unexpected {name} pixels: values must be finite (no missing values or infinity).")

        # Pixel intensities include both boundaries, 0 and 255.
        if ((pixels < 0) | (pixels > 255)).any():
            raise ValueError(f"Unexpected {name} pixels: values must be within 0–255 inclusive.")

        # Whole-valued floats are allowed, but fractional intensities are not.
        if (pixels != np.floor(pixels)).any():
            raise ValueError(f"Unexpected {name} pixels: values must be whole numbers.")


def validate_labels(datasets) -> None:
    """Check train labels are finite whole numbers from 0 through 9, allowing floats."""
    labels = datasets["train"]["label"].to_numpy()

    # Reject nonnumeric labels before applying NumPy arithmetic.
    if labels.dtype.kind not in "iuf":
        raise ValueError("Unexpected train labels: values must be numeric.")

    # Missing labels and infinities cannot identify a digit.
    if not np.isfinite(labels).all():
        raise ValueError("Unexpected train labels: values must be finite (no missing values or infinity).")

    # Digit labels include both boundaries, 0 and 9.
    if ((labels < 0) | (labels > 9)).any():
        raise ValueError("Unexpected train labels: values must be within 0–9 inclusive.")

    # Whole-valued floats are allowed, but fractional labels are not.
    if (labels != np.floor(labels)).any():
        raise ValueError("Unexpected train labels: values must be whole numbers.")
