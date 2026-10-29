"""Evaluate digit classifiers on held-out data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch


def plot_validation_confusion_matrix(model, val_loader):
    """Return the validation confusion matrix and its figure for digits 0 through 9."""
    device = next(model.parameters()).device
    model.eval()
    actual_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            logits = model(batch_images.to(device))
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
            actual_labels.extend(batch_labels.cpu().tolist())

    matrix = confusion_matrix(actual_labels, predicted_labels, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
    )
    ax.set_title("Validation confusion matrix — best model")
    fig.tight_layout()
    return matrix, fig


def build_per_digit_metrics(matrix):
    """Summarize a confusion matrix whose rows are actual digits and columns predictions.

    Count and incorrect predictions refer to actual examples of each digit.
    Accuracy treats each digit versus all others and includes true negatives.
    Undefined precision, recall, or F1 values are reported as zero.
    """
    matrix = np.asarray(matrix)
    count = matrix.sum(axis=1)
    correct = matrix.diagonal()
    predicted_count = matrix.sum(axis=0)
    total = matrix.sum()
    false_positive = predicted_count - correct
    false_negative = count - correct
    true_negative = total - correct - false_positive - false_negative

    def divide(numerator, denominator):
        return np.divide(
            numerator, denominator,
            out=np.zeros_like(numerator, dtype=float), where=denominator != 0,
        )

    return pd.DataFrame({
        "Digit": np.arange(len(matrix)),
        "Count": count,
        "Correctly Predicted": correct,
        "Incorrectly Predicted": false_negative,
        "Precision": divide(correct, predicted_count),
        "Recall": divide(correct, count),
        "F1": divide(2 * correct, predicted_count + count),
        "Accuracy": divide(correct + true_negative, total),
    }).set_index("Digit")


def plot_misclassified_digit(model, val_loader, digit=3, columns=8):
    """Display every validation image of the given digit predicted as another digit."""
    device = next(model.parameters()).device
    model.eval()
    images = []
    predictions = []
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            predicted = model(batch_images.to(device)).argmax(dim=1).cpu()
            labels = batch_labels.cpu()
            mask = (labels == digit) & (predicted != labels)
            images.extend(batch_images.cpu()[mask])
            predictions.extend(predicted[mask].tolist())

    print(f"Actual digit {digit}: {len(images)} misclassified validation images")
    if not images:
        return None
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, prediction in zip(axes.flat, images, predictions):
        ax.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Predicted {prediction}")
    fig.suptitle(f"Misclassified validation images — actual digit {digit}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
