"""Provide digit classifiers and shared training/evaluation helpers."""
# Standard imports
from pathlib import Path
import math
import random

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from scripts.dataset.digit_dataset_augmentation import augment_dataset
from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS
from scripts.model.digit_model_definitions import BespokeCNN_upsampling, BespokeCNN, LeNet
from scripts.model.digit_model_metrics import build_epoch_metrics
from scripts.model.digit_model_training_options import (
    linear_warmup_rate, regularization_configuration, regularized_weights,
    explicit_l2_penalty,
)


def build_model(model_name="lenet", **model_options):
    """Construct a classifier from explicit saved configuration; preserve LeNet default."""
    if model_name == "lenet":
        if model_options:
            raise ValueError("LeNet does not accept model options")
        return LeNet()
    if model_name == "bespoke_upsampling":
        return BespokeCNN_upsampling(**model_options)
    if model_name == "bespoke":
        return BespokeCNN(**model_options)
    raise ValueError("model_name must be 'lenet', 'bespoke_upsampling', or 'bespoke'")


def seed_training(seed):
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_training_device():
    """Select CUDA, Apple MPS, or CPU in that order of availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data_loaders(train, validation_fraction=0.2, batch_size=64, seed=42,
                       rotation_degrees=0.0, augmentation_seed=42, rotation_angles=None):
    """Normalize and split data, optionally augment the training set, and return
    training and validation loaders.

    train is a labeled pixel DataFrame. validation_fraction sets the stratified
    holdout share; batch_size controls batching; seed controls the split and
    training shuffle. Rotation options go to augment_dataset; validation
    remains unaugmented."""

    # Convert the training data to tensors and normalize pixel values to [0, 1]
    all_images = torch.tensor(
        train[PIXEL_COLUMNS].to_numpy(), dtype=torch.float32
    ).reshape(-1, 1, 28, 28) / 255.0
    all_labels = torch.tensor(train["label"].to_numpy(), dtype=torch.long)

    X_train, X_val, y_train, y_val = train_test_split(
        all_images,
        all_labels,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train["label"].to_numpy(),
    )

    # Send the training dataset for optional rotations; use the returned dataset.
    originals = TensorDataset(X_train, y_train)
    dataset = augment_dataset(originals, augmentation_seed, rotation_angles,
                               rotation_degrees)

    # use the dataset to create the training loader; validation loader uses the unaugmented tensors.
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def build_training_optimizer(model, learning_rate, weight_decay=0.0,
                             decoupled_weight_decay=0.0):
    """Create the optimizer for training.

    Weight decay determines which optimizer is used and how it is configured.
    Use AdamW for decoupled weight decay; otherwise use Adam.
    """
    if decoupled_weight_decay:
        if weight_decay:
            raise ValueError("Legacy coupled decay and decoupled decay cannot both be enabled")
        weights = regularized_weights(model)
        identities = {id(parameter) for parameter in weights}
        others = [parameter for parameter in model.parameters() if id(parameter) not in identities]
        return torch.optim.AdamW([{"params": weights, "weight_decay": decoupled_weight_decay},
                                  {"params": others, "weight_decay": 0.0}],
                                 lr=learning_rate, weight_decay=0.0)

    # if no decoupled decay, use legacy Adam with weight_decay applied to all parameters
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_one_epoch(model, train_loader, loss_function, optimizer, device,
                    learning_rate_schedule=None, step_offset=0, l2_strength=0.0, objective_metrics=None):
    """Train on every batch and return total loss and number of correct predictions."""
    model.train()
    train_loss_sum = 0.0
    train_correct = 0

    for batch_step, (batch_images, batch_labels) in enumerate(train_loader, start=1):
        if learning_rate_schedule is not None:
            rate = learning_rate_schedule(step_offset + batch_step)
            for group in optimizer.param_groups:
                group["lr"] = rate
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        loss = loss_function(logits, batch_labels)
        if l2_strength:
            penalty = explicit_l2_penalty(model, l2_strength)
            (loss.mean() + penalty).backward()
            if objective_metrics is not None:
                objective_metrics["penalty_sum"] += penalty.detach().item() * len(batch_labels)
        else:
            loss.mean().backward()
        optimizer.step()

        train_loss_sum += loss.sum().item()
        train_correct += (logits.argmax(dim=1) == batch_labels).sum().item()

    return train_loss_sum, train_correct


def validate_one_epoch(model, val_loader, loss_function, device):
    """Return validation loss total, correct count, and batches of logits and labels."""
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    epoch_val_logits = []
    epoch_val_labels = []
    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_images)
            epoch_val_logits.append(logits.detach().cpu().clone())
            epoch_val_labels.append(batch_labels.detach().cpu().clone())
            loss = loss_function(logits, batch_labels)
            val_loss_sum += loss.sum().item()
            val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()

    return val_loss_sum, val_correct, epoch_val_logits, epoch_val_labels


def update_best_checkpoint(model, metrics, checkpoint_path, epoch_val_logits,
                           epoch_val_labels, best_val_loss, best_epoch,
                           best_val_logits, best_val_labels):
    """Save improved validation-loss weights and return the matching best-epoch data."""
    if metrics["val_loss"] < best_val_loss:
        best_val_loss = metrics["val_loss"]
        best_epoch = metrics["epoch"]
        best_val_logits = torch.cat(epoch_val_logits)
        best_val_labels = torch.cat(epoch_val_labels)
        torch.save(
            {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            checkpoint_path,
        )
    return best_val_loss, best_epoch, best_val_logits, best_val_labels


def train_model(model, train_loader, val_loader, loss_function,
                epochs, checkpoint_path, learning_rate=0.001,
                warmup_fraction=0.0, weight_decay=0.0,
                l2_strength=0.0, decoupled_weight_decay=0.0):
    """Train a classifier, restore its best validation-loss weights, and return metrics.

    The loss function must return a tensor containing the individual loss values,
    one for each image in the batch.
    Labels must be integer class indices.
    Also return the best epoch's validation logits and matching labels on CPU,
    in validation-loader order. Only the current and best epochs are retained.
    Optional linear warmup covers ceil(warmup_fraction * total optimizer steps),
    followed by a constant rate. Defaults preserve the original training path.
    Explicit L2 adds lambda/2 * sum(weight**2) to mean training CE only.
    AdamW decay is separate; both exclude biases. train_loss/val_loss remain CE;
    train_objective includes L2, but decoupled decay is not a loss penalty.
    """
    regularization_configuration(l2_strength, decoupled_weight_decay)
    if l2_strength and weight_decay:
        raise ValueError("Explicit L2 cannot be combined with legacy coupled decay")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if not math.isfinite(warmup_fraction) or not 0 <= warmup_fraction <= 1:
        raise ValueError("warmup_fraction must be between zero and one")

    linear_warmup_rate(learning_rate, 0, 0)
    total_steps = epochs * len(train_loader)
    if total_steps == 0:
        raise ValueError("Training loader must contain at least one batch")
    warmup_steps = math.ceil(warmup_fraction * total_steps)
    schedule = (lambda step: linear_warmup_rate(learning_rate, step, warmup_steps)) if warmup_steps else None

    device = get_training_device()
    print(f"Training device: {device}", flush=True)
    model = model.to(device)

    optimizer = build_training_optimizer(model, learning_rate, weight_decay, decoupled_weight_decay)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_val_logits = None
    best_val_labels = None

    # below we will assume each loader processes every image in its dataset once per epoch.
    train_count = len(train_loader.dataset)
    val_count = len(val_loader.dataset)

    # Training loop for each epoch
    for epoch in range(1, epochs + 1):
        objective_metrics = {"penalty_sum": 0.0}
        train_loss_sum, train_correct = train_one_epoch(
            model, train_loader, loss_function, optimizer, device,
            learning_rate_schedule=schedule, step_offset=(epoch - 1) * len(train_loader),
            l2_strength=l2_strength, objective_metrics=objective_metrics,
        )
        val_loss_sum, val_correct, epoch_val_logits, epoch_val_labels = validate_one_epoch(
            model, val_loader, loss_function, device,
        )

        metrics = build_epoch_metrics(
            epoch, train_loss_sum, val_loss_sum, train_correct,
            val_correct, train_count, val_count,
        )

        if l2_strength or decoupled_weight_decay:
            metrics["train_l2_penalty"] = objective_metrics["penalty_sum"] / train_count
            metrics["train_objective"] = metrics["train_loss"] + metrics["train_l2_penalty"]

        if warmup_steps:
            metrics["learning_rate_first"] = schedule((epoch - 1) * len(train_loader) + 1)
            metrics["learning_rate_last"] = schedule(epoch * len(train_loader))
            metrics["optimizer_steps_completed"] = epoch * len(train_loader)
        history.append(metrics)

        # Save the best validation-loss weights and return the matching best-epoch data.
        best_val_loss, best_epoch, best_val_logits, best_val_labels = update_best_checkpoint(
            model, metrics, checkpoint_path, epoch_val_logits, epoch_val_labels,
            best_val_loss, best_epoch, best_val_logits, best_val_labels,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {metrics['train_loss']:.4f}, accuracy {metrics['train_accuracy']:.2%} | "
            f"val loss {metrics['val_loss']:.4f}, accuracy {metrics['val_accuracy']:.2%}",
            flush=True,
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Restored epoch {best_epoch}: validation loss {best_val_loss:.4f}")
    print(f"Best weights saved to: {checkpoint_path}")
    return model, history, best_epoch, best_val_logits, best_val_labels


def plot_training_history(history_df, best_epoch, chance_reference=None):
    """Return a figure comparing training and validation loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for column, label in [("train_loss", "Training"), ("val_loss", "Validation")]:
        axes[0].plot(history_df["epoch"], history_df[column], marker="o", label=label)
    for column, label in [("train_accuracy", "Training"), ("val_accuracy", "Validation")]:
        axes[1].plot(history_df["epoch"], history_df[column], marker="o", label=label)
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    if chance_reference is not None:
        axes[1].axhline(chance_reference, color="crimson", linestyle="--", linewidth=2.2,
                        label=f"Chance reference ({chance_reference:.0%})", zorder=5)
        axes[1].annotate(f"{chance_reference:.0%} chance", xy=(1, chance_reference),
                         xytext=(1.2, chance_reference + 0.08), color="crimson",
                         arrowprops={"arrowstyle": "->", "color": "crimson"},
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    for ax in axes:
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Best validation loss")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def inspect_last_validation_prediction(best_val_logits, best_val_labels):
    """Return the last validation image's label, prediction, and ten output scores."""
    import pandas as pd

    logits = best_val_logits[-1].detach().cpu()
    actual_digit = best_val_labels[-1].item()
    predicted_digit = logits.argmax().item()
    scores = pd.DataFrame({
        "Digit": range(len(logits)),
        "Logit": logits.tolist(),
        "Probability": logits.softmax(dim=0).tolist(),
    }).set_index("Digit")
    return actual_digit, predicted_digit, scores
