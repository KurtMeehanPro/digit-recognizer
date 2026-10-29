"""Provide the LeNet-style digit classifier."""
# Standard imports
from pathlib import Path
import random

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS


class LeNet(nn.Module):
    """Classify 28 × 28 grayscale images using a LeNet-style CNN."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, images):
        return self.network(images)


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


def build_data_loaders(train, validation_fraction=0.2, batch_size=64, seed=42):
    """Normalize pixels and return stratified training and validation loaders."""

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
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),  # almost like zip, TensorDataset pairs images and labels
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


def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    """Train on every batch and return total loss and number of correct predictions."""
    model.train()
    train_loss_sum = 0.0
    train_correct = 0

    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images)
        loss = loss_function(logits, batch_labels)
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


def build_epoch_metrics(epoch, train_loss_sum, val_loss_sum, train_correct,
                        val_correct, train_count, val_count):
    """Calculate average loss and accuracy for a completed training/validation epoch."""
    return {
        "epoch": epoch,
        "train_loss": train_loss_sum / train_count,
        "val_loss": val_loss_sum / val_count,
        "train_accuracy": train_correct / train_count,
        "val_accuracy": val_correct / val_count,
    }


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
                epochs, checkpoint_path, learning_rate=0.001):
    """Train a classifier, restore its best validation-loss weights, and return metrics.

    The loss function must return a tensor containing the individual loss values,
    one for each image in the batch.
    Labels must be integer class indices.
    Also return the best epoch's validation logits and matching labels on CPU,
    in validation-loader order. Only the current and best epochs are retained.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    device = get_training_device()
    print(f"Training device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        train_loss_sum, train_correct = train_one_epoch(
            model, train_loader, loss_function, optimizer, device,
        )
        val_loss_sum, val_correct, epoch_val_logits, epoch_val_labels = validate_one_epoch(
            model, val_loader, loss_function, device,
        )

        metrics = build_epoch_metrics(
            epoch, train_loss_sum, val_loss_sum, train_correct,
            val_correct, train_count, val_count,
        )
        history.append(metrics)
        best_val_loss, best_epoch, best_val_logits, best_val_labels = update_best_checkpoint(
            model, metrics, checkpoint_path, epoch_val_logits, epoch_val_labels,
            best_val_loss, best_epoch, best_val_logits, best_val_labels,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {metrics['train_loss']:.4f}, accuracy {metrics['train_accuracy']:.2%} | "
            f"val loss {metrics['val_loss']:.4f}, accuracy {metrics['val_accuracy']:.2%}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Restored epoch {best_epoch}: validation loss {best_val_loss:.4f}")
    print(f"Best weights saved to: {checkpoint_path}")
    return model, history, best_epoch, best_val_logits, best_val_labels


def plot_training_history(history_df, best_epoch):
    """Return a figure comparing training and validation loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for column, label in [("train_loss", "Training"), ("val_loss", "Validation")]:
        axes[0].plot(history_df["epoch"], history_df[column], marker="o", label=label)
    for column, label in [("train_accuracy", "Training"), ("val_accuracy", "Validation")]:
        axes[1].plot(history_df["epoch"], history_df[column], marker="o", label=label)
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Classification accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
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
