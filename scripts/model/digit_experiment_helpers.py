"""Notebook orchestration and saved-output reporting for digit experiments.

These helpers keep notebook settings readable. Reporting reads saved outputs;
training starts only when run_bespoke_sweep is explicitly called.
"""
import importlib
import json
from pathlib import Path
import subprocess
import signal
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import Image, Markdown, display
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from scripts.dataset.digit_dataset_config import PROJECT_ROOT
from scripts.model import digit_model_definitions as definitions
from scripts.model import digit_model_helpers as models
from scripts.model import digit_model_metrics as metrics


def refresh_helpers():
    """Refresh definitions/training/plot modules without rerunning an experiment."""
    importlib.reload(definitions)
    importlib.reload(models)
    importlib.reload(metrics)


def show_dataset_summary():
    for label, filename in (("Training", "train.csv"), ("Test", "test.csv")):
        frame = pd.read_csv(PROJECT_ROOT / "data/raw" / filename)
        print(f"{label} shape: {frame.shape}")


def inspect_bespoke_model(activation="relu", upsample_mode="bilinear", seed=42):
    """CPU-only shape inspection, no training or dataset access."""
    models.seed_training(seed)
    model = models.build_model("bespoke", activation=activation, upsample_mode=upsample_mode)
    print(model)
    with torch.no_grad():
        print("Output shape:", tuple(model(torch.zeros(1, 1, 28, 28)).shape))


def run_bespoke_sweep(*, epochs, learning_rates, warmup_fraction, weight_decay,
                       sweep_directory=None):
    """Run serially using the existing utility and this notebook's interpreter.

    warmup_fraction is forwarded to train_model through the utility. Completed
    configurations in an existing directory are kept; incomplete ones restart
    with fresh IDs. A held managed-run lock prevents duplicate training.
    """
    from scripts.utilities.bespoke_lr_sweep import active_sweep_directories
    active = active_sweep_directories()
    if active:
        raise RuntimeError(f"A sweep is already running at {active[0]}; no duplicate training started.")
    command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts/utilities/bespoke_lr_sweep.py"),
               "--run", "--epochs", str(epochs), "--learning-rates", *map(str, learning_rates),
               "--warmup-fraction", str(warmup_fraction), "--weight-decay", str(weight_decay)]
    selected = Path(sweep_directory) if sweep_directory is not None else None
    if selected is not None:
        command.extend(["--resume-directory", str(selected)])
    with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("SWEEP_DIRECTORY "):
                    selected = Path(line.removeprefix("SWEEP_DIRECTORY ").strip())
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
        if process.wait() != 0:
            raise RuntimeError("Sweep stopped; inspect its preserved log/status before restarting.")
    return selected


def _show_baseline_runs(run_files):
    """Render the named historical baseline exactly once from aligned saved logits."""
    import hashlib
    runs = {name: torch.load(Path(path), map_location="cpu", weights_only=True)
            for name, path in run_files.items()}
    train = pd.read_csv(PROJECT_ROOT / "data/raw/train.csv")
    _, loader = models.build_data_loaders(train, validation_fraction=.2, batch_size=64, seed=42)
    images, labels = loader.dataset.tensors
    fingerprint = hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    summary = []
    for activation, run in runs.items():
        assert run["validation_sha256"] == fingerprint and torch.equal(run["labels"], labels)
        display(Markdown(f"#### Historical {activation.upper()} — {len(run['history'])} epochs"))
        history = pd.DataFrame(run["history"])
        display(history)
        figure = models.plot_training_history(history, run["best_epoch"], chance_reference=.1)
        display(figure)
        plt.close(figure)
        predictions = run["logits"].argmax(1)
        matrix = confusion_matrix(labels.numpy(), predictions.numpy(), labels=list(range(10)))
        figure, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title(f"Historical {activation.upper()} — best-checkpoint confusion matrix")
        figure.tight_layout()
        display(figure)
        plt.close(figure)
        per_digit = metrics.build_per_digit_metrics(matrix).sort_values("F1")
        display(per_digit)
        figure = metrics.plot_misclassified_digit(None, loader, digit=3, max_images=10,
                                                  validation_logits=run["logits"])
        if figure is not None:
            display(figure)
            plt.close(figure)
        actual, predicted, scores = models.inspect_last_validation_prediction(run["logits"], labels)
        print(f"Actual digit: {actual} | Predicted digit: {predicted}")
        display(scores)
        best = run["history"][run["best_epoch"] - 1]
        summary.append({"activation": activation, "run_id": run["run_id"],
                        "best_epoch": run["best_epoch"], "val_loss": best["val_loss"],
                        "val_accuracy": best["val_accuracy"], "macro_f1": per_digit["F1"].mean()})
    display(pd.DataFrame(summary))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for activation, color in (("relu", "tab:blue"), ("tanh", "tab:orange")):
        if activation not in runs:
            continue
        history = pd.DataFrame(runs[activation]["history"])
        for split, style in (("train", "-"), ("val", "--")):
            for ax, metric in zip(axes, ("loss", "accuracy")):
                ax.plot(history.epoch, history[f"{split}_{metric}"], color=color, linestyle=style,
                        marker="o", label=f"{activation} {split}")
    axes[0].set(title="Historical baseline — loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="Historical baseline — accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].axhline(.1, color="crimson", linestyle="--", label="Chance reference (10%)")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    figure.tight_layout()
    display(figure)
    plt.close(figure)


def show_results(sweep_directory=None, *, historical_runs=None):
    """One final notebook report; never starts training or model inference."""
    if sweep_directory is not None:
        directory = Path(sweep_directory)
        status = json.loads((directory / "status.json").read_text())
        display(Markdown("### Three-epoch warm-up sweep"))
        print("Status:", status["status"], "| completed:", len(status["completed_runs"]), "/ 8")
        if status.get("partial_run"):
            print("Partial run (not a completed three-epoch result):", status["partial_run"])
        if status.get("active_run"):
            print("Active:", status["active_run"])
        if status["completed_runs"]:
            display(pd.DataFrame(status["completed_runs"]))
        for name in ("learning_rate_comparison.png", "epoch_comparison.png"):
            if (directory / name).is_file():
                display(Image(filename=str(directory / name)))
    if historical_runs:
        display(Markdown("### Preserved historical ten-epoch baseline"))
        _show_baseline_runs(historical_runs)
