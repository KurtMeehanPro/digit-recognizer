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
    print(f"BespokeCNN: activation={activation}, resize={upsample_mode}, input=(1, 1, 28, 28)")
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


def _show_run_details(run, loader):
    """Current run tables/figures from saved best-epoch outputs, without inference."""
    activation = run["model_config"]["activation"]
    rate = run["training_config"]["learning_rate"]
    display(Markdown(f"### {activation.upper()} · target {rate:g} · {run['run_id']}"))
    display(pd.DataFrame(run["history"]))
    figure = models.plot_training_history(pd.DataFrame(run["history"]), run["best_epoch"], chance_reference=.1)
    display(figure)
    plt.close(figure)
    labels = run["labels"]
    predictions = run["logits"].argmax(1)
    matrix = confusion_matrix(labels.numpy(), predictions.numpy(), labels=list(range(10)))
    figure, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{activation.upper()}, LR {rate:g} — best-checkpoint confusion matrix")
    figure.tight_layout()
    display(figure)
    plt.close(figure)
    display(metrics.build_per_digit_metrics(matrix).sort_values("F1"))
    figure = metrics.plot_misclassified_digit(None, loader, digit=3, max_images=10,
                                              validation_logits=run["logits"])
    if figure is not None:
        display(figure)
        plt.close(figure)
    actual, predicted, scores = models.inspect_last_validation_prediction(run["logits"], labels)
    print(f"Actual digit: {actual} | Predicted digit: {predicted}")
    display(scores)


def show_results(sweep_directory=None):
    """One current-experiment report; never starts training or model inference."""
    import hashlib
    if sweep_directory is None:
        print("Select a saved sweep directory after running the experiment.")
        return
    directory = Path(sweep_directory)
    status = json.loads((directory / "status.json").read_text())
    display(Markdown("### Three-epoch warm-up sweep"))
    print("Status:", status["status"], "| completed:", len(status["completed_runs"]), "/ 8")
    if status.get("partial_run"):
        partial = status["partial_run"]
        print("Partial run — not a completed three-epoch result:", partial)
        log = (directory / "sweep.log").read_text()
        segment = log.rsplit("START " + partial["run_id"], 1)[-1]
        print("\n".join(line for line in segment.splitlines() if line.startswith("Epoch ")))
    if status.get("active_run"):
        print("Active:", status["active_run"])
    if not status["completed_runs"]:
        print("No completed validation artifacts saved yet.")
        return
    display(pd.DataFrame(status["completed_runs"]))
    for name in ("learning_rate_comparison.png", "epoch_comparison.png"):
        if (directory / name).is_file():
            display(Image(filename=str(directory / name)))
    train = pd.read_csv(PROJECT_ROOT / "data/raw/train.csv")
    _, loader = models.build_data_loaders(train, validation_fraction=.2, batch_size=64, seed=42)
    images, labels = loader.dataset.tensors
    fingerprint = hashlib.sha256(images.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    for row in status["completed_runs"]:
        run = torch.load(row["validation_outputs"], map_location="cpu", weights_only=True)
        assert run["sweep_directory"] == str(directory)
        assert run["validation_sha256"] == fingerprint and torch.equal(run["labels"], labels)
        _show_run_details(run, loader)
