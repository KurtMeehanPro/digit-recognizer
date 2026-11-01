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
    importlib.reload(metrics)
    importlib.reload(models)


def show_dataset_summary():
    for label, filename in (("Training", "train.csv"), ("Test", "test.csv")):
        frame = pd.read_csv(PROJECT_ROOT / "data/raw" / filename)
        print(f"{label} shape: {frame.shape}")


def inspect_bespoke_model(activation="relu", upsample_mode="bilinear", seed=42, model_name="bespoke"):
    """CPU-only shape inspection, no training or dataset access."""
    models.seed_training(seed)
    options = {"activation": activation}
    if model_name == "bespoke":
        options["upsample_mode"] = upsample_mode
    model = models.build_model(model_name, **options)
    print(f"{type(model).__name__}: activation={activation}, input=(1, 1, 28, 28)")
    with torch.no_grad():
        print("Output shape:", tuple(model(torch.zeros(1, 1, 28, 28)).shape))


def run_bespoke_sweep(*, epochs, learning_rates, warmup_fraction, weight_decay,
                       sweep_directory=None, model_name="bespoke", activations=None,
                       rotation_degrees=None, augmentation_seed=42, rotation_angles=None, l2_strength=0.0, decoupled_weight_decay=0.0):
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
               "--warmup-fraction", str(warmup_fraction), "--weight-decay", str(weight_decay),
               "--model", model_name]
    if activations is not None:
        command.extend(["--activations", *activations])
    if rotation_degrees is not None:
        command.extend(["--rotation-degrees", str(rotation_degrees), "--augmentation-seed", str(augmentation_seed)])
    if rotation_angles is not None:
        command.extend(["--rotation-angles", *map(str, rotation_angles), "--augmentation-seed", str(augmentation_seed)])
    if l2_strength or decoupled_weight_decay:
        command.extend(["--l2-strength", str(l2_strength), "--decoupled-weight-decay", str(decoupled_weight_decay)])
    selected = Path(sweep_directory) if sweep_directory is not None else None
    if selected is not None:
        command.extend(["--resume-directory", str(selected)])
    with subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                for prefix in ("Learning-rate testing results: ", "SWEEP_DIRECTORY "):
                    if line.startswith(prefix):
                        selected = Path(line.removeprefix(prefix).strip())
                        break
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
    peak = max(run["history"], key=lambda h: h["val_accuracy"])
    selected = run["history"][run["best_epoch"] - 1]
    print(f"Selected minimum-loss checkpoint: epoch {run['best_epoch']}, accuracy {selected['val_accuracy']:.2%}, loss {selected['val_loss']:.6f}")
    print(f"Peak observed accuracy: {peak['val_accuracy']:.2%}, epoch {peak['epoch']}; peak weights may not be retained.")
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
    title = "Bespoke Conventional CNN — learning-rate testing" if status.get("model_name") == "bespoke28" else "Three-epoch warm-up sweep"
    display(Markdown("### " + title))
    if "regularization" in status:
        print("Regularization:", status["regularization"])
    if "augmentation" in status:
        print("Training augmentation:", status["augmentation"])
    print("Status:", status["status"], "| completed:", len(status["completed_runs"]), "/", len(status.get("activations", ("relu", "tanh"))) * len(status.get("target_rates", (1e-6,1e-5,1e-4,1e-3))))
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


def run_rotation_experiment(*, rotation_ranges, epochs=3, learning_rate=.0001,
                            warmup_fraction=.1, weight_decay=0.0,
                            augmentation_seed=42, result_directories=None, activation="tanh"):
    """Explicit manual execution only; separate unique artifact group per range.

    Mutates the supplied map as each group completes so earlier results remain
    selectable if a later trial is interrupted. No model or validation changes.
    """
    selected = result_directories if result_directories is not None else {}
    for degrees in rotation_ranges:
        print(f"Training rotation range: ±{degrees:g}° (0 = unaugmented baseline)")
        selected[degrees] = run_bespoke_sweep(
            epochs=epochs, learning_rates=(learning_rate,), warmup_fraction=warmup_fraction,
            weight_decay=weight_decay, model_name="bespoke28", activations=(activation,),
            rotation_degrees=degrees, augmentation_seed=augmentation_seed,
            sweep_directory=selected.get(degrees))
    return selected


def show_rotation_results(result_directories):
    """Read-only comparison table plus existing detailed reports; no inference."""
    rows = []
    for degrees, directory in result_directories.items():
        status = json.loads((Path(directory) / "status.json").read_text())
        for run in status["completed_runs"]:
            rows.append({"rotation_degrees": degrees, **run})
    if not rows:
        print("No augmentation experiment has been run. Proposed ranges are settings, not results.")
        return
    display(pd.DataFrame(rows))
    for degrees, directory in result_directories.items():
        display(Markdown(f"## Training rotations ±{degrees:g}°"))
        show_results(directory)


def run_augmented_comparison(*, tanh_rate, relu_rates, rotation_angles, epochs=3,
                             warmup_fraction=.1, weight_decay=0.0,
                             augmentation_seed=42, saved_groups=(), fresh=False):
    """Tanh first, then ReLU rates; reuse only explicitly matching saved runs.

    fresh=True ignores saved groups and allocates a new group per configuration.
    No training happens until this function is explicitly called by the notebook.
    """
    expected_augmentation = models.rotation_configuration(0, augmentation_seed, rotation_angles)
    available = []
    if not fresh:
        for group in saved_groups:
            path = Path(group) / "status.json"
            if not path.is_file():
                continue
            status = json.loads(path.read_text())
            for row in status.get("completed_runs", []):
                run = torch.load(row["validation_outputs"], map_location="cpu", weights_only=True)
                c = run["training_config"]
                expected = {"epochs": epochs, "seed": 42, "batch_size": 64,
                            "validation_fraction": .2, "optimizer": "Adam",
                            "weight_decay": weight_decay, "augmentation": expected_augmentation,
                            "checkpoint_selection": "lowest_validation_loss"}
                if c.get("regularization", models.regularization_configuration()) == models.regularization_configuration() and run["model_config"]["model_name"] == "bespoke28" and all(c.get(k) == v for k,v in expected.items()) and c["schedule"]["fraction"] == warmup_fraction and Path(run["checkpoint_path"]).is_file():
                    available.append((row, Path(group)))
    selected = []
    for activation, rate in [("tanh", tanh_rate), *(("relu", rate) for rate in relu_rates)]:
        match = next(((row, group) for row,group in available
                      if row["activation"] == activation and row["target_learning_rate"] == rate), None)
        if match:
            row, group = match
            print(f"Already completed, no retraining | {activation} | target_lr={rate:g} | validation_accuracy={row['val_accuracy']:.2%} | {row['run_id']}")
        else:
            group = run_bespoke_sweep(epochs=epochs, learning_rates=(rate,), warmup_fraction=warmup_fraction,
                                      weight_decay=weight_decay, model_name="bespoke28", activations=(activation,),
                                      rotation_angles=rotation_angles, augmentation_seed=augmentation_seed)
            row = json.loads((group / "status.json").read_text())["completed_runs"][0]
        selected.append({**row, "results_directory": str(group), "reused": match is not None})
    return selected


def show_augmented_comparison(selected):
    """Selected minimum-loss checkpoint and peak observed accuracy are distinct."""
    if not selected:
        print("No results selected. Run the experiment/reuse cell to select matching saved runs.")
        return
    rows=[]
    for row in selected:
        run=torch.load(row["validation_outputs"], map_location="cpu", weights_only=True)
        peak=max(run["history"],key=lambda h:h["val_accuracy"])
        rows.append({"activation":row["activation"],"learning_rate":row["target_learning_rate"],
                     "selected_epoch":run["best_epoch"],"selected_accuracy":row["val_accuracy"],
                     "selected_loss":row["val_loss"],"peak_accuracy":peak["val_accuracy"],
                     "peak_epoch":peak["epoch"],"reused":row["reused"],"run_id":row["run_id"]})
    display(pd.DataFrame(rows))
    print("Saved weights/logits use minimum validation loss. Peak epoch weights may not be retained.")
    for directory in dict.fromkeys(row["results_directory"] for row in selected):
        show_results(directory)
