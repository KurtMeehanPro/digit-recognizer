"""Run the authorized serial, three-epoch bespoke activation/learning-rate sweep.

No benchmark, concurrency, architecture, precision, or batch-size changes.
Use --run explicitly. Existing notebooks and artifact files are never rewritten.
"""
import argparse
import fcntl
import os
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from uuid import uuid4
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS, PROJECT_ROOT
from scripts.dataset.digit_dataset_augmentation import rotation_configuration
from scripts.model.digit_model_training_options import linear_warmup_rate, regularization_configuration
from scripts.model.digit_model_train import (
    build_data_loaders, build_model, get_training_device,
    seed_training, train_model,
)

from utilities.digit_run_artifacts import (allocate_run, iter_validation_paths, load_run, save_run_record, write_run_readme)

TARGET_RATES = (1e-6, 1e-5, 1e-4, 1e-3)
ACTIVATIONS = ("relu", "tanh")
MODEL_NAME = "bespoke_upsampling"
SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
VALIDATION_FRACTION = 0.2
WARMUP_FRACTION = 0.1
WEIGHT_DECAY = 0.0
ROTATION_DEGREES = 0.0
AUGMENTATION_SEED = 42
ROTATION_ANGLES = None
L2_STRENGTH = 0.0
DECOUPLED_WEIGHT_DECAY = 0.0


def unique_run_id():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def run_paths(activation, run_id, root=PROJECT_ROOT, model_name="bespoke_upsampling"):
    """Use the existing model/validation pairing convention; refuse reused paths."""
    if model_name not in {"bespoke_upsampling", "bespoke"}:
        raise ValueError("Unsupported bespoke model name")
    return allocate_run(run_id, root)


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def order_fingerprint(loader):
    """Probe the identical sampler on indices without consuming the real generator."""
    generator = torch.Generator()
    generator.set_state(loader.generator.get_state())
    probe = DataLoader(TensorDataset(torch.arange(len(loader.dataset))),
                       batch_size=loader.batch_size, shuffle=True, generator=generator)
    indices = torch.cat([batch[0] for batch in probe])
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def save_comparison(directory, summaries, runs):
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for activation, color in (("relu", "tab:blue"), ("tanh", "tab:orange")):
        rows = [row for row in summaries if row["activation"] == activation]
        if not rows:
            continue
        rates = [row["target_learning_rate"] for row in rows]
        axes[0].plot(rates, [row["val_loss"] for row in rows], marker="o", color=color, label=activation)
        axes[1].plot(rates, [row["val_accuracy"] for row in rows], marker="o", color=color, label=activation)
    axes[0].set(title=f"Best validation loss — {EPOCHS} epochs with {WARMUP_FRACTION:.0%} warm-up", ylabel="Cross-entropy loss")
    axes[1].set(title="Best-checkpoint validation accuracy", ylabel="Accuracy", ylim=(0, 1))
    axes[1].axhline(.1, color="crimson", linestyle="--", label="Chance reference (10%)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Target learning rate")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_rate_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(TARGET_RATES), 2, figsize=(13, 4 * len(TARGET_RATES)), sharex=True, squeeze=False)
    for index, rate in enumerate(TARGET_RATES):
        for run in runs:
            if run["training_config"]["learning_rate"] != rate:
                continue
            activation = run["model_config"]["activation"]
            h = pd.DataFrame(run["history"])
            for split, style in (("train", "-"), ("val", "--")):
                label = f"{activation} {split}"
                color = "tab:blue" if activation == "relu" else "tab:orange"
                axes[index, 0].plot(h.epoch, h[f"{split}_loss"], style, marker="o", color=color, label=label)
                axes[index, 1].plot(h.epoch, h[f"{split}_accuracy"], style, marker="o", color=color, label=label)
        axes[index, 0].set(title=f"Target {rate:g}: loss", ylabel="Cross-entropy loss")
        axes[index, 1].set(title=f"Target {rate:g}: accuracy", ylabel="Accuracy", ylim=(0, 1))
        axes[index, 1].axhline(.1, color="crimson", linestyle=":", label="Chance reference (10%)")
        for ax in axes[index]:
            ax.set_xticks(range(1, EPOCHS + 1))
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.3)
            if ax.lines:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "epoch_comparison.png", dpi=150)
    plt.close(fig)


def capture_protected_files(root=PROJECT_ROOT):
    """Protect prior result artifacts and notebook 02, not editable run interfaces.

    Notebook 03 autosaves outputs during its own run. Evaluation notes are also
    editable. Neither is an immutable training artifact. Model/training source
    used by a run is identified by hashes; project source remains in Git.
    """
    files = [*root.glob("data/models/*.pt"), *root.glob("data/validation_outputs/*.pt"), *root.glob("data/runs/*/*.pt"),
             root / "notebooks/02_lenet_style_cnn.ipynb"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files if path.is_file()}


def verify_protected_files(fingerprints):
    """Reject changed/deleted prior result artifacts."""
    for name, expected in fingerprints.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Protected file changed: {path}")


def execute_sweep(directory, resume=False):
    device = get_training_device()
    if device.type != "mps":
        raise RuntimeError(f"Expected authorized Mac Studio MPS device; got {device}")
    train = pd.read_csv(PROJECT_ROOT / "data/raw/train.csv")
    dataset_hash = hashlib.sha256(train[["label", *PIXEL_COLUMNS]].to_numpy().tobytes()).hexdigest()
    protected_hashes = capture_protected_files()
    source_paths = [Path(__file__), PROJECT_ROOT / "scripts/model/digit_model_train.py",
                    PROJECT_ROOT / "scripts/model/digit_model_definitions.py",
                    PROJECT_ROOT / "scripts/model/digit_model_metrics.py",
                    PROJECT_ROOT / "utilities/digit_run_artifacts.py"]
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                   "pandas": pd.__version__, "sklearn": sklearn.__version__, "device": str(device)}
    status = {"status": "running", "mode": "sequential", "architecture": ("unchanged 224-upsample eight-stage bespoke" if MODEL_NAME == "bespoke_upsampling" else "28-input five-stage BespokeCNN"),
              "model_name": MODEL_NAME, "regularization": regularization_configuration(L2_STRENGTH, DECOUPLED_WEIGHT_DECAY), "augmentation": rotation_configuration(ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES),
              "target_rates": TARGET_RATES, "activations": ACTIVATIONS, "epochs_per_run": EPOCHS,
              "warmup_fraction": WARMUP_FRACTION, "weight_decay": WEIGHT_DECAY,
              "source_sha256": source_hashes, "environment": environment, "completed_runs": []}
    summaries, runs = [], []
    if resume:
        prior = json.loads((directory / "status.json").read_text())
        status["resume_events"] = prior.get("resume_events", []) + [{
            "utc": datetime.now(ZoneInfo("UTC")).isoformat(),
            "previous_active_run": prior.get("active_run"),
            "reason": "Previous runner absent after tool transport disconnect; incomplete run restarts with fresh ID",
        }]
        # Recover completed artifacts even if the process ended before updating status.
        for path in iter_validation_paths(PROJECT_ROOT):
            saved = load_run(path)
            if saved.get("sweep_directory") != str(directory):
                continue
            if len(saved["history"]) != EPOCHS or not Path(saved["checkpoint_path"]).is_file():
                raise RuntimeError(f"Incomplete saved artifact requires inspection: {path}")
            if saved["model_config"]["model_name"] != MODEL_NAME:
                raise RuntimeError("Saved model differs; choose a new experiment directory")
            config = saved["training_config"]
            expected = {"seed": SEED, "validation_fraction": VALIDATION_FRACTION,
                        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "optimizer": "AdamW" if DECOUPLED_WEIGHT_DECAY else "Adam",
                        "weight_decay": WEIGHT_DECAY}
            if any(config.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Resume configuration differs from saved runs; choose a new sweep directory")
            if config.get("regularization", regularization_configuration()) != regularization_configuration(L2_STRENGTH, DECOUPLED_WEIGHT_DECAY):
                raise RuntimeError("Resume regularization differs; choose a new experiment directory")
            if config.get("augmentation", rotation_configuration()) != rotation_configuration(ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES):
                raise RuntimeError("Resume augmentation differs; choose a new experiment directory")
            if config["schedule"]["fraction"] != WARMUP_FRACTION:
                raise RuntimeError("Resume warmup differs from saved runs; choose a new sweep directory")
            if config["learning_rate"] not in TARGET_RATES or saved["model_config"]["activation"] not in ACTIVATIONS:
                raise RuntimeError("Resume search space differs from saved runs; choose a new sweep directory")
            best = saved["history"][saved["best_epoch"] - 1]
            predictions = saved["logits"].argmax(1)
            matrix = torch.bincount(saved["labels"] * 10 + predictions, minlength=100).reshape(10, 10).double()
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = torch.where(denominator > 0, 2 * matrix.diag() / denominator, 0).mean().item()
            summaries.append({"run_id": saved["run_id"], "activation": saved["model_config"]["activation"],
                              "target_learning_rate": saved["training_config"]["learning_rate"],
                              "best_epoch": saved["best_epoch"], "val_loss": best["val_loss"],
                              "val_accuracy": best["val_accuracy"], "macro_f1": f1,
                              "elapsed_seconds": saved["elapsed_seconds"], "checkpoint_path": saved["checkpoint_path"],
                              "validation_outputs": str(path)})
            runs.append(saved)
        keys = [(row["target_learning_rate"], row["activation"]) for row in summaries]
        if len(set(keys)) != len(keys):
            raise RuntimeError("Duplicate completed configurations; refusing automatic resume")
        status["completed_runs"] = summaries
    expected_init = runs[0]["initial_weights_sha256"] if runs else None
    expected_order = runs[0]["first_epoch_order_sha256"] if runs else None
    sweep_start = time.perf_counter()
    for rate in TARGET_RATES:
        for activation in ACTIVATIONS:
            completed = next((row for row in summaries if row["activation"] == activation
                              and row["target_learning_rate"] == rate), None)
            if completed is not None:
                print(f"Already completed, no retraining | {activation} | target_lr={rate:g} | "
                      f"validation_accuracy={completed['val_accuracy']:.2%}", flush=True)
                continue
            seed_training(SEED)
            train_loader, val_loader = build_data_loaders(train, VALIDATION_FRACTION, BATCH_SIZE, SEED,
                                                         ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES)
            model_config = {"model_name": MODEL_NAME, "activation": activation}
            if MODEL_NAME == "bespoke_upsampling":
                model_config["upsample_mode"] = "bilinear"
            model = build_model(**model_config)
            initial_hash, order_hash = state_fingerprint(model), order_fingerprint(train_loader)
            expected_init = expected_init or initial_hash
            expected_order = expected_order or order_hash
            assert initial_hash == expected_init and order_hash == expected_order
            run_id = unique_run_id()
            checkpoint_path, validation_path = run_paths(activation, run_id, root=PROJECT_ROOT, model_name=MODEL_NAME)
            total_steps = EPOCHS * len(train_loader)
            warmup_steps = math.ceil(WARMUP_FRACTION * total_steps)
            schedule = {"type": "linear_warmup_then_constant", "fraction": WARMUP_FRACTION,
                        "total_optimizer_steps": total_steps, "warmup_steps": warmup_steps,
                        "step_numbering": "one-based optimizer updates; no reset between epochs",
                        "rate_at_step_zero": 0.0, "first_update_rate": linear_warmup_rate(rate, 1, warmup_steps),
                        "target_rate": rate, "target_reached_at_step": warmup_steps, "decay": "none"}
            status["active_run"] = {"run_id": run_id, "activation": activation, "target_learning_rate": rate}
            write_json(directory / "status.json", status)
            print(f"\nSTART {run_id} | {activation} | target_lr={rate:g} | {EPOCHS} epochs | warmup={warmup_steps}/{total_steps} steps", flush=True)
            synchronize(device)
            run_start = time.perf_counter()
            run_directory = checkpoint_path.parent
            write_json(run_directory / "config.json", {
                "model_config": model_config, "seed": SEED, "validation_fraction": VALIDATION_FRACTION,
                "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": rate,
                "schedule": schedule, "augmentation": rotation_configuration(ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES),
                "regularization": regularization_configuration(L2_STRENGTH, DECOUPLED_WEIGHT_DECAY),
                "optimizer": "AdamW" if DECOUPLED_WEIGHT_DECAY else "Adam", "weight_decay": WEIGHT_DECAY})
            write_json(run_directory / "record.json", {"run_id": run_id, "experiment_group": str(directory)})
            write_run_readme(run_directory, status="running")
            try:
                with (run_directory / "run.log").open("x") as run_log, redirect_stdout(Tee(sys.stdout, run_log)), redirect_stderr(Tee(sys.stderr, run_log)):
                    model, history, best_epoch, logits, labels = train_model(
                        model, train_loader, val_loader, torch.nn.CrossEntropyLoss(reduction="none"),
                        EPOCHS, checkpoint_path, learning_rate=rate,
                        warmup_fraction=WARMUP_FRACTION, weight_decay=WEIGHT_DECAY,
                        l2_strength=L2_STRENGTH, decoupled_weight_decay=DECOUPLED_WEIGHT_DECAY,
                    )
            except BaseException as error:
                write_json(run_directory / "record.json", {"run_id": run_id, "error": str(error), "experiment_group": str(directory)})
                write_run_readme(run_directory, status="interrupted_or_failed")
                raise
            synchronize(device)
            elapsed = time.perf_counter() - run_start
            val_images, val_labels = val_loader.dataset.tensors
            val_hash = hashlib.sha256(val_images.numpy().tobytes() + val_labels.numpy().tobytes()).hexdigest()
            artifact = {"run_id": run_id, "checkpoint_path": str(checkpoint_path), "logits": logits,
                        "labels": labels, "best_epoch": best_epoch, "seed": SEED,
                        "validation_fraction": VALIDATION_FRACTION, "model_config": model_config,
                        "training_config": {"seed": SEED, "validation_fraction": VALIDATION_FRACTION,
                                            "batch_size": BATCH_SIZE, "learning_rate": rate, "epochs": EPOCHS,
                                            "optimizer": "AdamW" if DECOUPLED_WEIGHT_DECAY else "Adam", "weight_decay": WEIGHT_DECAY,
                                            "regularization": regularization_configuration(L2_STRENGTH, DECOUPLED_WEIGHT_DECAY),
                                            "checkpoint_selection": "lowest_validation_loss", "schedule": schedule,
                                            "augmentation": rotation_configuration(ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES),
                                            "training_examples_per_epoch": len(train_loader.dataset),
                                            "optimizer_steps_per_epoch": len(train_loader),
                                            "validation_examples": len(val_loader.dataset)},
                        "environment": environment, "dataset_sha256": dataset_hash, "validation_sha256": val_hash,
                        "initial_weights_sha256": initial_hash, "first_epoch_order_sha256": order_hash,
                        "source_sha256": source_hashes, "history": history, "elapsed_seconds": elapsed,
                        "sweep_directory": str(directory), "execution_mode": "sequential"}
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            with validation_path.open("xb") as output:
                torch.save(artifact, output)
            save_run_record(run_directory, artifact, status="completed")
            best = history[best_epoch - 1]
            predictions = logits.argmax(1)
            matrix = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10).double()
            denominator = matrix.sum(0) + matrix.sum(1)
            f1 = torch.where(denominator > 0, 2 * matrix.diag() / denominator, 0).mean().item()
            summary = {"run_id": run_id, "activation": activation, "target_learning_rate": rate,
                       "best_epoch": best_epoch, "val_loss": best["val_loss"], "val_accuracy": best["val_accuracy"],
                       "macro_f1": f1, "elapsed_seconds": elapsed, "checkpoint_path": str(checkpoint_path),
                       "validation_outputs": str(validation_path)}
            summaries.append(summary)
            runs.append(artifact)
            status["completed_runs"] = summaries
            write_json(directory / "status.json", status)
            save_comparison(directory, summaries, runs)
            print(f"DONE {run_id} | best_epoch={best_epoch} | val_accuracy={best['val_accuracy']:.6%} | {elapsed:.1f}s", flush=True)
            del model, train_loader, val_loader
            torch.mps.empty_cache()
    verify_protected_files(protected_hashes)
    status.update(status="completed", elapsed_seconds=time.perf_counter() - sweep_start,
                  protected_files_verified=len(protected_hashes))
    status.pop("active_run", None)
    write_json(directory / "status.json", status)
    print(f"\nSWEEP COMPLETE: {len(summaries)} runs; protected files verified; {directory}", flush=True)


def active_sweep_directories():
    """Read managed-run locks without starting or interrupting any work."""
    active = []
    for lock_path in (PROJECT_ROOT / "data/runs/experiments").glob("bespoke_lr_sweep_*/.sweep.lock"):
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active.append(lock_path.parent)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return active


def main():
    global EPOCHS, TARGET_RATES, WARMUP_FRACTION, WEIGHT_DECAY, MODEL_NAME, ACTIVATIONS, ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES, L2_STRENGTH, DECOUPLED_WEIGHT_DECAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute all eight authorized serial runs")
    parser.add_argument("--resume-directory", type=Path, help="Recover completed runs and restart incomplete configurations with new IDs")
    parser.add_argument("--model", choices=("bespoke_upsampling", "bespoke"), default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=TARGET_RATES)
    parser.add_argument("--warmup-fraction", type=float, default=WARMUP_FRACTION)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--activations", nargs="+", choices=("relu", "tanh"), default=ACTIVATIONS)
    parser.add_argument("--rotation-degrees", type=float, default=0.0)
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--rotation-angles", type=float, nargs="+", default=None)
    parser.add_argument("--l2-strength", type=float, default=0.0)
    parser.add_argument("--decoupled-weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    if args.epochs < 1 or any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
        parser.error("Epochs and learning rates must be positive")
    if not 0 <= args.warmup_fraction <= 1 or args.weight_decay != 0:
        parser.error("Warmup must be between 0 and 1; this experiment requires zero weight decay")
    if not math.isfinite(args.rotation_degrees) or not 0 <= args.rotation_degrees <= 180:
        parser.error("Rotation degrees must be finite and between 0 and 180")
    if args.rotation_angles is not None:
        if args.rotation_degrees or not args.rotation_angles or len(set(args.rotation_angles)) != len(args.rotation_angles) or any(not math.isfinite(a) or abs(a) > 180 for a in args.rotation_angles):
            parser.error("Fixed angles must be unique finite degrees within [-180, 180], without random rotation")
    ROTATION_ANGLES = tuple(args.rotation_angles) if args.rotation_angles is not None else None
    try:
        regularization_configuration(args.l2_strength, args.decoupled_weight_decay)
    except ValueError as error:
        parser.error(str(error))
    L2_STRENGTH, DECOUPLED_WEIGHT_DECAY = args.l2_strength, args.decoupled_weight_decay
    MODEL_NAME = args.model
    ACTIVATIONS = tuple(args.activations)
    ROTATION_DEGREES, AUGMENTATION_SEED = args.rotation_degrees, args.augmentation_seed
    EPOCHS, TARGET_RATES = args.epochs, tuple(args.learning_rates)
    WARMUP_FRACTION, WEIGHT_DECAY = args.warmup_fraction, args.weight_decay
    active = active_sweep_directories()
    if active:
        parser.error("A managed sweep is already running; no duplicate training started: " + str(active[0]))
    if args.resume_directory:
        directory = args.resume_directory.resolve()
        if directory.parent != (PROJECT_ROOT / "data/runs/experiments").resolve():
            raise ValueError("Resume directory must be an existing project experiment")
        if not (directory / "status.json").is_file():
            raise ValueError("Resume status file not found")
        prior_status = json.loads((directory / "status.json").read_text())
        if prior_status.get("regularization", regularization_configuration()) != regularization_configuration(L2_STRENGTH, DECOUPLED_WEIGHT_DECAY):
            parser.error("Saved experiment uses different regularization; choose a new results directory")
        if prior_status.get("augmentation", rotation_configuration()) != rotation_configuration(ROTATION_DEGREES, AUGMENTATION_SEED, ROTATION_ANGLES):
            parser.error("Saved experiment uses different augmentation; choose a new results directory")
        prior_model = prior_status.get("model_name", "bespoke_upsampling")
        if prior_model != MODEL_NAME:
            parser.error("Saved experiment uses a different model; choose a new results directory")
    else:
        directory = PROJECT_ROOT / "data/runs/experiments" / f"bespoke_lr_sweep_{unique_run_id()}"
        directory.mkdir(parents=True, exist_ok=False)
    lock = (directory / ".sweep.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    with (directory / "sweep.log").open("a" if args.resume_directory else "x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"Learning-rate testing results: {directory}", flush=True)
            try:
                execute_sweep(directory, resume=bool(args.resume_directory))
            except BaseException as error:
                status_path = directory / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                status.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(status_path, status)
                raise


if __name__ == "__main__":
    main()
