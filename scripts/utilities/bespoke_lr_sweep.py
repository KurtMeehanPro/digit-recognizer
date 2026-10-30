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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS, PROJECT_ROOT
from scripts.model.digit_model_helpers import (
    build_data_loaders, build_model, get_training_device, linear_warmup_rate,
    seed_training, train_model,
)

TARGET_RATES = (1e-6, 1e-5, 1e-4, 1e-3)
ACTIVATIONS = ("relu", "tanh")
SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
VALIDATION_FRACTION = 0.2
WARMUP_FRACTION = 0.1
WEIGHT_DECAY = 0.0


def unique_run_id():
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def run_paths(activation, run_id, root=PROJECT_ROOT):
    """Use the existing model/validation pairing convention; refuse reused paths."""
    paths = (root / "data/models" / f"bespoke_{activation}_{run_id}_best.pt",
             root / "data/validation_outputs" / f"bespoke_{activation}_{run_id}_validation.pt")
    if any(path.exists() for path in paths):
        raise FileExistsError("Run ID already has an artifact; refusing overwrite")
    return paths


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
    axes[0].set(title="Best validation loss — 3 epochs with 10% warm-up", ylabel="Cross-entropy loss")
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
    fig, axes = plt.subplots(4, 2, figsize=(13, 16), sharex=True)
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
            ax.set_xticks((1, 2, 3))
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.3)
            if ax.lines:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "epoch_comparison.png", dpi=150)
    plt.close(fig)


def execute_sweep(directory, resume=False):
    device = get_training_device()
    if device.type != "mps":
        raise RuntimeError(f"Expected authorized Mac Studio MPS device; got {device}")
    train = pd.read_csv(PROJECT_ROOT / "data/raw/train.csv")
    dataset_hash = hashlib.sha256(train[["label", *PIXEL_COLUMNS]].to_numpy().tobytes()).hexdigest()
    protected = [*PROJECT_ROOT.glob("data/models/*.pt"), *PROJECT_ROOT.glob("data/validation_outputs/*.pt")]
    protected += [PROJECT_ROOT / "notebooks/02_lenet_style_cnn.ipynb", PROJECT_ROOT / "notebooks/03_bespoke_cnn.ipynb",
                  PROJECT_ROOT / "docs/evaluations/bespoke-first-run.md", PROJECT_ROOT / "docs/model_implementation_choices.md"]
    protected_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    source_paths = [Path(__file__), PROJECT_ROOT / "scripts/model/digit_model_helpers.py",
                    PROJECT_ROOT / "scripts/model/digit_model_definitions.py"]
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                   "pandas": pd.__version__, "sklearn": sklearn.__version__, "device": str(device)}
    status = {"status": "running", "mode": "sequential", "architecture": "unchanged 224-upsample eight-stage bespoke",
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
        for path in sorted((PROJECT_ROOT / "data/validation_outputs").glob("bespoke*_validation.pt")):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("sweep_directory") != str(directory):
                continue
            if len(saved["history"]) != EPOCHS or not Path(saved["checkpoint_path"]).is_file():
                raise RuntimeError(f"Incomplete saved artifact requires inspection: {path}")
            config = saved["training_config"]
            expected = {"seed": SEED, "validation_fraction": VALIDATION_FRACTION,
                        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "optimizer": "Adam",
                        "weight_decay": WEIGHT_DECAY}
            if any(config.get(key) != value for key, value in expected.items()):
                raise RuntimeError("Resume configuration differs from saved runs; choose a new sweep directory")
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
    snapshot = directory / "source_snapshots" / unique_run_id()
    snapshot.mkdir(parents=True, exist_ok=False)
    for source in source_paths:
        target = snapshot / source.relative_to(PROJECT_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    expected_init = runs[0]["initial_weights_sha256"] if runs else None
    expected_order = runs[0]["first_epoch_order_sha256"] if runs else None
    sweep_start = time.perf_counter()
    for rate in TARGET_RATES:
        for activation in ACTIVATIONS:
            if any(row["activation"] == activation and row["target_learning_rate"] == rate for row in summaries):
                print(f"KEEP completed {activation} target_lr={rate:g}; no retraining", flush=True)
                continue
            seed_training(SEED)
            train_loader, val_loader = build_data_loaders(train, VALIDATION_FRACTION, BATCH_SIZE, SEED)
            model_config = {"model_name": "bespoke", "activation": activation, "upsample_mode": "bilinear"}
            model = build_model(**model_config)
            initial_hash, order_hash = state_fingerprint(model), order_fingerprint(train_loader)
            expected_init = expected_init or initial_hash
            expected_order = expected_order or order_hash
            assert initial_hash == expected_init and order_hash == expected_order
            run_id = unique_run_id()
            checkpoint_path, validation_path = run_paths(activation, run_id)
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
            model, history, best_epoch, logits, labels = train_model(
                model, train_loader, val_loader, torch.nn.CrossEntropyLoss(reduction="none"),
                EPOCHS, checkpoint_path, learning_rate=rate,
                warmup_fraction=WARMUP_FRACTION, weight_decay=WEIGHT_DECAY,
            )
            synchronize(device)
            elapsed = time.perf_counter() - run_start
            val_images, val_labels = val_loader.dataset.tensors
            val_hash = hashlib.sha256(val_images.numpy().tobytes() + val_labels.numpy().tobytes()).hexdigest()
            artifact = {"run_id": run_id, "checkpoint_path": str(checkpoint_path), "logits": logits,
                        "labels": labels, "best_epoch": best_epoch, "seed": SEED,
                        "validation_fraction": VALIDATION_FRACTION, "model_config": model_config,
                        "training_config": {"seed": SEED, "validation_fraction": VALIDATION_FRACTION,
                                            "batch_size": BATCH_SIZE, "learning_rate": rate, "epochs": EPOCHS,
                                            "optimizer": "Adam", "weight_decay": WEIGHT_DECAY,
                                            "checkpoint_selection": "lowest_validation_loss", "schedule": schedule},
                        "environment": environment, "dataset_sha256": dataset_hash, "validation_sha256": val_hash,
                        "initial_weights_sha256": initial_hash, "first_epoch_order_sha256": order_hash,
                        "source_sha256": source_hashes, "history": history, "elapsed_seconds": elapsed,
                        "sweep_directory": str(directory), "execution_mode": "sequential"}
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            with validation_path.open("xb") as output:
                torch.save(artifact, output)
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
    for path, fingerprint in protected_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == fingerprint, f"Protected file changed: {path}"
    status.update(status="completed", elapsed_seconds=time.perf_counter() - sweep_start,
                  protected_files_verified=len(protected_hashes))
    status.pop("active_run", None)
    write_json(directory / "status.json", status)
    print(f"\nSWEEP COMPLETE: {len(summaries)} runs; protected files verified; {directory}", flush=True)


def active_sweep_directories():
    """Read managed-run locks without starting or interrupting any work."""
    active = []
    for lock_path in (PROJECT_ROOT / "data/experiments").glob("bespoke_lr_sweep_*/.sweep.lock"):
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                active.append(lock_path.parent)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return active


def main():
    global EPOCHS, TARGET_RATES, WARMUP_FRACTION, WEIGHT_DECAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute all eight authorized serial runs")
    parser.add_argument("--resume-directory", type=Path, help="Recover completed runs and restart incomplete configurations with new IDs")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=TARGET_RATES)
    parser.add_argument("--warmup-fraction", type=float, default=WARMUP_FRACTION)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    if args.epochs < 1 or any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
        parser.error("Epochs and learning rates must be positive")
    if not 0 <= args.warmup_fraction <= 1 or args.weight_decay != 0:
        parser.error("Warmup must be between 0 and 1; this experiment requires zero weight decay")
    EPOCHS, TARGET_RATES = args.epochs, tuple(args.learning_rates)
    WARMUP_FRACTION, WEIGHT_DECAY = args.warmup_fraction, args.weight_decay
    active = active_sweep_directories()
    if active:
        parser.error("A managed sweep is already running; no duplicate training started: " + str(active[0]))
    if args.resume_directory:
        directory = args.resume_directory.resolve()
        if directory.parent != (PROJECT_ROOT / "data/experiments").resolve():
            raise ValueError("Resume directory must be an existing project experiment")
        if not (directory / "status.json").is_file():
            raise ValueError("Resume status file not found")
    else:
        directory = PROJECT_ROOT / "data/experiments" / f"bespoke_lr_sweep_{unique_run_id()}"
        directory.mkdir(parents=True, exist_ok=False)
    lock = (directory / ".sweep.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    with (directory / "sweep.log").open("a" if args.resume_directory else "x", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"SWEEP_DIRECTORY {directory}", flush=True)
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
