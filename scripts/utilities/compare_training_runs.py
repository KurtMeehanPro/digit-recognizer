"""Compare saved model weights and validation outputs from two training runs."""

import argparse
from pathlib import Path

import torch

# Support direct execution as well as python -m.
import sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scripts.dataset.digit_dataset_config import PROJECT_ROOT


def compare_training_runs(first_path, second_path):
    """Print exact equality checks for two validation files and their checkpoints."""
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    second = torch.load(second_path, map_location="cpu", weights_only=True)
    first_weights = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=True)
    second_weights = torch.load(second["checkpoint_path"], map_location="cpu", weights_only=True)

    print(f"First run: {Path(first_path).name}")
    print(f"Second run: {Path(second_path).name}")
    weights_equal = first_weights.keys() == second_weights.keys() and all(
        torch.equal(first_weights[key], second_weights[key]) for key in first_weights
    )
    print(f"Weights exactly equal: {weights_equal}")
    print(f"Labels exactly equal: {torch.equal(first['labels'], second['labels'])}")
    print(f"Logits exactly equal: {torch.equal(first['logits'], second['logits'])}")

    if first["logits"].shape != second["logits"].shape:
        print("Different logit shapes; element-by-element differences cannot be calculated.")
        return
    if not torch.equal(first["labels"], second["labels"]):
        print("Validation labels differ; confirm image alignment before interpreting differences.")
    print(f"Largest logit difference: {(first['logits'] - second['logits']).abs().max().item()}")
    changed = (first["logits"].argmax(1) != second["logits"].argmax(1)).sum().item()
    print(f"Changed predicted digits: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Two validation-output files; otherwise use the latest two complete runs.")
    args = parser.parse_args()
    if args.runs:
        if len(args.runs) != 2:
            parser.error("Provide exactly two files, or no files to use the latest two runs.")
        paths = args.runs
    else:
        folder = PROJECT_ROOT / "data" / "validation_outputs"
        paths = []
        for path in sorted(folder.glob("*.pt"), key=lambda p: p.stat().st_mtime):
            data = torch.load(path, map_location="cpu", weights_only=True)
            if "checkpoint_path" in data and Path(data["checkpoint_path"]).is_file():
                paths.append(path)
        if len(paths) < 2:
            parser.error("Fewer than two validation-output files with matching checkpoints were found.")
        paths = paths[-2:]
    compare_training_runs(*paths)


if __name__ == "__main__":
    main()
