"""Load saved LeNet weights and predict digits from a competition-style pixel CSV."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Support direct execution as well as python -m.
import sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from scripts.dataset.digit_dataset_config import PIXEL_COLUMNS, PROJECT_ROOT
from scripts.model.digit_model_helpers import build_model, get_training_device


def predict_digits(input_path, model_checkpoint_path, output_path, batch_size=64,
                   model_config=None):
    """Load trained weights and write a predicted digit for each input image.

    Args:
        input_path: Location of a CSV with pixel0 through pixel783 columns.
            Each row is a flattened 28 × 28 grayscale image with pixel values from 0 to 255.
        model_checkpoint_path: Location of the saved LeNet weights, such as lenet_best.pt.
        output_path: Where to save the predictions CSV. Parent folders are created
            if needed; an existing file at this location is overwritten.
        batch_size: Number of images to process together, defaulting to 64.
        model_config: build_model keyword arguments; defaults to the original LeNet.

    Returns:
        A DataFrame containing ImageId (row numbers starting at 1) and Label
        (predicted digits from 0 through 9), also saved to output_path.
        Predictions follow the input CSV's row order. No training is performed.
    """
    frame = pd.read_csv(input_path)
    images = torch.tensor(
        frame[PIXEL_COLUMNS].to_numpy(), dtype=torch.float32,
    ).reshape(-1, 1, 28, 28) / 255.0
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)

    device = get_training_device()
    model = build_model(**(model_config or {"model_name": "lenet"})).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    predictions = []
    with torch.no_grad():
        for (batch_images,) in loader:
            logits = model(batch_images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    # create a DataFrame with the predictions
    results = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})

    # ensure the output directory exists before writing the CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # write the dataframe to csv
    results.to_csv(output_path, index=False)

    # print a message indicating the number of predictions saved
    print(f"Saved {len(results):,} predictions to {output_path}")

    #return dataframe containing predictions
    return results


def main():
    input_path = PROJECT_ROOT / "data/raw/test.csv"
    model_checkpoint_path = None
    output_path = PROJECT_ROOT / "data/predictions/lenet_predictions.csv"
    batch_size = 64

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=input_path)
    parser.add_argument("--checkpoint", type=Path, default=model_checkpoint_path,
                        help="Saved model weights; defaults to the newest LeNet checkpoint.")
    parser.add_argument("--output", type=Path, default=output_path)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--config-from", type=Path,
                        help="Validation-output file containing model_config and checkpoint_path.")
    args = parser.parse_args()
    model_config = None
    if args.config_from is not None:
        saved = torch.load(args.config_from, map_location="cpu", weights_only=True)
        model_config = saved.get("model_config", {"model_name": "lenet"})
        if args.checkpoint is None:
            args.checkpoint = Path(saved["checkpoint_path"])
    if args.checkpoint is None:
        checkpoints = list((PROJECT_ROOT / "data/models").glob("lenet*_best.pt"))
        if not checkpoints:
            parser.error("No LeNet checkpoint found. Train a model or provide --checkpoint.")
        args.checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    print(f"Loading model weights from: {args.checkpoint}")
    predict_digits(args.input, args.checkpoint, args.output, args.batch_size, model_config)


if __name__ == "__main__":
    main()
