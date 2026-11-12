"""Provide reusable helpers for downloading, loading, and describing the dataset."""

import kagglehub
import pandas as pd

from scripts.dataset.digit_dataset_config import COMPETITION, DATA_DIR, FILES


def ensure_dataset_downloaded() -> None:
    """Acquire missing competition files in the configured local data directory."""
    for path in FILES.values():
        if path.is_file():
            continue
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            kagglehub.competition_download(
                COMPETITION, path=path.name, output_dir=str(DATA_DIR)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {path.name} from Kaggle competition {COMPETITION}."
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(f"Expected competition file was not found: {path}")


def load_datasets() -> dict[str, pd.DataFrame]:
    """Return DataFrames keyed by train, test, and sample_submission, acquiring missing files."""
    try:
        ensure_dataset_downloaded()
    except Exception as exc:
        raise RuntimeError("Could not prepare dataset files for loading.") from exc
    return {name: pd.read_csv(path) for name, path in FILES.items()}


def build_dataset_overview(datasets) -> dict:
    """Return train/test shapes and training label counts in digit order."""
    return {
        "train": {
            "shape": datasets["train"].shape,
            "label_counts": datasets["train"]["label"].value_counts().sort_index().to_dict(),
        },
        "test": {"shape": datasets["test"].shape},
    }
