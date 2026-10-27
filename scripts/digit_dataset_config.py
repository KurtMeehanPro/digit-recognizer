"""Define dataset configuration, file paths, and expected data structure."""

from pathlib import Path

COMPETITION = "digit-recognizer"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FILES = {
    "train": DATA_DIR / "train.csv",
    "test": DATA_DIR / "test.csv",
    "sample_submission": DATA_DIR / "sample_submission.csv",
}
PIXEL_COLUMNS = [f"pixel{i}" for i in range(784)]
EXPECTED_COLUMNS = {
    "train": ["label", *PIXEL_COLUMNS],
    "test": PIXEL_COLUMNS,
}
