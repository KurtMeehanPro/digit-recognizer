"""Run dataset checks and print the overview."""

from digit_dataset_helpers import (
    build_dataset_overview,
    load_datasets,
    validate_dataset_columns,
    validate_image_values,
    validate_labels,
)


def main() -> None:
    """Load once, validate columns and values, then print shapes and label counts."""
    # Load
    datasets = load_datasets()

    # Validate
    validate_dataset_columns(datasets)
    print("Column names and order are as expected.")
    validate_image_values(datasets)
    print("Pixel values are as expected.")
    validate_labels(datasets)
    print("Training labels are as expected.")

    # Print overview
    overview = build_dataset_overview(datasets)
    print(f"\ntrain.csv shape: {overview['train']['shape']}")
    print("Training label counts:")
    for digit, count in overview["train"]["label_counts"].items():
        print(f"  {digit}: {count}")
    print(f"\ntest.csv shape: {overview['test']['shape']}")


if __name__ == "__main__":
    main()
