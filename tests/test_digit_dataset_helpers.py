"""Test acquisition, loading, and column validation without network access."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.dataset import digit_dataset_helpers as helpers


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data_dir = Path(temporary.name) / "raw"
        self.files = {
            name: self.data_dir / f"{name}.csv"
            for name in ("train", "test", "sample_submission")
        }
        self.tables = {
            "train": pd.DataFrame({"label": [7], "pixel0": [12]}),
            "test": pd.DataFrame({"pixel0": [34]}),
            "sample_submission": pd.DataFrame({"ImageId": [1], "Label": [0]}),
        }
        for attribute, value in (("DATA_DIR", self.data_dir), ("FILES", self.files)):
            patcher = patch.object(helpers, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(helpers.kagglehub, "competition_download")
        self.downloader = patcher.start()
        self.addCleanup(patcher.stop)

    def write_table(self, name):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tables[name].to_csv(self.files[name], index=False)

    def assert_tables_equal(self, actual):
        self.assertEqual(set(actual), set(self.tables))
        for name, expected in self.tables.items():
            pd.testing.assert_frame_equal(actual[name], expected)

    def test_existing_files_load_without_downloading(self):
        for name in self.tables:
            self.write_table(name)

        self.assert_tables_equal(helpers.load_datasets())
        self.downloader.assert_not_called()

    def test_only_missing_files_are_downloaded_and_loaded(self):
        self.write_table("train")
        original_train = self.files["train"].read_bytes()
        downloaded = []

        def download(competition, path, output_dir):
            self.assertEqual(competition, "digit-recognizer")
            name = Path(path).stem
            downloaded.append(name)
            self.tables[name].to_csv(Path(output_dir) / path, index=False)

        self.downloader.side_effect = download

        self.assert_tables_equal(helpers.load_datasets())
        self.assertCountEqual(downloaded, ["test", "sample_submission"])
        self.assertEqual(self.files["train"].read_bytes(), original_train)

    def test_download_failure_stops_loading_and_preserves_cause(self):
        failure = ConnectionError("Simulated connection failure")
        self.downloader.side_effect = failure

        with patch.object(helpers.pd, "read_csv") as read_csv:
            with self.assertRaises(RuntimeError) as raised:
                helpers.load_datasets()
            read_csv.assert_not_called()

        causes = []
        error = raised.exception
        while error is not None:
            causes.append(error)
            error = error.__cause__
        self.assertIs(causes[-1], failure)
        self.assertIn("loading", str(raised.exception))
        self.assertTrue(any("train.csv" in str(error) for error in causes))


class DatasetColumnTests(unittest.TestCase):
    def make_datasets(self, rows):
        pixels = [f"pixel{i}" for i in range(784)]
        return {
            "train": pd.DataFrame(index=range(rows), columns=["label", *pixels]),
            "test": pd.DataFrame(index=range(rows + 1), columns=pixels),
        }

    def test_matching_columns_allow_varying_row_counts(self):
        for rows in (0, 1, 3):
            with self.subTest(rows=rows):
                helpers.validate_dataset_columns(self.make_datasets(rows))

    def test_incorrect_columns_are_rejected(self):
        for name in ("train", "test"):
            for problem in ("missing", "extra", "duplicate", "shuffled", "renamed"):
                with self.subTest(dataset=name, problem=problem):
                    datasets = self.make_datasets(0)
                    columns = list(datasets[name].columns)
                    if problem == "missing":
                        columns.pop()
                    elif problem == "extra":
                        columns.append("unexpected")
                    elif problem == "duplicate":
                        columns[-1] = columns[0]
                    elif problem == "shuffled":
                        columns[0], columns[1] = columns[1], columns[0]
                    else:
                        columns[-1] = "wrong_name"
                    datasets[name] = pd.DataFrame(columns=columns)
                    with self.assertRaisesRegex(ValueError, f"Unexpected {name} columns"):
                        helpers.validate_dataset_columns(datasets)


class ImageValueTests(unittest.TestCase):
    def make_datasets(self):
        pixels = [f"pixel{i}" for i in range(784)]
        train = pd.DataFrame(0.0, index=range(2), columns=pixels)
        train.insert(0, "label", ["not checked", None])
        test = pd.DataFrame(255.0, index=range(1), columns=pixels)
        return {"train": train, "test": test}

    def test_boundaries_and_whole_floats_are_valid_and_labels_are_ignored(self):
        datasets = self.make_datasets()
        datasets["train"].loc[0, "pixel783"] = 12.0
        helpers.validate_image_values(datasets)

    def test_invalid_pixels_identify_table_and_condition(self):
        cases = [
            (np.nan, "finite"),
            (np.inf, "finite"),
            (-np.inf, "finite"),
            (-1, "0–255"),
            (256, "0–255"),
            (12.5, "whole numbers"),
            ("invalid", "numeric"),
        ]
        for name in ("train", "test"):
            for value, condition in cases:
                with self.subTest(table=name, value=value):
                    datasets = self.make_datasets()
                    if isinstance(value, str):
                        datasets[name]["pixel783"] = datasets[name]["pixel783"].astype(object)
                    datasets[name].loc[0, "pixel783"] = value
                    with self.assertRaisesRegex(ValueError, f"{name} pixels:.*{condition}"):
                        helpers.validate_image_values(datasets)


class LabelValueTests(unittest.TestCase):
    def test_valid_boundaries_and_whole_floats_ignore_pixels_and_other_tables(self):
        datasets = {
            "train": pd.DataFrame({"label": [0.0, 4.0, 9.0], "pixel0": [None, "bad", -1]}),
            "test": pd.DataFrame({"label": ["ignored"]}),
            "sample_submission": pd.DataFrame({"Label": ["ignored"]}),
        }
        helpers.validate_labels(datasets)
        helpers.validate_labels({"train": pd.DataFrame({"label": [0, 9]})})

    def test_invalid_labels_identify_condition(self):
        cases = [
            (None, "finite"),
            (np.nan, "finite"),
            (np.inf, "finite"),
            (-np.inf, "finite"),
            ("invalid", "numeric"),
            (3.5, "whole numbers"),
            (-1, "0–9"),
            (10, "0–9"),
        ]
        for value, condition in cases:
            with self.subTest(value=value):
                datasets = {"train": pd.DataFrame({"label": [0, value]})}
                with self.assertRaisesRegex(ValueError, f"train labels:.*{condition}"):
                    helpers.validate_labels(datasets)


class DatasetOverviewTests(unittest.TestCase):
    def test_overview_returns_only_shapes_and_sorted_training_label_counts(self):
        datasets = {
            "train": pd.DataFrame({"label": [9, 0, 9], "pixel0": [0, 1, 2]}),
            "test": pd.DataFrame({"pixel0": [3, 4]}),
        }

        overview = helpers.build_dataset_overview(datasets)

        self.assertEqual(overview, {
            "train": {"shape": (3, 2), "label_counts": {0: 1, 9: 2}},
            "test": {"shape": (2, 1)},
        })
        self.assertEqual(list(overview["train"]["label_counts"]), [0, 9])


if __name__ == "__main__":
    unittest.main()
