"""Integration checks with small, synthetic M5-shaped CSVs; no raw data required."""

import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5.data import DEFAULT_CONFIG, load_raw_data


class DataReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        identifiers = ["FOODS_1_001_CA_1_validation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"]
        for suffix, last_day in [("validation", 1913), ("evaluation", 1941)]:
            ids = [identifiers[0].replace("validation", suffix)] + identifiers[1:]
            self.write(
                f"sales_train_{suffix}.csv",
                ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
                + [f"d_{d}" for d in range(1, last_day + 1)],
                [ids + [0] * last_day, ids + [2] * last_day],
            )
        self.write("calendar.csv", [
            "date", "wm_yr_wk", "d", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ], [["2011-01-29", 11101, "d_1", 1, 1, 2011, "", "", "", "", 0, 0, 0]])
        self.write("sell_prices.csv", ["store_id", "item_id", "wm_yr_wk", "sell_price"],
                   [["CA_1", "FOODS_1_001", 11101, ""]])
        self.write("sample_submission.csv", ["id"] + [f"F{d}" for d in range(1, 29)],
                   [[identifiers[0]] + [0] * 28])

    def write(self, filename, columns, rows):
        with (self.root / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_reading_preserves_dates_counts_and_missing_prices(self):
        tables = load_raw_data(self.root)
        self.assertEqual(len(tables), 5)
        self.assertEqual(tables["sales_train_evaluation"].shape, (2, 1947))
        self.assertEqual(tables["sales_train_evaluation"].iloc[1]["d_1941"], 2)
        self.assertEqual(tables["calendar"]["date"].iloc[0], pd.Timestamp("2011-01-29"))
        self.assertTrue(pd.isna(tables["sell_prices"]["sell_price"].iloc[0]))
        self.assertEqual(len(load_raw_data(self.root, nrows=1)["sales_train_validation"]), 1)

    def test_explicit_directory_overrides_environment(self):
        with patch.dict("os.environ", {"M5_DATA_DIR": str(self.root / "missing")}):
            self.assertEqual(len(load_raw_data(self.root)), 5)
            with self.assertRaisesRegex(FileNotFoundError, "calendar.csv"):
                load_raw_data()

    def test_missing_file_and_wrong_schema_fail_clearly(self):
        (self.root / "sample_submission.csv").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "sample_submission.csv"):
            load_raw_data(self.root)
        self.write("sample_submission.csv", ["id", "F1"], [["example", 0]])
        with self.assertRaisesRegex(ValueError, "sample_submission.csv"):
            load_raw_data(self.root)

    def test_nested_kaggle_input_and_ambiguous_copies(self):
        with DEFAULT_CONFIG.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["paths"]["raw"] = str(self.root / "not-a-dataset")
        config["paths"]["kaggle_input"] = str(self.root)
        config_path = self.root / "data.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with patch.dict("os.environ", {"M5_DATA_DIR": str(self.root)}):
            self.assertEqual(len(load_raw_data()), 5)
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(len(load_raw_data(config_path=config_path)), 5)
            duplicate = self.root / "another-input"
            duplicate.mkdir()
            for filename in config["files"].values():
                (duplicate / filename).write_bytes((self.root / filename).read_bytes())
            with self.assertRaisesRegex(ValueError, "nhieu bo du lieu"):
                load_raw_data(config_path=config_path)


class ConfigurationTest(unittest.TestCase):
    def test_sample_split_and_feature_settings_are_consistent(self):
        configs = {}
        for name in ["data", "sample", "features"]:
            with (DEFAULT_CONFIG.parent / f"{name}.yaml").open(encoding="utf-8") as handle:
                configs[name] = yaml.safe_load(handle)
        data, sample, features = (configs[name] for name in ["data", "sample", "features"])
        horizon = data["forecast_horizon"]
        train, validation, test = (data["splits"][name] for name in ["train", "validation", "test"])
        self.assertEqual(train[1] + 1, validation[0])
        self.assertEqual(validation[1] + 1, test[0])
        self.assertEqual(validation[1] - validation[0] + 1, horizon)
        self.assertEqual(test[1] - test[0] + 1, horizon)
        self.assertEqual(sample["n_stores"] * sample["n_categories"] * sample["series_per_stratum"],
                         sample["n_series"])
        self.assertEqual(features["forecast_horizon"], horizon)
        self.assertGreaterEqual(min(features["lags"]), horizon)
        self.assertGreaterEqual(features["rolling"]["shift"], horizon)
        self.assertFalse(features["price"]["fill_missing"])


if __name__ == "__main__":
    unittest.main()
