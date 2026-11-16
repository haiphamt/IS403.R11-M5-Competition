"""Correctness checks on explicitly synthetic data; NEVER benchmark evidence for M5."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "c02_benchmark.py"
TMP_ROOT = Path(__file__).resolve().parents[2] / "tmp"
TMP_ROOT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("c02_benchmark", MODULE_PATH)
c02 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = c02
spec.loader.exec_module(c02)


def synthetic_panel():
    """10 stores, 3 categories, 2 items/stratum; known boundary/missing cases."""
    rows = []
    for i in range(10):
        for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
            for item in range(2):
                sid = f"{cat}_{item}_{i}_validation"
                row = dict(id=sid, item_id=f"{cat}_{item}", dept_id=f"{cat}_1", cat_id=cat,
                           store_id=f"S{i:02}", state_id=["CA", "TX", "WI"][i % 3])
                row.update({f"d_{d}": (d + item + i) % 7 if d > 10 else 0 for d in range(1, 1914)})
                rows.append(row)
    sales = pd.DataFrame(rows)
    days = np.arange(1, 1914)
    calendar = pd.DataFrame({"d": [f"d_{d}" for d in days], "wm_yr_wk": (days - 1) // 7,
                             "wday": days % 7 + 1, "month": days % 12 + 1, "year": 2011 + days // 365,
                             "snap_CA": days % 2, "snap_TX": days % 2, "snap_WI": days % 2,
                             "event_name_1": None, "event_type_1": None,
                             "event_name_2": None, "event_type_2": None})
    # Weekly prices start late on purpose. Initial missing prices must remain NaN.
    prices = pd.DataFrame([dict(store_id=r["store_id"], item_id=r["item_id"], wm_yr_wk=w, sell_price=2.5)
                           for r in rows for w in range(3, 274)])
    return sales, calendar, prices


class TestC02(unittest.TestCase):
    def test_environment_reports_missing_packages_before_measurement(self):
        def installed_version(name):
            if name in {"catboost", "psutil"}:
                raise c02.PackageNotFoundError(name)
            return "1.0"

        with patch.object(c02, "version", side_effect=installed_version):
            with self.assertRaisesRegex(RuntimeError, "catboost, psutil.*mục 1"):
                c02.environment()

    def test_environment_recognizes_colab(self):
        with patch.object(c02.Path, "is_dir", return_value=False), patch.dict(
                c02.os.environ, {"COLAB_RELEASE_TAG": "test-runtime"}):
            self.assertEqual(c02.environment()["platform"], "Colab")

    @classmethod
    def setUpClass(cls):
        cls.sales, cls.calendar, cls.prices = synthetic_panel()
        cls.selected = c02.select_series(cls.sales[c02.IDS], 10, 42)
        cls.sample = cls.sales[cls.sales.id.isin(cls.selected.id)].copy()
        cls.frame = c02.create_features(cls.sample, cls.calendar, cls.prices)

    def test_seeded_selection_coverage_and_row_order_independence(self):
        shuffled = self.sales[c02.IDS].sample(frac=1, random_state=9)
        pd.testing.assert_frame_equal(self.selected, c02.select_series(shuffled, 10, 42))
        self.assertEqual(self.selected.store_id.nunique(), 10)
        self.assertEqual(self.selected.cat_id.nunique(), 3)
        self.assertFalse(self.selected.id.duplicated().any())

    def test_price_and_early_lags_stay_nan(self):
        early = self.frame[self.frame.day_num == 1]
        self.assertTrue(early.sell_price.isna().all())
        self.assertTrue((early.price_missing == 1).all())
        self.assertTrue(early[c02.HISTORY_FEATURES].isna().all().all())
        self.assertEqual(len(self.frame[self.frame.day_num <= 1885]), 18850)

    def test_validation_perturbation_and_manual_boundaries(self):
        checks = c02.audit_features(self.sample, self.calendar, self.prices, self.frame)
        self.assertEqual(checks["max_target_source_day_for_validation"], 1885)
        self.assertEqual(checks["validation_target_perturbation"], "PASS")

    def test_duplicate_price_keys_fail(self):
        duplicate = pd.concat([self.prices, self.prices.iloc[[0]]], ignore_index=True)
        with self.assertRaises(pd.errors.MergeError):
            c02.create_features(self.sample, self.calendar, duplicate)

    def test_category_mapping_and_unseen_category(self):
        train = self.frame[self.frame.day_num <= 1885]
        val = self.frame[self.frame.day_num > 1885].copy()
        val.loc[val.index[0], "event_name_1"] = "NEW_EVENT"
        a, b, unknown = c02.encode_features(train, val, "XGBoost")
        self.assertEqual(list(a.event_name_1.cat.categories), list(b.event_name_1.cat.categories))
        self.assertTrue(pd.isna(b.event_name_1.iloc[0]))
        self.assertEqual(unknown["event_name_1"], 1)

    def test_rmsse_manual_leading_zeros_and_degenerate(self):
        self.assertEqual(c02.rmsse_scale([0, 0, 1, 3, 2]), (2.5, False))
        for history in [[0, 0, 0], [0, 0, 1], [2, 2, 2]]:
            self.assertEqual(c02.rmsse_scale(history), (1.0, True))
        val = pd.DataFrame({"id": ["a", "a"], "sales": [2., 4.]})
        m, _ = c02.evaluate(val, [1., 5.], {"a": [0, 0, 1, 3, 2]})
        self.assertAlmostEqual(m["RMSSE_mean"], np.sqrt(1 / 2.5))
        self.assertEqual(m["MAE"], 1.)
        self.assertEqual(m["RMSE"], 1.)
        self.assertEqual(m["R2_pooled"], 0.)
        m, _ = c02.evaluate(val, val.sales.values, {"a": [0, 0, 1, 3, 2]})
        self.assertEqual(m["RMSSE_mean"], 0.)
        self.assertEqual(m["R2_pooled"], 1.)

    def test_constant_zero_targets_finite_and_flagged(self):
        val = pd.DataFrame({"id": ["a", "a"], "sales": [0., 0.]})
        m, _ = c02.evaluate(val, [0., 1.], {"a": [0, 0, 0]})
        self.assertTrue(all(np.isfinite(v) for v in m.values()))
        self.assertEqual(m["scale_fallback_count"], 1)
        self.assertEqual(m["R2_pooled"], 0.)

    def test_explicit_bad_path_never_silently_falls_back(self):
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            with self.assertRaises(ValueError):
                c02.find_data_dir(tmp)

    def test_chunk_loader_uses_validation_only(self):
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            path = Path(tmp)
            self.sales.to_csv(path / "sales_train_validation.csv", index=False)
            self.calendar.to_csv(path / "calendar.csv", index=False)
            self.prices.to_csv(path / "sell_prices.csv", index=False)
            sales, _, _, selected, info = c02.load_sample(path, c02.Config(data_dir=tmp))
            self.assertEqual(len(sales), 10)
            self.assertEqual(info["sales_rows_scanned"], 60)
            self.assertNotIn("d_1914", sales.columns)
            pd.testing.assert_frame_equal(selected, self.selected)

    def test_drive_and_kaggle_csvs_produce_identical_inputs_and_features(self):
        sales = self.sales.copy()
        sales["id"] = sales.item_id + "_" + sales.store_id + "_validation"
        calendar = self.calendar.copy()
        calendar["date"] = pd.date_range("2011-01-29", periods=len(calendar)).strftime("%Y-%m-%d")
        with tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
            root = Path(tmp)
            inputs = []
            for source in ["kaggle", "drive"]:
                folder = root / source
                folder.mkdir()
                sales.drop(columns=["id"] if source == "drive" else []).to_csv(
                    folder / "sales_train_validation.csv", index=False)
                calendar.drop(columns=["d"] if source == "drive" else []).to_csv(
                    folder / "calendar.csv", index=False)
                self.prices.to_csv(folder / "sell_prices.csv", index=False)
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                inputs.append(c02.load_sample(folder, c02.Config(data_dir=str(folder))))
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            for a, b in zip(inputs[0][:4], inputs[1][:4]):
                pd.testing.assert_frame_equal(a, b)
            pd.testing.assert_frame_equal(c02.create_features(*inputs[0][:3]),
                                          c02.create_features(*inputs[1][:3]))
            self.assertTrue(inputs[1][4]["input_adaptations"]["calendar_d_derived_from_date"])

    def test_calendar_day_keys_use_dates_not_row_order(self):
        dates = pd.DataFrame({"date": ["2011-01-31", "2011-01-29"]})
        self.assertEqual(c02.with_calendar_days(dates).d.tolist(), ["d_3", "d_1"])
        for invalid in [[None], ["2011-01-28"], ["2011-01-29", "2011-01-29"]]:
            with self.assertRaises(ValueError):
                c02.with_calendar_days(pd.DataFrame({"date": invalid}))

    def test_real_estimators_fit_predict_with_native_categories_and_nan(self):
        # Synthetic integration only. Default estimators, no fabricated M5 timings.
        train = self.frame[self.frame.day_num <= 1885]
        val = self.frame[self.frame.day_num > 1885]
        for name in ["XGBoost", "CatBoost"]:
            with self.subTest(model=name):
                a, b, _ = c02.encode_features(train, val, name)
                model = c02.build_model(name, c02.Config())
                kwargs = {"cat_features": c02.CATEGORICAL} if name == "CatBoost" else {}
                model.fit(a, train.sales.values, **kwargs)
                prediction = np.maximum(model.predict(b), 0)
                self.assertEqual(len(prediction), 280)
                self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
