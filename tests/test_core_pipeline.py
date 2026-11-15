import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5 import evaluate, features, sample, schema, split


def fake_sales(n_per=50, T=1941, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for store in ["CA_1", "TX_1"]:
        for cat in ["FOODS", "HOBBIES"]:
            for i in range(n_per):
                item = f"{cat}_{i}"
                rows.append(dict(id=f"{item}_{store}_evaluation", item_id=item, dept_id=cat + "_1",
                                 cat_id=cat, store_id=store, state_id=store[:2],
                                 **{f"d_{d}": int(rng.poisson(2 + (d % 7 == 0))) for d in range(1, T + 1)}))
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def test_splits_cover_and_do_not_overlap(self):
        wide = split.to_wide(fake_sales(3))
        parts = split.split_wide(wide)
        self.assertEqual([parts[k].shape[1] for k in ("train", "validation", "test")], [1885, 28, 28])
        self.assertEqual(parts["validation"].columns[0], "d_1886")
        self.assertEqual(parts["test"].columns[-1], "d_1941")
        with self.assertRaises(ValueError):
            split.assert_no_overlap({"a": wide[["d_1", "d_2"]], "b": wide[["d_2", "d_3"]]})

    def test_stable_series_id_across_suffix(self):
        self.assertEqual(schema.series_id_from_id("FOODS_1_001_CA_1_validation"),
                         schema.series_id_from_id("FOODS_1_001_CA_1_evaluation"))


class SampleTests(unittest.TestCase):
    def test_deterministic_and_stratified(self):
        s = fake_sales(50)
        a = sample.select_stratified(s, per_stratum=10, seed=42)
        b = sample.select_stratified(s.sample(frac=1, random_state=1), per_stratum=10, seed=42)
        self.assertTrue(a.equals(b))
        self.assertEqual(len(a), 40)
        self.assertTrue((a.groupby(["store_id", "cat_id"]).size() == 10).all())
        self.assertEqual(sample.manifest_hash(a), sample.manifest_hash(b))

    def test_small_stratum_fails(self):
        with self.assertRaises(ValueError):
            sample.select_stratified(fake_sales(5), per_stratum=10)


class FeatureTests(unittest.TestCase):
    def test_no_future_leakage(self):
        rng = np.random.default_rng(1)
        y = rng.poisson(3, size=(4, 400)).astype(float)
        origin = 300
        _, f1 = features.build_origin_features(y, origin)
        y2 = y.copy(); y2[:, origin:] = 9999  # change the future
        _, f2 = features.build_origin_features(y2, origin)
        for k in f1:
            np.testing.assert_array_equal(f1[k], f2[k])

    def test_lag_values_and_group_boundary(self):
        y = np.vstack([np.arange(100.0), 1000 + np.arange(100.0)])
        feats = features.lag_roll_features(y)
        self.assertEqual(feats["lag_28"][0, 50], 22.0)
        self.assertEqual(feats["lag_28"][1, 50], 1022.0)  # no bleed between series
        self.assertTrue(np.isnan(feats["lag_28"][:, :28]).all())
        # rolling mean 7 at t uses days t-34..t-28
        self.assertAlmostEqual(feats["rolling_mean_7"][0, 60], np.mean(np.arange(26, 33)))

    def test_lag_below_28_rejected(self):
        with self.assertRaises(ValueError):
            features.lag_roll_features(np.zeros((1, 50)), lags=(7,))


def make_pred(actuals, split_name, model, noise, run="r1"):
    days = schema.forecast_days(split_name)
    rows = []
    for sid in actuals.index:
        for h, d in enumerate(days, 1):
            rows.append((run, model, split_name, sid, d, h, actuals.loc[sid, d] + noise))
    return pd.DataFrame(rows, columns=schema.PREDICTION_COLUMNS)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        wide = split.to_wide(fake_sales(3))
        self.actuals = wide.iloc[:6]

    def test_perfect_forecast_and_shuffled_rows(self):
        pred = make_pred(self.actuals, "validation", "perfect", 0.0)
        pred = pred.sample(frac=1, random_state=3).reset_index(drop=True)  # order must not matter
        per, summ = evaluate.evaluate(pred, self.actuals, "validation")
        self.assertTrue(np.allclose(per[["mae", "rmse", "rmsse"]], 0))
        self.assertTrue(np.allclose(per["r2"], 1))
        self.assertEqual(summ.loc[0, "pooled_mae"], 0)

    def test_undefined_not_patched(self):
        a = self.actuals.copy()
        a.iloc[0, :] = 0  # all-zero series: RMSSE and R2 undefined
        pred = make_pred(a, "test", "m", 0.5)
        per, summ = evaluate.evaluate(pred, a, "test")
        self.assertTrue(np.isnan(per.loc[per.series_id == a.index[0], "rmsse"]).all())
        self.assertEqual(summ.loc[0, "n_rmsse_undefined"], 1)
        self.assertEqual(summ.loc[0, "n_r2_undefined"], 1)

    def test_validation_rejects_bad_predictions(self):
        pred = make_pred(self.actuals, "validation", "m", 0.0)
        bad = pred.copy(); bad.loc[0, "y_pred"] = np.nan
        with self.assertRaises(ValueError): evaluate.evaluate(bad, self.actuals, "validation")
        with self.assertRaises(ValueError): evaluate.evaluate(pd.concat([pred, pred.iloc[:1]]), self.actuals, "validation")
        with self.assertRaises(ValueError): evaluate.evaluate(pred.iloc[:-1], self.actuals, "validation")
        with self.assertRaises(ValueError): evaluate.evaluate(pred, self.actuals, "test")


if __name__ == "__main__":
    unittest.main()
