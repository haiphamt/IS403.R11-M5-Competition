import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5 import global_models, metrics, optional, panel, univariate  # noqa: E402
from test_no_leakage import small_panel  # noqa: E402


class OptionalPartsTests(unittest.TestCase):
    def test_noprice_ablation_drops_price_and_event_features(self):
        p = panel.build_panel(*small_panel())
        X, y, Xf, _ = global_models.build_table(p, 1885)
        _, info, model = global_models.fit_predict("lightgbm_noprice", X, y, Xf, n_jobs=2)
        names = set(model.feature_name_)
        self.assertFalse(names & set(global_models.NOPRICE_EVENT))
        self.assertIn("snap", names)  # only price and event groups are removed
        imp = global_models.importance_or_coefficients("lightgbm_noprice", model)
        self.assertEqual(len(imp), len(names))

    def test_tuned_model_requires_params_and_applies_them(self):
        p = panel.build_panel(*small_panel())
        X, y, Xf, _ = global_models.build_table(p, 1885)
        with self.assertRaises(ValueError):
            global_models.fit_predict("lightgbm_tuned", X, y, Xf, n_jobs=2)
        _, _, model = global_models.fit_predict("lightgbm_tuned", X, y, Xf, n_jobs=2, lgbm_params={"n_estimators": 7, "num_leaves": 5})
        self.assertEqual((model.n_estimators, model.num_leaves), (7, 5))

    def test_xgboost_parameters_record_real_booster_settings(self):
        import json
        p = panel.build_panel(*small_panel())
        X, y, Xf, _ = global_models.build_table(p, 1885)
        _, _, model = global_models.fit_predict("xgboost", X, y, Xf, n_jobs=2)
        params = json.loads(json.dumps(global_models.model_parameters("xgboost", model)))
        self.assertEqual(params["n_boosted_rounds"], 100)  # library default, not None
        learner = params["booster_config"]["learner"]["gradient_booster"]["tree_train_param"]
        self.assertAlmostEqual(float(learner["eta"]), 0.3, places=6)
        self.assertEqual(int(learner["max_depth"]), 6)

    def test_macro_rmsse_matches_metrics_module(self):
        rng = np.random.default_rng(0)
        y = rng.poisson(2, (5, 120)).astype(float)
        y[0, :10] = 0  # leading zeros are dropped in the scale
        y[4, :] = 0    # all-zero history: undefined, excluded
        truth, pred = rng.poisson(2, (5, 28)).astype(float), rng.poisson(2, (5, 28)).astype(float)
        scales = optional.rmsse_scales(y, 120)
        self.assertTrue(np.isnan(scales[4]))
        expected = np.mean([metrics.rmsse(truth[i], pred[i], y[i]) for i in range(4)])
        self.assertAlmostEqual(optional.macro_rmsse(pred, truth, scales), expected)

    def test_ridge_meta_recovers_known_combination(self):
        rng = np.random.default_rng(1)
        n = 400
        base = pd.DataFrame(rng.random((n, 3)), columns=["a", "b", "c"])
        base["y_true"] = 0.5 * base["a"] + 0.3 * base["b"] + 0.2 * base["c"] + 0.001 * rng.normal(size=n)
        base["origin"] = np.repeat([1, 2], n // 2)
        meta, alpha, cv = optional.fit_ridge_meta(base, ["a", "b", "c"], alphas=(0.001, 1.0))
        np.testing.assert_allclose(meta.coef_, [0.5, 0.3, 0.2], atol=0.02)
        self.assertEqual(alpha, 0.001)

    def test_croston_sba_forecasts_28_finite_values(self):
        y = np.random.default_rng(2).poisson(0.4, 300).astype(float)
        fc, _ = univariate.forecast_one("croston_sba", y)
        self.assertEqual(fc.shape, (28,))
        self.assertTrue(np.isfinite(fc).all())
        classic, _ = univariate.forecast_one("croston", y)
        self.assertLess(fc[0], classic[0])  # SBA shrinks the classic estimate by (1 - alpha/2)


if __name__ == "__main__":
    unittest.main()
