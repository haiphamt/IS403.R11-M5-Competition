import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_results", ROOT / "scripts" / "analyze_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAnalyzeHelpers(unittest.TestCase):
    def setUp(self):
        self.m = load_module()

    def test_order_keeps_core_order_then_sorts_unknown_models(self):
        got = self.m.order(["xgboost", "stacking", "ets", "croston_sba", "seasonal_naive"])
        self.assertEqual(got, ["seasonal_naive", "ets", "xgboost", "croston_sba", "stacking"])

    def test_win_and_tie_rates_on_toy_data(self):
        mae = pd.DataFrame({"seasonal_naive": [1.0, 1.0, 1.0, 1.0], "ets": [0.5, 1.0, 2.0, 0.9], "new_model": [0.1, 0.2, 0.3, 0.4]})
        w = self.m.win_rates(mae, ["seasonal_naive", "ets", "new_model"])
        self.assertEqual(list(w.index), ["ets", "new_model"])
        self.assertEqual(w.loc["ets", "win_rate_vs_snaive"], 0.5)   # 0.5 and 0.9 beat 1.0
        self.assertEqual(w.loc["ets", "tie_rate"], 0.25)           # exactly one tie
        self.assertAlmostEqual(w.loc["ets", "median_mae_diff"], -0.05)   # diffs -0.5, 0, 1.0, -0.1 -> median -0.05
        self.assertEqual(w.loc["new_model", "win_rate_vs_snaive"], 1.0)

    def test_unknown_model_has_a_label_fallback(self):
        self.assertEqual(self.m.LABEL.get("brand_new", "brand_new"), "brand_new")


if __name__ == "__main__":
    unittest.main()
