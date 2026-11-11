from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5.lightgbm_forecast import START, _feature_matrix, lightgbm_forecast


class LightGBMForecastTest(unittest.TestCase):
    def test_constant_history_forecasts_the_constant(self):
        np.testing.assert_array_equal(lightgbm_forecast([3.0] * 120), [3.0] * 28)

    def test_all_zero_history_forecasts_zero(self):
        np.testing.assert_array_equal(lightgbm_forecast([0] * 120), [0] * 28)

    def test_repeats_a_clean_weekly_pattern(self):
        week = np.arange(1.0, 8.0)
        # 60 weeks: LightGBM's default min_child_samples=20 needs enough rows per weekday value
        np.testing.assert_allclose(lightgbm_forecast(np.tile(week, 60), horizon=14),
                                   np.tile(week, 2)[:14], atol=0.1)

    def test_forecast_rows_never_read_future_values(self):
        rng = np.random.default_rng(42)
        full = rng.poisson(3, 120).astype(float)
        n = 100
        days = np.arange(n, n + 28)
        np.testing.assert_array_equal(_feature_matrix(full[:n], days),
                                      _feature_matrix(full, days))

    def test_invalid_input_is_rejected(self):
        for history in [[], [1] * START, [[1] * 80], [1] * 79 + [np.nan], [1] * 79 + [np.inf]]:
            with self.subTest(history=history), self.assertRaises(ValueError):
                lightgbm_forecast(history)
        for horizon in [0, 29, 1.5]:
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                lightgbm_forecast([1] * 120, horizon=horizon)


if __name__ == "__main__":
    unittest.main()
