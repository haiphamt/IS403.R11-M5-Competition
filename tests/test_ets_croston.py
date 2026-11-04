from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5.ets_croston import croston_forecast, ets_forecast


class EtsCrostonTest(unittest.TestCase):
    def test_ets_repeats_a_clean_weekly_pattern(self):
        week = np.arange(1.0, 8.0)
        np.testing.assert_allclose(ets_forecast(np.tile(week, 20), horizon=14),
                                   np.tile(week, 2), atol=1e-3)

    def test_croston_is_a_flat_line_inside_the_demand_range(self):
        rng = np.random.default_rng(42)
        history = rng.poisson(0.5, 200).astype(float)
        history[-1] = 1  # guarantee at least one recent demand
        forecast = croston_forecast(history)
        self.assertEqual(forecast.shape, (28,))
        self.assertEqual(len(set(forecast)), 1)
        self.assertTrue(0 < forecast[0] <= history.max())

    def test_all_zero_history_forecasts_zero(self):
        for fn in (ets_forecast, croston_forecast):
            with self.subTest(fn=fn.__name__):
                np.testing.assert_array_equal(fn([0] * 60), [0] * 28)

    def test_invalid_input_is_rejected(self):
        for fn in (ets_forecast, croston_forecast):
            for history in [[], [1] * 13, [[1] * 20], [1] * 19 + [np.nan], [1] * 19 + [np.inf]]:
                with self.subTest(fn=fn.__name__, history=history), self.assertRaises(ValueError):
                    fn(history)
            for kwargs in [{"horizon": 0}, {"horizon": 1.5}, {"season_length": 0}]:
                with self.subTest(fn=fn.__name__, kwargs=kwargs), self.assertRaises(ValueError):
                    fn([1] * 20, **kwargs)


if __name__ == "__main__":
    unittest.main()
