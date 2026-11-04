from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5.metrics import mae, r2, rmse, rmsse


class MetricsTest(unittest.TestCase):
    def test_perfect_forecast_of_zeros(self):
        zeros = [0] * 28
        train = [0, 0, 1, 0, 2, 0, 1]
        self.assertEqual(mae(zeros, zeros), 0)
        self.assertEqual(rmse(zeros, zeros), 0)
        self.assertEqual(rmsse(zeros, zeros, train), 0)

    def test_hand_computed_example(self):
        y, p = [1, 2, 3, 4], [2, 2, 5, 4]  # errors -1, 0, -2, 0
        self.assertAlmostEqual(mae(y, p), 0.75)
        self.assertAlmostEqual(rmse(y, p), np.sqrt(1.25))
        self.assertAlmostEqual(r2(y, p), 0.0)  # SSE = SST = 5
        self.assertAlmostEqual(r2(y, [1, 2, 3, 5]), 0.8)
        self.assertAlmostEqual(r2(y, y), 1.0)

    def test_rmsse_scale_and_leading_zeros(self):
        # train diffs 2, -1, 3 -> scale 14/3; mse = (1 + 4) / 2
        expected = np.sqrt(2.5 / (14 / 3))
        self.assertAlmostEqual(rmsse([2, 4], [1, 2], [1, 3, 2, 5]), expected)
        # leading zeros are ignored, so the scale must not change
        self.assertAlmostEqual(rmsse([2, 4], [1, 2], [0, 0, 0, 0, 1, 3, 2, 5]), expected)

    def test_undefined_cases_raise_instead_of_returning_nan_or_inf(self):
        for train in [[0] * 10, [0, 0, 5], [3, 3, 3, 3]]:
            with self.subTest(train=train), self.assertRaises(ValueError):
                rmsse([1, 2], [1, 2], train)
        with self.assertRaises(ValueError):
            r2([0] * 5, [0] * 5)

    def test_invalid_input_is_rejected(self):
        bad = [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf])]
        for y, p in bad:
            for fn in (mae, rmse, r2):
                with self.subTest(fn=fn.__name__, y=y, p=p), self.assertRaises(ValueError):
                    fn(y, p)
            with self.subTest(fn="rmsse", y=y, p=p), self.assertRaises(ValueError):
                rmsse(y, p, [1, 2, 3])
        with self.assertRaises(ValueError):
            rmsse([1], [1], [1, np.nan, 3])

    def test_results_are_finite_floats(self):
        y, p, train = [0, 3, 0, 1], [0.5, 2, 0, 1], [0, 1, 4, 0, 2, 3]
        for value in (mae(y, p), rmse(y, p), rmsse(y, p, train), r2(y, p)):
            self.assertIsInstance(value, float)
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
