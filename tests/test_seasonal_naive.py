from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5.seasonal_naive import seasonal_naive


class SeasonalNaiveTest(unittest.TestCase):
    def test_same_weekday_is_repeated_for_four_weeks(self):
        np.testing.assert_array_equal(seasonal_naive(range(1, 11)), [4, 5, 6, 7, 8, 9, 10] * 4)
        np.testing.assert_array_equal(seasonal_naive(range(1, 11), horizon=10),
                                      [4, 5, 6, 7, 8, 9, 10, 4, 5, 6])
        np.testing.assert_array_equal(seasonal_naive([0] * 7), [0] * 28)

    def test_incomplete_or_nonfinite_history_is_rejected(self):
        for history in [[], [1] * 6, [[1] * 7], [1] * 6 + [np.nan], [1] * 6 + [np.inf]]:
            with self.subTest(history=history), self.assertRaises(ValueError):
                seasonal_naive(history)
        for kwargs in [{"horizon": 0}, {"season_length": 0}, {"horizon": 1.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                seasonal_naive([1] * 7, **kwargs)


if __name__ == "__main__":
    unittest.main()
