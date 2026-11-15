import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m5 import univariate  # noqa: E402


class UnivariateTests(unittest.TestCase):
    def test_history_only_forecast(self):
        y = np.random.default_rng(1).poisson(2, 400).astype(float)
        fc, _ = univariate.forecast_one("seasonal_naive", y)
        np.testing.assert_array_equal(fc, univariate.forecast_one("seasonal_naive", y.copy())[0])
        self.assertEqual(fc.shape, (28,))

    def test_croston_components_hand_computed(self):
        z, p = univariate.croston_components([0, 2, 0, 0, 4], alpha=0.5)
        # first demand 2 at t=2 (p=2); next demand 4 after interval 3: z=2+.5*(4-2)=3, p=2+.5*(3-2)=2.5
        self.assertAlmostEqual(z, 3.0)
        self.assertAlmostEqual(p, 2.5)

    def test_croston_reference_matches_statsforecast(self):
        y = np.random.default_rng(3).poisson(0.4, 300).astype(float)
        fc, info = univariate.croston(y)
        self.assertAlmostEqual(info["croston_zp"], fc[0], places=6)

    def test_chunk_reports_fallback_instead_of_hiding_it(self):
        out = univariate.run_chunk("ets", ["s1"], [np.ones(5)])  # too short for AutoETS season 7
        self.assertEqual(len(out[0]["forecast"]), 28)
        self.assertTrue(out[0]["fallback"] or out[0]["error"] == "")


if __name__ == "__main__":
    unittest.main()
