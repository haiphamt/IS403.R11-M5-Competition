import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wrmsse", ROOT / "scripts" / "wrmsse.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class TestWrmsse(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.meta = pd.DataFrame({"item_id": ["A_1", "A_2", "B_1", "A_1"], "dept_id": ["A", "A", "B", "A"], "cat_id": ["A", "A", "B", "A"],
                                  "store_id": ["CA_1", "CA_1", "CA_1", "TX_1"], "state_id": ["CA", "CA", "CA", "TX"]})
        self.y = rng.poisson(2.0, (4, 100)).astype(float)
        self.prices = np.full((4, 100), 1.5)

    def test_levels_and_perfect_forecast(self):
        agg, lv = W.aggregation(self.meta)
        self.assertEqual(agg.shape[1], 4)
        self.assertEqual(len(np.unique(lv)), 12)
        self.assertTrue(np.allclose(agg[lv == 0].toarray(), 1))                   # level 1 = total of every bottom series
        self.assertEqual(int((lv == 11).sum()), 4)                                 # level 12 = the bottom series themselves
        total, per_level = W.wrmsse(self.y, self.prices, self.y[:, 72:100], agg, lv, origin=72)
        self.assertEqual(total, 0.0)

    def test_worse_forecast_has_larger_score(self):
        agg, lv = W.aggregation(self.meta)
        good, _ = W.wrmsse(self.y, self.prices, self.y[:, 72:100] + 0.1, agg, lv, origin=72)
        bad, _ = W.wrmsse(self.y, self.prices, np.zeros((4, 28)), agg, lv, origin=72)
        self.assertLess(good, bad)


if __name__ == "__main__":
    unittest.main()
