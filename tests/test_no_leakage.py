import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5 import global_models, panel  # noqa: E402
from test_core_pipeline import fake_sales  # noqa: E402


def fake_calendar_prices(sales, n_days=1969):
    d = [f"d_{i}" for i in range(1, n_days + 1)]
    cal = pd.DataFrame({"d": d, "date": pd.date_range("2011-01-29", periods=n_days),
                        "wm_yr_wk": [11101 + i // 7 for i in range(n_days)],
                        "wday": [(i % 7) + 1 for i in range(n_days)], "month": [(i // 30) % 12 + 1 for i in range(n_days)],
                        "year": [2011 + i // 365 for i in range(n_days)],
                        "event_name_1": [None] * n_days, "event_type_1": [None] * n_days,
                        "event_name_2": [None] * n_days, "event_type_2": [None] * n_days,
                        "snap_CA": [i % 2 for i in range(n_days)], "snap_TX": [0] * n_days, "snap_WI": [1] * n_days})
    cal.loc[10, ["event_name_1", "event_type_1"]] = ["Super", "Sporting"]
    rows = [(s, i, w, 1.0 + (hash(i) % 5) / 10) for s, i in zip(sales["store_id"], sales["item_id"])
            for w in sorted(set(cal["wm_yr_wk"])) if w > 11110]  # no price in first weeks
    prices = pd.DataFrame(rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).drop_duplicates(
        ["store_id", "item_id", "wm_yr_wk"])
    return cal, prices


def small_panel(seed=0):
    sales = fake_sales(3, seed=seed)
    cal, prices = fake_calendar_prices(sales)
    ids = [f"{i}_{s}" for i, s in zip(sales["item_id"], sales["store_id"])][:8]
    return sales, cal, prices, ids


class PanelTests(unittest.TestCase):
    def test_panel_shapes_and_price_missing(self):
        sales, cal, prices, ids = small_panel()
        p = panel.build_panel(sales, cal, prices, ids)
        self.assertEqual(p.y.shape, (8, 1941))
        self.assertTrue(np.isnan(p.price[:, :70]).all())  # no price before first listed week, no back-fill
        self.assertFalse(np.isnan(p.price[:, 1900]).any())
        self.assertEqual(p.snap.shape, p.y.shape)

    def test_duplicate_prices_rejected(self):
        sales, cal, prices, ids = small_panel()
        with self.assertRaises(ValueError):
            panel.build_panel(sales, cal, pd.concat([prices, prices.iloc[:1]]), ids)


class GlobalFeatureLeakageTests(unittest.TestCase):
    def test_future_labels_do_not_change_features_or_targets(self):
        sales, cal, prices, ids = small_panel()
        for origin in (1885, 1913):
            a = panel.build_panel(sales, cal, prices, ids)
            b = panel.build_panel(sales, cal, prices, ids)
            b.y[:, origin:] = 999.0  # corrupt every label after the origin
            ta, tb = global_models.build_table(a, origin), global_models.build_table(b, origin)
            pd.testing.assert_frame_equal(ta[0], tb[0])
            pd.testing.assert_frame_equal(ta[2], tb[2])
            np.testing.assert_array_equal(ta[1], tb[1])

    def test_table_layout_and_group_boundaries(self):
        sales, cal, prices, ids = small_panel()
        p = panel.build_panel(sales, cal, prices, ids)
        X, y, Xf, keys = global_models.build_table(p, 1885)
        self.assertEqual(len(Xf), 8 * 28)
        self.assertEqual(keys["d_num"].min(), 1886)
        self.assertEqual(keys["d_num"].max(), 1913)
        self.assertEqual(len(X), 8 * (1885 - 55))
        # lag_28 of the first forecast day of series 0 is the sales at d_1858, not another series' day
        first = Xf.iloc[0]
        self.assertEqual(first["lag_28"], p.y[0, 1857])
        self.assertEqual(Xf.iloc[28]["lag_28"], p.y[1, 1857])


if __name__ == "__main__":
    unittest.main()
