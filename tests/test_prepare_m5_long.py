"""Check the Parquet output across stores, weeks and missing prices."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_m5_long import SALES_IDS, write_long_parquet


class PrepareM5LongTest(unittest.TestCase):
    def test_week_and_store_join_preserves_missing_prices(self):
        sales = pd.DataFrame([
            ["item_CA_1_evaluation", "item", "FOODS_1", "FOODS", "CA_1", "CA", 0, 2],
            ["item_TX_1_evaluation", "item", "FOODS_1", "FOODS", "TX_1", "TX", 3, 4],
        ], columns=SALES_IDS + ["d_1", "d_8"]).astype({key: "category" for key in SALES_IDS})
        calendar = pd.DataFrame({
            "d": ["d_1", "d_8"],
            "date": pd.to_datetime(["2011-01-29", "2011-02-05"]),
            "wm_yr_wk": [11101, 11102],
        })
        prices = pd.DataFrame([
            ["CA_1", "item", 11102, 2.5],
            ["TX_1", "item", 11101, 3.5],
            ["TX_1", "item", 11102, 4.5],
        ], columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])

        with TemporaryDirectory() as temp:
            output = Path(temp) / "m5_long.parquet"
            self.assertEqual(write_long_parquet(sales, calendar, prices, output, batch_series=1), (4, 1))
            long = pd.read_parquet(output).set_index(["store_id", "day"]).sort_index()

        self.assertEqual(long["sales"].tolist(), [0, 2, 3, 4])
        self.assertEqual(long["price_missing"].tolist(), [True, False, False, False])
        self.assertTrue(pd.isna(long.loc[("CA_1", 1), "sell_price"]))
        self.assertEqual(long["sell_price"].iloc[1:].tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(long.loc[("TX_1", 8), "date"], pd.Timestamp("2011-02-05"))


if __name__ == "__main__":
    unittest.main()
