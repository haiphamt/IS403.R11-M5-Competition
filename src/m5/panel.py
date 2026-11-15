"""Sample panel: sales matrix + calendar covariates + weekly prices, aligned by day.

The panel holds all d_1..d_1941 labels because the evaluator needs them, but model
code must only read ``y[:, :origin]`` (see ``global_models.build_table``).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import series_id_from_id

N_DAYS = 1941
META = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class Panel:
    series_ids: list
    meta: pd.DataFrame      # index series_id, columns META (str)
    y: np.ndarray           # (n_series, 1941) float, labels
    calendar: pd.DataFrame  # 1941 rows, row i = day d_{i+1}
    price: np.ndarray       # (n_series, 1941) float, NaN before the first listed price
    snap: np.ndarray        # (n_series, 1941) float, state-specific SNAP flag


def build_panel(sales, calendar, prices, series_ids):
    series_ids = list(series_ids)
    sales = sales.copy()
    sales["series_id"] = sales["id"].astype(str).map(series_id_from_id)
    sales = sales.set_index("series_id").loc[series_ids]
    meta = sales[META].astype(str)
    days = [f"d_{n}" for n in range(1, N_DAYS + 1)]
    cal = calendar.set_index("d").loc[days].reset_index()
    prices = prices[prices["item_id"].isin(set(meta["item_id"])) & prices["store_id"].isin(set(meta["store_id"]))]
    prices = prices.astype({"item_id": str, "store_id": str})
    keys = meta[["item_id", "store_id"]].reset_index()
    joined = prices.merge(keys, on=["item_id", "store_id"], validate="many_to_many")
    weekly = joined.pivot(index="series_id", columns="wm_yr_wk", values="sell_price")  # raises on duplicates
    price = weekly.reindex(index=meta.index, columns=cal["wm_yr_wk"]).to_numpy(float)
    snap = np.zeros((len(meta), N_DAYS))
    for state in meta["state_id"].unique():
        mask = (meta["state_id"] == state).to_numpy()
        snap[mask] = cal[f"snap_{state}"].to_numpy(float)
    return Panel(series_ids, meta, sales[days].to_numpy(float), cal, price, snap)
