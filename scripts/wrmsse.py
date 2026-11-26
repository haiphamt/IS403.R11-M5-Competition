"""WRMSSE of the M5 Accuracy competition (12 aggregation levels, 42,840 series) for full-data runs (all 30,490 series).

    python scripts/wrmsse.py --split test --runs results/full/runs/tes-A results/full/runs/tes-B-stack --out results/full/tables/test

Definition (M5 Competitors' Guide): bottom-level forecasts are summed to the 12 levels; each of the 42,840 series gets an RMSSE whose
denominator is the one-step naive MSE of its own history from its first non-zero sale up to the origin; weights are the dollar sales
(units x sell_price) of the last 28 days before the origin, normalised within each level; WRMSSE = mean over the 12 levels of the
weighted RMSSE. Our Test window d_1914..d_1941 is the period of the M5 public leaderboard (the private one is d_1942..d_1969).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m5.data import load_raw_data  # noqa: E402
from m5.schema import SPLITS, series_id_from_id  # noqa: E402

LEVELS = [("total",), ("state_id",), ("store_id",), ("cat_id",), ("dept_id",), ("state_id", "cat_id"), ("state_id", "dept_id"),
          ("store_id", "cat_id"), ("store_id", "dept_id"), ("item_id",), ("item_id", "state_id"), ("item_id", "store_id")]


def aggregation(meta):
    """Sparse (42,840 x n_bottom) 0/1 matrix and the level index of each aggregated row."""
    blocks, level_of_row = [], []
    m = meta.assign(total="all")
    for lv, cols in enumerate(LEVELS):
        key = m[list(cols)].astype(str).agg("|".join, axis=1)
        codes, uniq = pd.factorize(key, sort=True)
        blocks.append(sparse.csr_matrix((np.ones(len(m)), (codes, np.arange(len(m)))), shape=(len(uniq), len(m))))
        level_of_row += [lv] * len(uniq)
    return sparse.vstack(blocks).tocsr(), np.array(level_of_row)


def scales(hist):
    """One-step naive MSE of each row from its first non-zero value; NaN when undefined."""
    out = np.full(len(hist), np.nan)
    for i, row in enumerate(hist):
        nz = np.flatnonzero(row)
        if nz.size and len(row) - nz[0] > 1:
            s = np.mean(np.diff(row[nz[0]:]) ** 2)
            out[i] = s if s > 0 else np.nan
    return out


def wrmsse(y, prices, pred, agg, level_of_row, origin, horizon=28):
    """y, prices: (n_bottom, >= origin+horizon) daily units and prices; pred: (n_bottom, horizon). Returns (total, per-level array)."""
    Y = agg @ y[:, :origin + horizon]
    P = agg @ pred
    sc = scales(Y[:, :origin])
    dollars = agg @ np.nansum(y[:, origin - 28:origin] * prices[:, origin - 28:origin], axis=1)
    rmsse = np.sqrt(np.mean((Y[:, origin:origin + horizon] - P) ** 2, axis=1) / sc)
    per_level = np.zeros(len(LEVELS))
    for lv in range(len(LEVELS)):
        rows = level_of_row == lv
        w = dollars[rows] / dollars[rows].sum()
        ok = ~np.isnan(rmsse[rows])  # undefined scale: series without any sale before the origin carries no weight anyway
        per_level[lv] = np.sum(w[ok] * rmsse[rows][ok])
    return per_level.mean(), per_level


def daily_prices(sales, calendar, sell_prices):
    days = [f"d_{n}" for n in range(1, 1942)]
    wk = calendar.set_index("d").loc[days, "wm_yr_wk"].to_numpy()
    weekly = sell_prices.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    idx = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
    return weekly.reindex(index=idx, columns=wk).to_numpy(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    raw = load_raw_data(a.data_dir, table_names=("sales_train_evaluation", "calendar", "sell_prices"))
    sales = raw["sales_train_evaluation"]
    sales.index = sales["id"].map(series_id_from_id)
    if len(sales) != 30490:
        raise SystemExit("WRMSSE needs all 30,490 bottom-level series")
    y = sales[[f"d_{n}" for n in range(1, 1942)]].to_numpy(float)
    prices = daily_prices(sales, raw["calendar"], raw["sell_prices"])
    agg, level_of_row = aggregation(sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]])
    origin = SPLITS[a.split][0] - 1
    days = [f"d_{n}" for n in range(origin + 1, origin + 29)]
    pred = pd.concat([pd.read_parquet(r / "predictions.parquet") for r in a.runs], ignore_index=True)
    rows = []
    for model, g in pred.groupby("model", sort=False):
        wide = g.pivot(index="series_id", columns="d", values="y_pred").reindex(index=sales.index, columns=days)
        if wide.isna().any().any():
            raise SystemExit(f"{model}: forecasts missing for some of the 30,490 series/days")
        total, per_level = wrmsse(y, prices, wide.to_numpy(float), agg, level_of_row, origin)
        rows.append({"model": model, "wrmsse": total, **{f"level_{i + 1}": v for i, v in enumerate(per_level)}})
        print(model, round(total, 4), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(a.out / f"{a.split}_wrmsse.csv", index=False)


if __name__ == "__main__":
    main()
