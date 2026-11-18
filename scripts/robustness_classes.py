"""Robustness of the demand-class tables: classes recomputed with zeros counted only AFTER the first sale.

    python scripts/robustness_classes.py --split test --runs results/runs/tes-A results/runs/tes-B --out results/tables/test

The frozen classes (configs, tables, report) use the zero share over all of d_1..d_1885. About 944 of the 1,200 series have
a long run of zero sales before their first sale (the item was not yet listed; all price-missing days are zero sales), which
inflates the 'intermittent' share. This script keeps the frozen classes untouched and reports how the model ranking looks
under the alternative definition, plus how many series change class.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m5 import optional  # noqa: E402
from m5.schema import SPLITS  # noqa: E402

ORDER = ["seasonal_naive", "linear_regression", "ets", "croston", "lightgbm", "xgboost", "catboost"]
NAMES = ("low_zero(<=0.5)", "mid_zero(0.5-0.8]", "high_zero(>0.8)")


def classify(share):
    return np.where(share <= 0.5, NAMES[0], np.where(share <= 0.8, NAMES[1], NAMES[2]))


def zero_shares(history):
    """(share over all days, share after the first non-zero sale) per series; all-zero series count as 1.0."""
    first = np.array([np.flatnonzero(r)[0] if r.any() else len(r) for r in history])
    after = np.array([(r[f:] == 0).mean() if f < len(r) else 1.0 for r, f in zip(history, first)])
    return (history == 0).mean(axis=1), after, first


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    panel = optional.load_sample_panel(a.data_dir)
    all_days, after, first = zero_shares(panel.y[:, :SPLITS["train"][1]])
    old, new = classify(all_days), classify(after)
    alt = pd.Series(new, index=panel.series_ids, name="alt_class").rename_axis("series_id").reset_index()
    ps = pd.concat([pd.read_csv(r / "metrics_per_series.csv") for r in a.runs]).merge(alt, on="series_id")
    table = ps.groupby(["alt_class", "model"]).rmsse.mean().unstack()[ORDER]
    table.insert(0, "n_series", ps.groupby("alt_class").series_id.nunique())
    a.out.mkdir(parents=True, exist_ok=True)
    table.round(6).reset_index().rename(columns={"alt_class": "demand_class_after_first_sale"}).to_csv(
        a.out / f"{a.split}_by_demand_class_after_first_sale.csv", index=False)
    summary = pd.DataFrame({
        "n_series": [len(panel.series_ids)], "n_with_leading_zeros": [int((first > 0).sum())],
        "median_leading_days": [float(np.median(first[first > 0]))],
        "mean_zero_share_all_days": [float(all_days.mean())], "mean_zero_share_after_first_sale": [float(after.mean())],
        "n_change_class": [int((old != new).sum())], "n_high_to_lower": [int(((old == NAMES[2]) & (new != NAMES[2])).sum())],
        **{f"n_{n.split('(')[0]}_all_days": [int((old == n).sum())] for n in NAMES},
        **{f"n_{n.split('(')[0]}_after_first_sale": [int((new == n).sum())] for n in NAMES}})
    summary.round(6).to_csv(a.out.parent / "d02_demand_class_after_first_sale.csv", index=False)
    print(table.round(3).to_string()); print(summary.T.to_string(header=False))


if __name__ == "__main__":
    main()
