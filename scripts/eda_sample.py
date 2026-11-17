"""D07 EDA on the stratified sample, Train days only (d_1..d_1885): zero ratio, weekly cycle,
category/state, SNAP and event days, price variation. Associations only, no causal claims.

    python scripts/eda_sample.py            # writes results/tables/d07_eda_*.csv and results/figures/eda_*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m5.data import load_raw_data  # noqa: E402
from m5.panel import build_panel  # noqa: E402
from m5.schema import SPLITS  # noqa: E402

TRAIN = SPLITS["train"][1]
T, F = ROOT / "results" / "tables", ROOT / "results" / "figures"


def main():
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data(table_names=("sales_train_evaluation", "calendar", "sell_prices"))
    manifest = pd.read_csv(ROOT / "data" / "sample_series_ids_with_class.csv")
    p = build_panel(raw["sales_train_evaluation"], raw["calendar"], raw["sell_prices"], manifest["series_id"])
    y = p.y[:, :TRAIN]
    cal = p.calendar.iloc[:TRAIN]
    meta = p.meta.join(manifest.set_index("series_id")[["demand_class", "train_zero_ratio"]])

    # all-series zero ratio for comparison (Train only)
    allsales = raw["sales_train_evaluation"][[f"d_{n}" for n in range(1, TRAIN + 1)]].to_numpy()
    zr_all = (allsales == 0).mean(axis=1)
    zr = (y == 0).mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(zr_all, bins=bins, alpha=0.5, density=True, label="30.490 chuỗi")
    ax.hist(zr, bins=bins, alpha=0.5, density=True, label="Mẫu 1.200 chuỗi")
    ax.set_xlabel("Tỷ lệ ngày bán 0 (Train d_1–d_1885)")
    ax.set_ylabel("Mật độ")
    ax.legend()
    ax.set_title("Phân bố tỷ lệ ngày bán bằng 0")
    fig.tight_layout()
    fig.savefig(F / "eda_zero_ratio_hist.png", dpi=140)
    plt.close(fig)

    # weekday cycle (mean sales per series-day, indexed to the weekly mean)
    wd = pd.DataFrame({"weekday": pd.to_datetime(cal["date"]).dt.day_name().to_numpy(), "mean": y.mean(axis=0)})
    wk = wd.groupby("weekday")["mean"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    pd.DataFrame({"mean_sales": wk, "index_vs_week_mean": wk / wk.mean()}).round(4).to_csv(T / "d07_eda_weekday.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    (wk / wk.mean()).plot.bar(ax=ax)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("Chỉ số so với trung bình tuần")
    ax.set_title("Chu kỳ theo thứ trong tuần (Train)")
    fig.tight_layout()
    fig.savefig(F / "eda_weekday.png", dpi=140)
    plt.close(fig)

    # category / state / demand class
    grp = pd.DataFrame({"cat": meta["cat_id"].to_numpy(), "state": meta["state_id"].to_numpy(),
                        "cls": meta["demand_class"].to_numpy(), "mean_sales": y.mean(axis=1), "zero_ratio": zr})
    for key in ("cat", "state", "cls"):
        g = grp.groupby(key)[["mean_sales", "zero_ratio"]].agg(["mean", "count"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]  # flat header so pd.read_csv reads it back directly
        g.round(4).to_csv(T / f"d07_eda_by_{key}.csv")

    # SNAP and event days (descriptive, per state; ratio of the POOLED means over all series and days of the state, not a within-series ratio)
    rows = []
    for state in sorted(meta["state_id"].unique()):
        m = (meta["state_id"] == state).to_numpy()
        flags = p.snap[m, :TRAIN]
        assert (flags == flags[0]).all(), "SNAP is state-wide: every series of a state must share the same flags"
        snap = flags[0].astype(bool)
        ys = y[m]
        rows.append({"state": state, "snap_day_mean": ys[:, snap].mean(), "non_snap_mean": ys[:, ~snap].mean()})
    snap_tab = pd.DataFrame(rows)
    snap_tab["ratio"] = snap_tab["snap_day_mean"] / snap_tab["non_snap_mean"]
    snap_tab.round(4).to_csv(T / "d07_eda_snap.csv", index=False)
    ev = (cal["event_name_1"].notna() | cal["event_name_2"].notna()).to_numpy()  # either event slot
    dow = pd.to_datetime(cal["date"]).dt.dayofweek.to_numpy()
    # weekday-adjusted ratio: event vs non-event mean within each weekday, then averaged (events cluster on some weekdays)
    ratios = [y[:, ev & (dow == w)].mean() / y[:, ~ev & (dow == w)].mean() for w in range(7) if (ev & (dow == w)).any()]
    pd.DataFrame({"event_day_mean": [y[:, ev].mean()], "non_event_mean": [y[:, ~ev].mean()], "n_event_days": [int(ev.sum())],
                  "weekday_adjusted_ratio": [float(np.mean(ratios))]}).round(4).to_csv(T / "d07_eda_events.csv", index=False)

    # price variation: share of series-days with a missing price (before first listing) and relative price range
    price = p.price[:, :TRAIN]
    miss = np.isnan(price).mean()
    rng = (np.nanmax(price, axis=1) - np.nanmin(price, axis=1)) / np.nanmean(price, axis=1)
    pd.DataFrame({"missing_price_share": [miss], "median_relative_price_range": [np.nanmedian(rng)],
                  "series_with_price_change": [float((rng > 0).mean())]}).round(4).to_csv(T / "d07_eda_price.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
