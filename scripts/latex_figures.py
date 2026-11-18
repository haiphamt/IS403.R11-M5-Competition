"""Figures of the LaTeX report (Overleaf VI and EN) from the full-data results.

    python scripts/latex_figures.py      # writes report/latex_figures/{vi,en}_*.pdf

Example series for the forecast figures follow a public rule: in the <=50% and >80% zero-share groups (Train d_1..d_1885), the
series whose Train mean sales is the median of its group.
"""
import json
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
from m5.split import to_wide  # noqa: E402

FULL = ROOT / "results" / "full"
OUT = ROOT / "report" / "latex_figures"
ORDER = ["seasonal_naive", "ets", "croston_sba", "lightgbm", "stacking"]
NAME = {"seasonal_naive": "Seasonal Naive", "ets": "AutoETS", "croston_sba": "Croston SBA", "lightgbm": "LightGBM", "stacking": "Stacking"}
TXT = {"vi": {"daily_y": "Tổng số đơn vị bán", "daily_x": "Ngày (d_1–d_1885)", "ma": "Trung bình trượt 7 ngày", "day": "Theo ngày",
              "score_y": "Macro-RMSSE", "val": "Validation", "test": "Test", "n": "N = {} chuỗi",
              "rt_x": "Thời gian thực trên Test (giây, thang log)", "rt_y": "Macro-RMSSE Test", "rt_note": "Stacking: gồm ba mô hình cơ sở, chưa gồm hai backtest",
              "actual": "Thực tế", "fc": "Dự báo", "origin": "Mốc d_1913", "low": "Ít số 0: {} ({:.0%} ngày 0)", "high": "Nhiều số 0: {} ({:.0%} ngày 0)",
              "units": "Đơn vị bán"},
       "en": {"daily_y": "Total units sold", "daily_x": "Day (d_1–d_1885)", "ma": "7-day moving average", "day": "Daily",
              "score_y": "Macro-RMSSE", "val": "Validation", "test": "Test", "n": "N = {} series",
              "rt_x": "Wall-clock time on Test (s, log scale)", "rt_y": "Test macro-RMSSE", "rt_note": "Stacking: includes the three base models, not the two backtests",
              "actual": "Actual", "fc": "Forecast", "origin": "Origin d_1913", "low": "Few zeros: {} ({:.0%} zero days)", "high": "Many zeros: {} ({:.0%} zero days)",
              "units": "Units sold"}}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = to_wide(load_raw_data(table_names=("sales_train_evaluation",))["sales_train_evaluation"])
    cls = pd.read_csv(ROOT / "data" / "full_series_ids_with_class.csv").set_index("series_id")
    over = {sp: pd.read_csv(FULL / "tables" / sp / f"{sp}_overall.csv", index_col=0) for sp in ("validation", "test")}
    tim = json.loads((FULL / "tables" / "test" / "timings.json").read_text())
    pred = pd.read_parquet(FULL / "tables" / "test" / "predictions.parquet") if (FULL / "tables" / "test" / "predictions.parquet").exists() else \
        pd.concat([pd.read_parquet(p / "predictions.parquet") for p in sorted((FULL / "runs").glob("tes-*"))])
    train_mean = y.iloc[:, :1885].mean(axis=1)
    picks = {}
    for key, grp in (("low", "low_zero(<=0.5)"), ("high", "high_zero(>0.8)")):
        ids = cls.index[cls["demand_class"] == grp]
        s = train_mean.loc[ids].sort_values()
        picks[key] = s.index[len(s) // 2]
    wall = {"seasonal_naive": tim["seasonal_naive"]["wall_s"], "ets": tim["ets"]["wall_s"], "croston_sba": tim["croston_sba"]["wall_s"],
            "lightgbm": tim["lightgbm"]["fit_s"] + tim["lightgbm"]["predict_s"]}
    wall["stacking"] = tim["stacking"]["fit_s"] + wall["ets"] + wall["croston_sba"] + wall["lightgbm"]
    total = y.iloc[:, :1885].sum(axis=0).to_numpy()
    for lang, t in TXT.items():
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        ax.plot(np.arange(1, 1886), total, lw=0.4, color="0.6", label=t["day"])
        ax.plot(np.arange(1, 1886), pd.Series(total).rolling(7).mean(), lw=1.2, color="C0", label=t["ma"])
        ax.set_xlabel(t["daily_x"]); ax.set_ylabel(t["daily_y"]); ax.legend(fontsize=7, loc="upper left")
        save(fig, f"{lang}_eda_daily_sales")

        fig, ax = plt.subplots(figsize=(6.4, 2.8))
        x = np.arange(len(ORDER))
        for k, (sp, off) in enumerate((("validation", -0.2), ("test", 0.2))):
            v = over[sp].loc[ORDER, "macro_rmsse"].to_numpy()
            n = f"{int(over[sp].loc[ORDER[0], 'n_series']):,}".replace(",", "." if lang == "vi" else ",")
            bars = ax.bar(x + off, v, width=0.4, label=f"{t['val' if sp == 'validation' else 'test']} ({t['n'].format(n)})")
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.005, f"{val:.3f}".replace(".", "," if lang == "vi" else "."), ha="center", fontsize=6)
        ax.set_xticks(x, [NAME[m] for m in ORDER], fontsize=7); ax.set_ylabel(t["score_y"]); ax.set_ylim(0.6, 1.05); ax.legend(fontsize=7)
        save(fig, f"{lang}_score_validation_test")

        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        for m in ORDER:
            ax.scatter(wall[m], over["test"].loc[m, "macro_rmsse"])
            ax.annotate(NAME[m], (wall[m], over["test"].loc[m, "macro_rmsse"]), xytext=(0, 7 if m != "ets" else -12), textcoords="offset points",
                        ha="right" if m == "stacking" else "center", fontsize=7)
        ax.set_xscale("log"); ax.set_xlabel(t["rt_x"]); ax.set_ylabel(t["rt_y"]); ax.margins(x=0.15, y=0.15)
        ax.text(0.01, 0.02, t["rt_note"], transform=ax.transAxes, fontsize=6)
        save(fig, f"{lang}_accuracy_runtime_test")

        days = [f"d_{n}" for n in range(1914, 1942)]
        hist = list(range(1858, 1914))
        for i, m in enumerate(ORDER, start=1):
            fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
            for ax, key in zip(axes, ("low", "high")):
                sid = picks[key]
                p = pred[(pred["series_id"] == sid) & (pred["model"] == m)].set_index("d").loc[days, "y_pred"].to_numpy()
                ymax = max(y.loc[sid, [f"d_{n}" for n in hist] + days].max(), pred[pred["series_id"] == sid]["y_pred"].max()) * 1.1 + 0.5
                ax.plot(hist, y.loc[sid, [f"d_{n}" for n in hist]], color="0.5", lw=0.8)
                ax.plot(range(1914, 1942), y.loc[sid, days], color="k", lw=0.9, marker="o", ms=2, label=t["actual"])
                ax.plot(range(1914, 1942), p, color="C3", lw=1.3, label=t["fc"])
                ax.axvline(1913.5, color="C0", ls="--", lw=0.8, label=t["origin"])
                ax.set_ylim(0, ymax); ax.set_title(t[key].format(sid, cls.loc[sid, "train_zero_ratio"]), fontsize=7)
                ax.tick_params(labelsize=6)
            axes[0].set_ylabel(t["units"], fontsize=7); axes[1].legend(fontsize=6, loc="upper left")
            fig.suptitle(NAME[m], fontsize=8)
            save(fig, f"{lang}_forecast_test_{i}")
    print(json.dumps({k: [v, float(cls.loc[v, "train_zero_ratio"]), float(train_mean[v])] for k, v in picks.items()}))
    print({m: round(w, 1) for m, w in wall.items()})


if __name__ == "__main__":
    main()
