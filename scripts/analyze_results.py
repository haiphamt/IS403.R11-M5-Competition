"""Tables and figures from a collected run (output of scripts/collect_runs.py).

    python scripts/analyze_results.py --split validation --tables results/tables/validation --figures results/figures/validation

Reads predictions.parquet, metrics_per_series.csv, timings.json (+ sample classes and calendar) and writes
group tables (demand class from Train zero ratio, category, state), weekday and horizon error, runtime,
win-rate versus Seasonal Naive and objective actual-vs-predicted examples. Only passes over existing predictions.
"""
import argparse
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
from m5.schema import SPLITS  # noqa: E402
from m5.split import to_wide  # noqa: E402

ORDER = ["seasonal_naive", "linear_regression", "ets", "croston", "lightgbm", "xgboost", "catboost"]
LABEL = {"seasonal_naive": "Seasonal Naive", "linear_regression": "Linear Regression", "ets": "ETS",
         "croston": "Croston", "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}


def order(models):
    return [m for m in ORDER if m in set(models)] + sorted(set(models) - set(ORDER))


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--classes", type=Path, default=ROOT / "data" / "sample_series_ids_with_class.csv")
    a = ap.parse_args()
    a.figures.mkdir(parents=True, exist_ok=True)
    tag = a.split
    pred = pd.read_parquet(a.tables / "predictions.parquet")
    ps = pd.read_csv(a.tables / "metrics_per_series.csv")
    summary = pd.read_csv(a.tables / "metrics_summary.csv")
    timings = json.loads((a.tables / "timings.json").read_text())
    models = order(pred["model"].unique())
    cls = pd.read_csv(a.classes).set_index("series_id")
    ps = ps.join(cls[["store_id", "cat_id", "demand_class"]], on="series_id")
    ps["state"] = ps["store_id"].str.split("_").str[0]

    # 1. overall table with relative-to-baseline columns
    base = summary.set_index("model").loc["seasonal_naive"]
    overall = summary.set_index("model").loc[models].copy()
    overall["mae_vs_snaive_pct"] = 100 * (overall["macro_mae"] / base["macro_mae"] - 1)
    overall["rmsse_vs_snaive_pct"] = 100 * (overall["macro_rmsse"] / base["macro_rmsse"] - 1)
    overall.round(4).to_csv(a.tables / f"{tag}_overall.csv")

    # 2. group tables (macro mean of per-series metrics; undefined RMSSE/R2 are NaN and skipped)
    for key in ("demand_class", "cat_id", "state"):
        g = ps.groupby([key, "model"])[["mae", "rmse", "rmsse", "r2"]].mean().round(4).reset_index()
        g["model"] = pd.Categorical(g["model"], models, ordered=True)
        g.sort_values([key, "model"]).to_csv(a.tables / f"{tag}_by_{key}.csv", index=False)
        pivot = g.pivot(index=key, columns="model", values="rmsse")[models]
        fig, ax = plt.subplots(figsize=(8, 4))
        pivot.plot.bar(ax=ax, width=0.85)
        ax.set_ylabel("RMSSE trung bình (macro)")
        ax.set_title(f"RMSSE theo {key} — {tag}")
        ax.legend([LABEL[m] for m in models], fontsize=7, ncol=2)
        savefig(fig, a.figures / f"{tag}_rmsse_by_{key}.png")

    # 3. weekday and horizon error from predictions joined to labels by key
    sales = load_raw_data(a.data_dir, table_names=("sales_train_evaluation", "calendar"))
    actuals = to_wide(sales["sales_train_evaluation"]).loc[sorted(pred["series_id"].unique())]
    cal = sales["calendar"].set_index("d")
    lab = actuals[[f"d_{n}" for n in range(SPLITS[tag][0], SPLITS[tag][1] + 1)]].stack().rename("y_true")
    lab = lab.rename_axis(["series_id", "d"]).reset_index()
    j = pred.merge(lab, on=["series_id", "d"], how="left", validate="many_to_one")
    j["abs_err"] = (j["y_true"] - j["y_pred"]).abs()
    j["wday"] = j["d"].map(cal["wday"])  # 1 = Saturday in M5 calendar
    j["weekday"] = j["d"].map(pd.to_datetime(cal["date"]).dt.day_name())
    for key, name in (("weekday", "weekday"), ("horizon", "horizon")):
        t = j.groupby([key, "model"])["abs_err"].mean().unstack("model")[models]
        if key == "weekday":
            t = t.loc[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
        t.round(4).to_csv(a.tables / f"{tag}_mae_by_{name}.csv")
        fig, ax = plt.subplots(figsize=(8, 4))
        t.plot(ax=ax, marker="o" if key == "weekday" else None)
        ax.set_ylabel("MAE gộp (đơn vị/ngày)")
        ax.set_title(f"MAE theo {name} — {tag}")
        ax.legend([LABEL[m] for m in models], fontsize=7, ncol=2)
        savefig(fig, a.figures / f"{tag}_mae_by_{name}.png")

    # 4. runtime / accuracy trade-off
    rows = []
    for m in models:
        t = timings[m]
        fp = t.get("fit_predict_sum_s", t.get("fit_s", 0) + t.get("predict_s", 0))
        rows.append({"model": m, "fit_predict_s": fp, "wall_s": t.get("wall_s", t.get("fit_s", 0) + t.get("predict_s", 0)),
                     "scope": t.get("scope", ""), "run_id": t.get("run_id", ""),
                     "macro_rmsse": float(overall.loc[m, "macro_rmsse"]), "macro_mae": float(overall.loc[m, "macro_mae"])})
    rt = pd.DataFrame(rows)
    rt.round(3).to_csv(a.tables / f"{tag}_runtime_tradeoff.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in rt.iterrows():
        ax.scatter(max(r["fit_predict_s"], 1e-3), r["macro_rmsse"])
        ax.annotate(LABEL[r["model"]], (max(r["fit_predict_s"], 1e-3), r["macro_rmsse"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("fit+predict (giây, thang log; không cùng phạm vi giữa mô hình cục bộ/global)")
    ax.set_ylabel("RMSSE macro")
    ax.set_title(f"Độ chính xác và thời gian — {tag}")
    savefig(fig, a.figures / f"{tag}_accuracy_runtime.png")

    # 5. win-rate versus Seasonal Naive (per-series MAE; descriptive, series are not independent)
    mae = ps.pivot(index="series_id", columns="model", values="mae")
    win = pd.DataFrame({m: {"win_rate_vs_snaive": float((mae[m] < mae["seasonal_naive"]).mean()),
                            "tie_rate": float((mae[m] == mae["seasonal_naive"]).mean()),
                            "median_mae_diff": float((mae[m] - mae["seasonal_naive"]).median())}
                        for m in models if m != "seasonal_naive"}).T
    win.round(4).to_csv(a.tables / f"{tag}_win_rate_vs_snaive.csv")

    # 6. objective actual-vs-predicted examples: series at the 10th/50th/90th percentile of Train mean sales
    train = actuals[[f"d_{n}" for n in range(1, SPLITS["train"][1] + 1)]].mean(axis=1).sort_values()
    picks = [train.index[int(q * (len(train) - 1))] for q in (0.1, 0.5, 0.9)]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3 * len(picks)), sharex=True)
    for ax, sid in zip(axes, picks):
        sub = j[j["series_id"] == sid]
        ax.plot(sub[sub["model"] == models[0]]["horizon"], sub[sub["model"] == models[0]]["y_true"], "k-o", label="thực tế", ms=3)
        for m in models:
            s = sub[sub["model"] == m]
            ax.plot(s["horizon"], s["y_pred"], label=LABEL[m], lw=1)
        ax.set_title(f"{sid} (phân vị Train mean: {list(picks).index(sid) * 0.4 + 0.1:.1f})", fontsize=8)
    axes[0].legend(fontsize=6, ncol=4)
    axes[-1].set_xlabel("horizon (ngày)")
    savefig(fig, a.figures / f"{tag}_actual_vs_pred_examples.png")
    print(overall.round(4).to_string())


if __name__ == "__main__":
    main()
