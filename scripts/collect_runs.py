"""Merge several run directories (e.g. Kaggle + Colab jobs) and score them together.

    python scripts/collect_runs.py --split validation --runs results/runs/val-A results/runs/val-B --out results/tables/validation

Predictions are concatenated by their own run_id/model, re-validated and evaluated against the
labels of the split; per-run timings and environment files are copied next to the tables so every
number stays traceable to a run_id. Do not mix runs with different samples or origins.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m5 import evaluate  # noqa: E402
from m5.data import load_raw_data  # noqa: E402
from m5.schema import validate_predictions  # noqa: E402
from m5.split import to_wide  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    pred = pd.concat([pd.read_parquet(r / "predictions.parquet") for r in a.runs], ignore_index=True)
    ids = sorted(pred["series_id"].unique())
    validate_predictions(pred, a.split, ids)
    sales = load_raw_data(a.data_dir, table_names=("sales_train_evaluation",))["sales_train_evaluation"]
    actuals = to_wide(sales).loc[ids]
    per_series, summary = evaluate.evaluate(pred, actuals, a.split)
    timings = {}
    for r in a.runs:
        t = json.loads((r / "timings.json").read_text())
        run_id = json.loads((r / "environment.json").read_text())["run_id"]
        for model, v in t.items():
            if isinstance(v, dict):
                timings[model] = {**v, "run_id": run_id}
    summary["fit_predict_s"] = summary["model"].map(
        lambda m: timings[m].get("fit_predict_sum_s", timings[m].get("fit_s", 0) + timings[m].get("predict_s", 0)))
    a.out.mkdir(parents=True, exist_ok=True)
    per_series.to_csv(a.out / "metrics_per_series.csv", index=False)
    summary.to_csv(a.out / "metrics_summary.csv", index=False)
    pred.to_parquet(a.out / "predictions.parquet", index=False)
    (a.out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    for r in a.runs:
        for f in r.glob("diag_*.csv"):
            shutil.copy(f, a.out / f.name)
        for f in ("model_parameters.json", "environment.json"):
            shutil.copy(r / f, a.out / f"{r.name}_{f}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
