"""T02/F06: Ridge stacking of Linear Regression, LightGBM (Default or Tuned), XGBoost, CatBoost.

    python scripts/run_stacking.py --split validation --oof results/oof/*.parquet --lgbm default \
        --base-runs results/runs/val-A results/runs/val-B --out results/runs

The meta-learner is fitted ONLY on the out-of-fold backtests inside Train (scripts/backtest_oof.py, labels <= d_1885) and its alpha is
chosen by leave-one-origin-out CV there. The same frozen meta-learner is then applied to the base forecasts of the evaluated split, so
Validation and Test scores are independent of the meta fit. Test needs tag config-freeze-v2.
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5 import evaluate, optional  # noqa: E402
from m5.schema import SPLITS, validate_predictions  # noqa: E402

BASES = ["linear_regression", "lightgbm", "xgboost", "catboost"]


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--oof", nargs="+", required=True, help="parquet files from backtest_oof.py (globs allowed)")
    ap.add_argument("--lgbm", choices=["default", "tuned"], required=True, help="which LightGBM feeds the stack (T01 rule)")
    ap.add_argument("--base-runs", nargs="+", required=True, type=Path, help="run dirs holding the base forecasts of --split")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    if a.split == "test" and not git("rev-parse", "-q", "--verify", "refs/tags/config-freeze-v2"):
        raise SystemExit("Test of the stack is locked: tag config-freeze-v2 not found.")
    lg = "lightgbm" if a.lgbm == "default" else "lightgbm_tuned"
    cols = {"linear_regression": "linear_regression", "lightgbm": lg, "xgboost": "xgboost", "catboost": "catboost"}

    files = [f for pat in a.oof for f in glob.glob(pat)]
    oof = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    wide = oof.pivot_table(index=["origin", "series_id", "d_num"], columns="model", values="y_pred").join(
        oof.drop_duplicates(["origin", "series_id", "d_num"]).set_index(["origin", "series_id", "d_num"])["y_true"])
    n_keys = len(wide)
    wide = wide[[cols[b] for b in BASES] + ["y_true"]].rename(columns={v: k for k, v in cols.items()}).dropna().reset_index()
    if len(wide) != n_keys:  # a base model missing for some (origin, series, day) would silently shrink the meta training set
        raise SystemExit(f"{n_keys - len(wide)} OOF rows lack a forecast from at least one base model; refusing to fit on a partial set")
    t0 = time.perf_counter()
    meta, alpha, cv = optional.fit_ridge_meta(wide, BASES)
    meta_s = time.perf_counter() - t0
    print(f"meta fitted on {len(wide):,} OOF rows, alpha={alpha}, CV rmse={cv}")

    pred = pd.concat([pd.read_parquet(r / "predictions.parquet") for r in a.base_runs], ignore_index=True)
    pw = pred.pivot_table(index=["series_id", "d"], columns="model", values="y_pred")[[cols[b] for b in BASES]]
    pw.columns = BASES
    ids = sorted(pw.index.get_level_values(0).unique())
    stacked = np.maximum(meta.predict(pw.to_numpy(float)), 0)  # same clip-at-0 policy as every model
    start = SPLITS[a.split][0]
    out = pw.reset_index()[["series_id", "d"]].assign(y_pred=stacked)
    out["horizon"] = out["d"].str.slice(2).astype(int) - start + 1
    sha = git("rev-parse", "HEAD")
    run_id = f"{a.split[:3]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha[:7] or 'nogit'}-stack"
    out = out.assign(run_id=run_id, model="stacking", split=a.split)[["run_id", "model", "split", "series_id", "d", "horizon", "y_pred"]]
    validate_predictions(out, a.split, ids)

    panel = optional.load_sample_panel(a.data_dir)
    actuals = pd.DataFrame(panel.y, index=panel.series_ids, columns=[f"d_{n}" for n in range(1, panel.y.shape[1] + 1)]).loc[ids]
    per_series, summary = evaluate.evaluate(out, actuals, a.split)
    summary["fit_predict_s"] = meta_s
    run = a.out / run_id
    run.mkdir(parents=True, exist_ok=True)
    out.to_parquet(run / "predictions.parquet", index=False)
    per_series.to_csv(run / "metrics_per_series.csv", index=False)
    summary.to_csv(run / "metrics_summary.csv", index=False)
    coef = pd.DataFrame({"feature": BASES + ["intercept"], "coefficient": list(meta.coef_) + [float(meta.intercept_)]})
    coef.to_csv(run / "diag_stacking_coefficients.csv", index=False)
    (run / "timings.json").write_text(json.dumps({"stacking": {"fit_s": meta_s, "predict_s": 0.0, "n_oof_rows": len(wide),
        "scope": "Ridge meta fit on Train out-of-fold backtests (base fits not included)", "n_negative_raw_predictions": int((meta.predict(pw.to_numpy(float)) < 0).sum())}}, indent=2))
    (run / "environment.json").write_text(json.dumps({
        "run_id": run_id, "split": a.split, "origin_day": start - 1, "git_sha": sha, "utc": datetime.now(timezone.utc).isoformat(),
        "models": ["stacking"], "meta": {"alpha": alpha, "cv_rmse_leave_one_origin_out": {str(k): v for k, v in cv.items()},
        "lgbm_in_stack": a.lgbm, "bases": BASES, "oof_files": [Path(f).name for f in files], "oof_rows": len(wide), "base_runs": [r.name for r in a.base_runs]},
        "postprocessing": "clip y_pred at 0; no rounding"}, indent=2))
    print(summary.round(4).to_string(index=False)); print(coef.round(4).to_string(index=False)); print("saved", run)


if __name__ == "__main__":
    main()
