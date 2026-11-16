"""Run the seven core models on the stratified sample for one split and score them.

    python scripts/run_experiment.py --split validation          # origin d_1885
    python scripts/run_experiment.py --split test                # origin d_1913, needs tag config-freeze
    python scripts/run_experiment.py --split validation --max-series 24 --models seasonal_naive ets   # smoke

Everything lands in results/runs/<run_id>/: predictions.parquet (schema run_id/model/split/
series_id/d/horizon/y_pred), metrics_per_series.csv, metrics_summary.csv, timings.json,
environment.json, diagnostics (ETS configs, Croston z/p, fallbacks, LR coefficients,
feature importance) and checkpoints (ckpt/, Git-ignored) so an interrupted run resumes with
--run-id <same id>. Labels after the origin are read only by the evaluator.
"""

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5 import evaluate, global_models, univariate  # noqa: E402
from m5.data import load_raw_data  # noqa: E402
from m5.panel import build_panel  # noqa: E402
from m5.sample import manifest_hash  # noqa: E402
from m5.schema import HORIZON, SPLITS, forecast_days, validate_predictions  # noqa: E402

LOCAL_MODELS = ["seasonal_naive", "ets", "croston", "croston_sba"]
ALL_MODELS = ["seasonal_naive", "linear_regression", "ets", "croston", "lightgbm", "xgboost", "catboost"]
OPTIONAL_MODELS = ["croston_sba", "lightgbm_noprice", "lightgbm_tuned"]  # X01, X02, T01 (frozen in config-freeze-v2)
FREEZE_TAG = "config-freeze"
FREEZE_TAG_V2 = "config-freeze-v2"


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def package_versions():
    out = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "xgboost", "catboost",
                 "statsforecast", "numba", "optuna", "scipy"):
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def _worker_init():
    univariate.warmup()


def run_local(model, panel, origin, workers, chunk, ckpt_dir):
    """Forecast every series with a local model; resumable per chunk of series."""
    ids = panel.series_ids
    jobs = [(i, ids[i:i + chunk]) for i in range(0, len(ids), chunk)]
    results, todo = {}, []
    for i, sub in jobs:
        f = ckpt_dir / f"{model}_{i:05d}.pkl"
        if f.exists():
            results[i] = pickle.loads(f.read_bytes())
        else:
            todo.append((i, sub, f))
    t0 = time.perf_counter()
    if todo:
        hist = {sid: panel.y[k, :origin] for k, sid in enumerate(ids)}
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
            futures = [(i, f, ex.submit(univariate.run_chunk, model, sub, [hist[s] for s in sub]))
                       for i, sub, f in todo]
            for i, f, fut in futures:
                res = fut.result()
                f.write_bytes(pickle.dumps(res))
                results[i] = res
                print(f"  {model}: chunk {i // chunk + 1}/{len(jobs)} done", flush=True)
    wall = time.perf_counter() - t0
    rows = [r for i in sorted(results) for r in results[i]]
    return rows, wall


def to_schema(run_id, model, split, series_ids, matrix):
    days = forecast_days(split)
    n = len(series_ids)
    return pd.DataFrame({
        "run_id": run_id, "model": model, "split": split,
        "series_id": np.repeat(series_ids, HORIZON), "d": np.tile(days, n),
        "horizon": np.tile(np.arange(1, HORIZON + 1), n), "y_pred": np.asarray(matrix).ravel()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS + OPTIONAL_MODELS)
    ap.add_argument("--lgbm-params", default=None, help="JSON with the Optuna best parameters (for lightgbm_tuned)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "sample_series_ids.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "runs"))
    ap.add_argument("--run-id", default=None, help="reuse to resume an interrupted run")
    ap.add_argument("--workers", type=int, default=4, help="processes for local models")
    ap.add_argument("--n-jobs", type=int, default=8, help="threads for global models")
    ap.add_argument("--chunk", type=int, default=50, help="series per checkpoint for local models")
    ap.add_argument("--max-series", type=int, default=None, help="smoke run on the first N sample series")
    args = ap.parse_args()

    if args.split == "test" and args.max_series is None and not git("rev-parse", "-q", "--verify", f"refs/tags/{FREEZE_TAG}"):
        raise SystemExit(f"Test is locked: tag {FREEZE_TAG} not found (freeze configs first).")
    if args.split == "test" and args.max_series is None and set(args.models) & set(OPTIONAL_MODELS)             and not git("rev-parse", "-q", "--verify", f"refs/tags/{FREEZE_TAG_V2}"):
        raise SystemExit(f"Test of optional models is locked: tag {FREEZE_TAG_V2} not found.")
    lgbm_params = json.loads(Path(args.lgbm_params).read_text()) if args.lgbm_params else None
    split = args.split
    origin = SPLITS[split][0] - 1
    sha = git("rev-parse", "HEAD")
    run_id = args.run_id or f"{split[:3]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha[:7] or 'nogit'}"
    run_dir = Path(args.out) / run_id
    ckpt = run_dir / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} origin=d_{origin}", flush=True)

    t_io = time.perf_counter()
    manifest = pd.read_csv(args.manifest)
    ids = list(manifest["series_id"])[: args.max_series]
    raw = load_raw_data(args.data_dir, table_names=("sales_train_evaluation", "calendar", "sell_prices"))
    panel = build_panel(raw["sales_train_evaluation"], raw["calendar"], raw["sell_prices"], ids)
    io_s = time.perf_counter() - t_io
    del raw

    timings = {"data_io_panel_s": io_s}
    frames, diag = [], {"ets": [], "croston": [], "fallbacks": []}
    warm = univariate.warmup() if any(m in LOCAL_MODELS for m in args.models) else 0.0
    timings["warmup_s"] = warm

    glob = [m for m in args.models if m in global_models.MODELS]
    table = None
    if glob:
        t0 = time.perf_counter()
        table = global_models.build_table(panel, origin)
        timings["feature_build_s"] = time.perf_counter() - t0
        print(f"feature table: train {len(table[0]):,} rows, forecast {len(table[2]):,} rows", flush=True)

    params, importances = {}, {}
    for model in args.models:
        print(f"== {model}", flush=True)
        if model in LOCAL_MODELS:
            rows, wall = run_local(model, panel, origin, args.workers, args.chunk, ckpt)
            matrix = np.vstack([r["forecast"] for r in rows])
            assert [r["series_id"] for r in rows] == ids
            timings[model] = {"fit_predict_sum_s": float(sum(r["fit_predict_s"] for r in rows)),
                              "wall_s": wall, "workers": args.workers, "n_series": len(rows),
                              "n_fallback": int(sum(r["fallback"] for r in rows)),
                              "scope": "sum over series of per-series fit+predict, single thread each"}
            for r in rows:
                if r["fallback"]:
                    diag["fallbacks"].append({"model": model, "series_id": r["series_id"], "error": r["error"]})
                if model in ("ets", "croston"):
                    diag[model].append({"series_id": r["series_id"], **r["info"]})
        else:
            X, y, Xf, keys = table
            pred, info, fitted = global_models.fit_predict(model, X, y, Xf, n_jobs=args.n_jobs, lgbm_params=lgbm_params)
            keys = keys.assign(pred=pred)
            order = keys.pivot(index="series_id", columns="d_num", values="pred").loc[ids]
            matrix = order.to_numpy()
            timings[model] = {**info, "n_train_rows": int(len(X)), "threads": args.n_jobs,
                              "scope": "pooled model: fit + predict wall time (features timed separately)"}
            params[model] = {k: str(v) for k, v in global_models.model_parameters(model, fitted).items()}
            importances[model] = global_models.importance_or_coefficients(model, fitted)
        raw_neg = int((matrix < 0).sum())
        timings[model]["n_negative_raw_predictions"] = raw_neg
        frames.append(to_schema(run_id, model, split, ids, np.maximum(matrix, 0.0)))  # uniform clip at 0

    pred = pd.concat(frames, ignore_index=True)
    validate_predictions(pred, split, ids)
    pred.to_parquet(run_dir / "predictions.parquet", index=False)

    actuals = pd.DataFrame(panel.y, index=ids, columns=[f"d_{n}" for n in range(1, panel.y.shape[1] + 1)])
    per_series, summary = evaluate.evaluate(pred, actuals, split)
    summary["fit_predict_s"] = summary["model"].map(
        lambda m: timings[m].get("fit_predict_sum_s", timings[m].get("fit_s", 0) + timings[m].get("predict_s", 0)))
    per_series.to_csv(run_dir / "metrics_per_series.csv", index=False)
    summary.to_csv(run_dir / "metrics_summary.csv", index=False)

    for name in ("ets", "croston", "fallbacks"):
        if diag[name]:
            pd.DataFrame(diag[name]).to_csv(run_dir / f"diag_{name}.csv", index=False)
    for model, table_ in importances.items():
        table_.to_csv(run_dir / f"diag_{model}_{'coefficients' if model == 'linear_regression' else 'importance'}.csv", index=False)
    (run_dir / "model_parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    raw_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / "raw"
    env = {"run_id": run_id, "split": split, "origin_day": origin, "git_sha": sha,
           "git_dirty": bool(git("status", "--porcelain")), "utc": datetime.now(timezone.utc).isoformat(),
           "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": __import__("os").cpu_count(), "packages": package_versions(),
           "sample": {"n_series": len(ids), "manifest_sha256": manifest_hash(manifest[manifest["series_id"].isin(ids)]),
                      "seed": 42, "per_stratum": 40},
           "data_sha256": {p.name: sha256(p) for p in sorted(raw_dir.glob("*.csv"))} if raw_dir.exists() else {},
           "models": args.models, "postprocessing": "clip y_pred at 0 for every model; no rounding",
           "args": {k: v for k, v in vars(args).items()}}
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
