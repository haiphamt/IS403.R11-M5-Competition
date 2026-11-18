"""T01: Optuna search for LightGBM on Validation only (<= 30 trials or 2 hours), origin d_1885.

    python scripts/tune_lightgbm.py --out results/tuning

Objective = Validation macro-RMSSE (same metric/policy as the core table, predictions clipped at 0). Test labels are never read.
Outputs: trials.csv, best_params.json (use with run_experiment.py --lgbm-params), tuning_summary.json (tuning time kept apart
from the Default cost; `choice` = "tuned" only if its Validation score beats the Default one).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5 import global_models, optional  # noqa: E402

ORIGIN = 1885
DEFAULTISH = dict(n_estimators=100, learning_rate=0.1, num_leaves=31, min_child_samples=20, subsample=1.0,
                  colsample_bytree=1.0, reg_lambda=0.001)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tuning")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=7200, help="seconds")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-series", type=int, default=None, help="smoke only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    panel = optional.load_sample_panel(a.data_dir, max_series=a.max_series)
    X, y, Xf, keys = global_models.build_table(panel, ORIGIN)
    ids = panel.series_ids
    truth = panel.y[:, ORIGIN:ORIGIN + 28]
    scales = optional.rmsse_scales(panel.y, ORIGIN)

    def score(name, params=None):
        pred, info, _ = global_models.fit_predict(name, X, y, Xf, n_jobs=a.n_jobs, lgbm_params=params)
        wide = keys.assign(pred=np.maximum(pred, 0)).pivot(index="series_id", columns="d_num", values="pred").loc[ids].to_numpy()
        return optional.macro_rmsse(wide, truth, scales), info["fit_s"]

    default_score, default_fit = score("lightgbm")

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        value, _ = score("lightgbm_tuned", params)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(DEFAULTISH)  # start the search at the default region
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)
    tune_s = time.perf_counter() - t0

    study.trials_dataframe().to_csv(a.out / "trials.csv", index=False)
    best = {**study.best_params, "subsample_freq": 1}
    (a.out / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    summary = {"origin": ORIGIN, "metric": "validation macro_rmsse (clip at 0)", "n_trials": len(study.trials),
               "tuning_seconds": tune_s, "best_value": study.best_value, "default_value": default_score,
               "default_fit_s": default_fit, "choice": "tuned" if study.best_value < default_score else "default",
               "optuna": optuna.__version__, "seed": 42,
               "note": "best of the trials is compared with Default on the same Validation window, so the gap is optimistic; Test is the independent check"}
    (a.out / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
