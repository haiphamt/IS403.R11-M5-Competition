"""Join predictions to labels by key (never by row order) and score them.

Policy (fixed before looking at Test):
* per-series MAE/RMSE/RMSSE/R2 via m5.metrics; RMSSE scale comes from d_1..origin
  history of that series (leading zeros dropped, one-step naive).
* Undefined RMSSE/R2 (zero scale, constant actuals) are NOT patched with epsilon;
  they are NaN with a reason and counted in the aggregate.
* macro = mean of per-series values over series where the metric is defined;
  pooled MAE/RMSE are computed over all (series, day) pairs.
"""

import numpy as np
import pandas as pd

from . import metrics
from .schema import HORIZON, SPLITS, forecast_days, validate_predictions


def _origin(split):
    return SPLITS[split][0] - 1


def per_series_metrics(pred, actuals, split):
    """pred: schema frame for ONE run/model set; actuals: wide frame indexed by series_id.

    Returns one row per (run_id, model, series_id).
    """
    days = forecast_days(split)
    origin = _origin(split)
    history_cols = [f"d_{n}" for n in range(1, origin + 1)]
    rows = []
    for (run, model, sid), grp in pred.groupby(["run_id", "model", "series_id"], sort=False):
        grp = grp.set_index("d").loc[days]  # align by day label
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        y_true = actuals.loc[sid, days].to_numpy(dtype=float)
        history = actuals.loc[sid, history_cols].to_numpy(dtype=float)
        row = {"run_id": run, "model": model, "series_id": sid,
               "mae": metrics.mae(y_true, y_pred), "rmse": metrics.rmse(y_true, y_pred),
               "rmsse": np.nan, "r2": np.nan, "rmsse_note": "", "r2_note": ""}
        for name, call in (("rmsse", lambda: metrics.rmsse(y_true, y_pred, history)),
                           ("r2", lambda: metrics.r2(y_true, y_pred))):
            try:
                row[name] = call()
            except ValueError as error:
                row[f"{name}_note"] = str(error)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(series_scores, pred, actuals, split):
    """Macro (defined series only) and pooled summaries per model, with undefined counts."""
    days = forecast_days(split)
    out = []
    for (run, model), grp in series_scores.groupby(["run_id", "model"], sort=False):
        p = pred[(pred["run_id"] == run) & (pred["model"] == model)]
        merged = p.merge(
            actuals[days].stack().rename("y_true").rename_axis(["series_id", "d"]).reset_index(),
            on=["series_id", "d"], how="left", validate="many_to_one")
        if merged["y_true"].isna().any():
            raise ValueError("labels missing for some predictions")
        e = merged["y_true"] - merged["y_pred"]
        out.append({
            "run_id": run, "model": model, "n_series": len(grp),
            "macro_mae": grp["mae"].mean(), "macro_rmse": grp["rmse"].mean(),
            "macro_rmsse": grp["rmsse"].mean(), "macro_r2": grp["r2"].mean(),
            "pooled_mae": float(np.abs(e).mean()), "pooled_rmse": float(np.sqrt((e ** 2).mean())),
            "n_rmsse_undefined": int(grp["rmsse"].isna().sum()),
            "n_r2_undefined": int(grp["r2"].isna().sum()),
        })
    return pd.DataFrame(out)


def evaluate(pred, actuals, split):
    """Validate, score and aggregate. Returns (per_series, summary)."""
    validate_predictions(pred, split, list(actuals.index))
    per_series = per_series_metrics(pred, actuals, split)
    return per_series, aggregate(per_series, pred, actuals, split)
