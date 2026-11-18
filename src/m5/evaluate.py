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


def _scales_and_notes(history):
    """RMSSE denominators per series via m5.metrics.rmsse_scale (NaN plus the error text when undefined)."""
    scales, notes = np.full(len(history), np.nan), [""] * len(history)
    for i, row in enumerate(history):
        try:
            scales[i] = metrics.rmsse_scale(row)
        except ValueError as error:
            notes[i] = str(error)
    return scales, notes


def per_series_metrics(pred, actuals, split):
    """pred: schema frame for ONE run/model set; actuals: wide frame indexed by series_id.

    Returns one row per (run_id, model, series_id). Vectorised over series (30,490 series x several models),
    with the same definitions as m5.metrics: RMSSE scale from d_1..origin, NaN plus a reason when undefined.
    """
    days = forecast_days(split)
    origin = _origin(split)
    history_cols = [f"d_{n}" for n in range(1, origin + 1)]
    cache = {}
    out = []
    for (run, model), grp in pred.groupby(["run_id", "model"], sort=False):
        sids = pd.unique(grp["series_id"])
        wide = grp.pivot(index="series_id", columns="d", values="y_pred").loc[sids, days]  # align by key and day label
        y_pred = wide.to_numpy(dtype=float)
        y_true = actuals.loc[sids, days].to_numpy(dtype=float)
        if not (np.isfinite(y_pred).all() and np.isfinite(y_true).all()):
            raise ValueError(f"{run}/{model}: NaN or infinity in predictions or labels")
        key = tuple(sids)
        if key not in cache:  # every model of a run shares the same series and history
            cache[key] = _scales_and_notes(actuals.loc[sids, history_cols].to_numpy(dtype=float))
        scales, scale_notes = cache[key]
        err = y_true - y_pred
        mse = np.mean(err ** 2, axis=1)
        sst = np.sum((y_true - y_true.mean(axis=1, keepdims=True)) ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(sst == 0, np.nan, 1 - np.sum(err ** 2, axis=1) / sst)
        out.append(pd.DataFrame({
            "run_id": run, "model": model, "series_id": sids,
            "mae": np.mean(np.abs(err), axis=1), "rmse": np.sqrt(mse),
            "rmsse": np.sqrt(mse / scales), "r2": r2,
            "rmsse_note": scale_notes, "r2_note": np.where(sst == 0, "y_true is constant, R2 is undefined", "")}))
    return pd.concat(out, ignore_index=True)


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
