"""Shared prediction schema (Phase 1). Labels are never part of a prediction file."""

import re

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ["run_id", "model", "split", "series_id", "d", "horizon", "y_pred"]
HORIZON = 28
SPLITS = {"train": (1, 1885), "validation": (1886, 1913), "test": (1914, 1941)}
_SUFFIX = re.compile(r"_(validation|evaluation)$")


def series_id_from_id(raw_id):
    """Stable key: item_store, independent of the _validation/_evaluation suffix."""
    return _SUFFIX.sub("", str(raw_id))


def series_id_from_parts(item_id, store_id):
    return f"{item_id}_{store_id}"


def day_number(d):
    return int(str(d).split("_")[1])


def forecast_days(split):
    """Day labels d_<n> of a forecast split, in horizon order 1..28."""
    start, end = SPLITS[split]
    if split == "train" or end - start + 1 != HORIZON:
        raise ValueError("only validation/test are 28-day forecast splits")
    return [f"d_{n}" for n in range(start, end + 1)]


def validate_predictions(pred, split, series_ids, *, models=None):
    """Raise ValueError unless pred is a complete, finite, duplicate-free forecast.

    Completeness is checked per (model, series_id, d); row order is irrelevant.
    """
    missing = [c for c in PREDICTION_COLUMNS if c not in pred.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if (pred["split"] != split).any():
        raise ValueError(f"all rows must have split == {split!r}")
    y = pd.to_numeric(pred["y_pred"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("y_pred contains NaN/inf/non-numeric values")
    if pred.duplicated(["run_id", "model", "series_id", "d"]).any():
        raise ValueError("duplicate (run_id, model, series_id, d) rows")
    days = forecast_days(split)
    expected_h = dict(zip(days, range(1, HORIZON + 1)))
    if set(pred["d"]) - set(days):
        raise ValueError("rows outside the split horizon")
    if (pred["d"].map(expected_h) != pred["horizon"]).any():
        raise ValueError("horizon does not match d")
    wanted = set(series_ids)
    if set(pred["series_id"]) - wanted:
        raise ValueError("predictions contain series outside the sample")
    for (run, model), grp in pred.groupby(["run_id", "model"], sort=False):
        if len(grp) != len(wanted) * HORIZON or set(grp["series_id"]) != wanted:
            raise ValueError(f"{model}/{run}: not exactly {HORIZON} days for every series")
    if models is not None and set(pred["model"]) != set(models):
        raise ValueError("model set differs from the expected list")
    return True
