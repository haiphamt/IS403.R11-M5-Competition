"""Forecast metrics for one series (or pooled values): MAE, RMSE, RMSSE, R2.

RMSSE follows the M5 definition: the squared error is scaled by the mean squared
one-step naive error on the training history, starting at the first non-zero sale.
"""

import numpy as np


def _as_1d(values, name):
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return arr


def _pair(y_true, y_pred):
    y_true, y_pred = _as_1d(y_true, "y_true"), _as_1d(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same number of values")
    return y_true, y_pred


def mae(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmsse(y_true, y_pred, y_train):
    """RMSE divided by the in-sample one-step naive RMSE of ``y_train``.

    Raises ValueError when the scale is zero (history constant after the first
    non-zero value, or too short), because the ratio would be NaN/inf.
    """
    y_true, y_pred = _pair(y_true, y_pred)
    train = _as_1d(y_train, "y_train")
    nonzero = np.flatnonzero(train)
    if nonzero.size == 0:
        raise ValueError("y_train is all zeros, RMSSE is undefined")
    scale = np.mean(np.diff(train[nonzero[0]:]) ** 2) if train.size - nonzero[0] > 1 else 0.0
    if scale == 0:
        raise ValueError("y_train has zero naive scale, RMSSE is undefined")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def r2(y_true, y_pred):
    """Coefficient of determination; undefined (ValueError) when y_true is constant."""
    y_true, y_pred = _pair(y_true, y_pred)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    if sst == 0:
        raise ValueError("y_true is constant, R2 is undefined")
    return float(1 - np.sum((y_true - y_pred) ** 2) / sst)
