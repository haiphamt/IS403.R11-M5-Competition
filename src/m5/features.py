"""D04: leakage-safe lag/rolling features on a wide matrix (series x day).

Feature at day t only reads y[:, <= t - 28]; with origin T and a 28-day horizon every
source day is <= T. ``y`` may contain NaN for unknown days: they never enter a
feature of an in-horizon day.
"""

import numpy as np
import pandas as pd

LAGS = (28, 35, 42)
WINDOWS = (7, 28)
SHIFT = 28


def lag_roll_features(y, *, lags=LAGS, windows=WINDOWS, shift=SHIFT):
    """y: (n_series, n_days) float. Returns dict name -> (n_series, n_days) array."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 2:
        raise ValueError("y must be 2-D (series x day)")
    n, t = y.shape
    out = {}
    for lag in lags:
        if lag < 28:
            raise ValueError("lag < 28 would use unobserved days inside the horizon")
        arr = np.full((n, t), np.nan)
        arr[:, lag:] = y[:, : t - lag]
        out[f"lag_{lag}"] = arr
    base = np.full((n, t), np.nan)
    base[:, shift:] = y[:, : t - shift]
    frame = pd.DataFrame(base.T)
    for w in windows:
        roll = frame.rolling(w, min_periods=w)
        out[f"rolling_mean_{w}"] = roll.mean().to_numpy().T
        out[f"rolling_std_{w}"] = roll.std(ddof=1).to_numpy().T
    return out


def build_origin_features(y_history, origin, horizon=28, **kwargs):
    """Features for days origin+1..origin+horizon using only y_history[:, :origin].

    Returns (train_features, forecast_features) as dicts of arrays; train columns are
    days whose lag/rolling windows are complete and whose target day is <= origin.
    """
    y_history = np.asarray(y_history, dtype=float)
    if y_history.shape[1] < origin:
        raise ValueError("history shorter than origin")
    padded = np.hstack([y_history[:, :origin], np.full((y_history.shape[0], horizon), np.nan)])
    feats = lag_roll_features(padded, **kwargs)
    start = max(max(LAGS), SHIFT + max(WINDOWS) - 1)  # first day with complete windows (55)
    train = {k: v[:, start:origin] for k, v in feats.items()}
    future = {k: v[:, origin:origin + horizon] for k, v in feats.items()}
    for name, arr in future.items():
        if np.isnan(arr).any():
            raise ValueError(f"forecast feature {name} contains NaN: history too short")
    return train, future
