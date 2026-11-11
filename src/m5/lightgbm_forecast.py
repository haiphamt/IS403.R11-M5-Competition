"""LightGBM forecast for one series: a per-series regressor, direct 28-day horizon.

Target day t uses only y up to day t-28 (lags 28/35/42, rolling mean and std
shifted by 28), so a forecast day never reads an observation it would not have.
shortcut: per-series model without calendar, SNAP, price or categorical features
(features.yaml not applied yet); upgrade in the full feature pipeline (D-tasks).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

LAGS = (28, 35, 42)
WINDOWS = (7, 28)
SHIFT = 28
START = max(max(LAGS), SHIFT + max(WINDOWS) - 1)  # first day where every feature exists


def _feature_matrix(y, days):
    """Features for target ``days`` (indices into ``y``); reads only y[days - k] with k >= 28."""
    columns = [y[days - lag] for lag in LAGS]
    for window in WINDOWS:
        rolling = pd.Series(y).rolling(window)
        columns += [rolling.mean().to_numpy()[days - SHIFT], rolling.std().to_numpy()[days - SHIFT]]
    return np.column_stack(columns)


def lightgbm_forecast(history, horizon=28):
    """Fit LGBMRegressor on ``history`` and forecast ``horizon`` days (1 to 28)."""
    if not isinstance(horizon, int) or not 1 <= horizon <= SHIFT:
        raise ValueError(f"horizon must be an integer from 1 to {SHIFT}")
    y = np.asarray(history, dtype=float)
    if y.ndim != 1 or len(y) <= START:
        raise ValueError(f"history must be one series longer than {START} days")
    if not np.isfinite(y).all():
        raise ValueError("history contains NaN or infinity")

    n = len(y)
    days = np.arange(START, n + horizon)
    features = _feature_matrix(y, days)
    is_train = days < n
    model = LGBMRegressor(random_state=42, verbose=-1)
    model.fit(features[is_train], y[days[is_train]])
    forecast = np.asarray(model.predict(features[~is_train]), dtype=float)
    if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
        raise RuntimeError("LightGBM returned an invalid forecast")
    return forecast
