"""ETS and Croston forecasts for one series, using statsforecast.

AutoETS picks the ETS form by AICc; CrostonClassic is for intermittent demand.
Forecasts are returned as fitted, with no clipping at zero.
"""

import numpy as np
from statsforecast.models import AutoETS, CrostonClassic


def _history(values, horizon, season_length):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(season_length, int) or season_length < 1:
        raise ValueError("season_length must be a positive integer")
    history = np.asarray(values, dtype=float)
    if history.ndim != 1 or len(history) < 2 * season_length:
        raise ValueError("history must be one series with at least two full seasons")
    if not np.isfinite(history).all():
        raise ValueError("history contains NaN or infinity")
    return history


def _checked(model, history, horizon):
    forecast = np.asarray(model.forecast(y=history, h=horizon)["mean"], dtype=float)
    if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
        raise RuntimeError(f"{type(model).__name__} returned an invalid forecast")
    return forecast


def ets_forecast(history, horizon=28, season_length=7):
    """Fit AutoETS (AICc) on ``history`` and forecast ``horizon`` steps."""
    history = _history(history, horizon, season_length)
    return _checked(AutoETS(season_length=season_length), history, horizon)


def croston_forecast(history, horizon=28, season_length=7):
    """Fit classic Croston on ``history``; the forecast is a flat line."""
    history = _history(history, horizon, season_length)
    return _checked(CrostonClassic(), history, horizon)
