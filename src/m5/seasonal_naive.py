"""Weekly seasonal-naive baseline; see https://otexts.com/fpp3/toolbox.html."""

import numpy as np


def seasonal_naive(history, horizon=28, season_length=7):
    """Repeat the last observed season without reading future observations."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(season_length, int) or season_length < 1:
        raise ValueError("season_length must be a positive integer")
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or len(values) < season_length:
        raise ValueError("history must be one series containing at least one full season")
    if not np.isfinite(values).all():
        raise ValueError("history contains NaN or infinity")
    return np.tile(values[-season_length:], (horizon + season_length - 1) // season_length)[:horizon]
