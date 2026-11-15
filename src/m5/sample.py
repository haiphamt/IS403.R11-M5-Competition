"""D02: stratified sample (store x category), fixed seed, Train-only statistics."""

import hashlib

import numpy as np
import pandas as pd

from .schema import SPLITS, series_id_from_id


def select_stratified(sales, *, per_stratum=40, seed=42, strata=("store_id", "cat_id")):
    """Pick per_stratum series per stratum. Deterministic: input order is irrelevant."""
    meta = sales[["id", *strata]].copy()
    meta["series_id"] = meta["id"].map(series_id_from_id)
    meta = meta.sort_values("series_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen = []
    for key, grp in meta.groupby(list(strata), sort=True, observed=True):
        if len(grp) < per_stratum:
            raise ValueError(f"stratum {key} has only {len(grp)} series (< {per_stratum})")
        idx = rng.choice(len(grp), size=per_stratum, replace=False)
        chosen.append(grp.iloc[np.sort(idx)])
    manifest = pd.concat(chosen).sort_values("series_id").reset_index(drop=True)
    return manifest[["series_id", *strata]]


def manifest_hash(manifest):
    return hashlib.sha256("\n".join(sorted(manifest["series_id"])).encode()).hexdigest()


def representativeness(wide, sample_ids):
    """Compare zero ratio / mean sales of the sample with all series, Train days only."""
    start, end = SPLITS["train"]
    train = wide[[f"d_{n}" for n in range(start, end + 1)]]
    def stats(frame):
        return {"n_series": len(frame), "zero_ratio": float((frame.to_numpy() == 0).mean()),
                "mean_sales": float(frame.to_numpy().mean())}
    return pd.DataFrame({"all": stats(train), "sample": stats(train.loc[list(sample_ids)])})
