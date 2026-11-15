"""D03: time splits of a sales table in wide form, keyed by series_id."""

import pandas as pd

from .schema import SPLITS, series_id_from_id


def to_wide(sales):
    """sales_train_evaluation-style frame -> wide frame indexed by stable series_id."""
    wide = sales.copy()
    wide.index = [series_id_from_id(i) for i in wide["id"]]
    wide.index.name = "series_id"
    if wide.index.duplicated().any():
        raise ValueError("duplicate series_id after stripping suffix")
    return wide[[c for c in wide.columns if c.startswith("d_")]]


def split_wide(wide, series_ids=None):
    """Return {'train','validation','test'} wide frames; no overlapping days."""
    if series_ids is not None:
        wide = wide.loc[list(series_ids)]
    parts = {}
    for name, (start, end) in SPLITS.items():
        cols = [f"d_{n}" for n in range(start, end + 1)]
        absent = [c for c in cols if c not in wide.columns]
        if absent:
            raise ValueError(f"{name}: missing day columns, first {absent[0]}")
        parts[name] = wide[cols]
    assert_no_overlap(parts)
    return parts


def assert_no_overlap(parts):
    seen = set()
    for name, frame in parts.items():
        cols = set(frame.columns)
        if seen & cols:
            raise ValueError(f"{name} overlaps an earlier split")
        seen |= cols


def to_long(frame, value_name="y"):
    long = frame.stack().rename(value_name).reset_index()
    long.columns = ["series_id", "d", value_name]
    long["d_num"] = long["d"].str.slice(2).astype(int)
    return long
