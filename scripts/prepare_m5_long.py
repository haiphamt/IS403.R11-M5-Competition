"""Convert M5 evaluation sales to daily rows with calendar and weekly prices."""

import argparse
from collections.abc import Iterator
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from m5.data import SALES_IDS, load_raw_data


def long_batches(sales, calendar, prices, day_columns, batch_series) -> Iterator[pd.DataFrame]:
    calendar = calendar.loc[calendar["d"].isin(day_columns)].copy()
    if len(calendar) != len(day_columns) or not calendar["d"].is_unique:
        raise ValueError("Calendar must contain exactly one row for each sales day.")
    if calendar[["date", "wm_yr_wk"]].isna().any().any():
        raise ValueError("Calendar dates and week keys must be present.")
    calendar["day"] = calendar["d"].str[2:].astype("int16")

    price_keys = ["item_id", "wm_yr_wk"]
    for store, store_sales in sales.groupby("store_id", observed=True, sort=False):
        store_prices = prices.loc[prices["store_id"].eq(store), price_keys + ["sell_price"]]
        if store_prices.duplicated(price_keys).any():
            raise ValueError(f"Duplicate item/week price keys for {store}.")
        for start in range(0, len(store_sales), batch_series):
            wide = store_sales.iloc[start:start + batch_series]
            long = wide.melt(
                id_vars=SALES_IDS, value_vars=day_columns, var_name="d", value_name="sales"
            )
            long = long.merge(calendar, on="d", how="left")
            long = long.merge(store_prices, on=price_keys, how="left")
            long["price_missing"] = long["sell_price"].isna()
            yield long


def write_long_parquet(sales, calendar, prices, output: Path, *, batch_series=256):
    """Write a complete Parquet file before replacing the destination."""
    day_columns = [column for column in sales if column.startswith("d_")]
    if batch_series < 1 or sales.empty or not day_columns:
        raise ValueError("Expected at least one series, day and series per batch.")
    if sales["id"].isna().any() or not sales["id"].is_unique:
        raise ValueError("Sales series IDs must be present and unique.")
    if sales[["store_id", "item_id"]].isna().any().any():
        raise ValueError("Sales store and item keys must be present.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".parquet.tmp", delete=False) as handle:
        temporary = Path(handle.name)

    expected_rows = len(sales) * len(day_columns)
    missing_prices = 0
    batches = long_batches(sales, calendar, prices, day_columns, batch_series)
    try:
        first = next(batches)
        table = pa.Table.from_pandas(first, preserve_index=False)
        missing_prices += int(first["price_missing"].sum())
        with pq.ParquetWriter(temporary, table.schema, compression="zstd") as writer:
            writer.write_table(table)
            del first, table
            for batch in batches:
                missing_prices += int(batch["price_missing"].sum())
                writer.write_table(pa.Table.from_pandas(batch, schema=writer.schema, preserve_index=False))

        with pq.ParquetFile(temporary) as parquet:
            if parquet.metadata.num_rows != expected_rows:
                raise ValueError(f"Expected {expected_rows:,} rows after joining the data.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return expected_rows, missing_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-series", type=int, default=256)
    parser.add_argument("--max-series", type=int, help="Limit the series for a smoke check.")
    args = parser.parse_args()
    if args.batch_series < 1 or (args.max_series is not None and args.max_series < 1):
        parser.error("Series counts must be positive.")

    filename = "m5_long_smoke.parquet" if args.max_series is not None else "m5_long.parquet"
    output = args.output or PROJECT_ROOT / "data" / "processed" / filename
    raw = load_raw_data(args.data_dir, table_names=("sales_train_evaluation", "calendar", "sell_prices"))
    sales = raw["sales_train_evaluation"].iloc[:args.max_series]
    rows, missing_prices = write_long_parquet(
        sales, raw["calendar"], raw["sell_prices"], output, batch_series=args.batch_series
    )
    print(f"Series: {len(sales):,} | Rows: {rows:,} | Missing prices: {missing_prices:,}")
    print(f"Parquet: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
