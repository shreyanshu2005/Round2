"""
backend/ingestion/batch_loader.py

Reads the raw violations CSV with Polars, enforces dtypes, drops rows with
null/out-of-bounds lat-lng, and writes data/processed/violations_clean.parquet.

Usage:
    python backend/ingestion/batch_loader.py [path/to/violations.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from backend.core.config import settings

# Real timestamp columns in the raw CSV (mixed precision: some have +00:00 only,
# some have microsecond fractional seconds). Polars infers most of these from
# CSV directly as Utf8; we parse them explicitly to avoid inference flakiness.
DATETIME_COLS = [
    "created_datetime",
    "closed_datetime",
    "modified_datetime",
    "action_taken_timestamp",
    "data_sent_to_scita_timestamp",
    "validation_timestamp",
]

STRING_COLS = [
    "id",
    "location",
    "vehicle_number",
    "vehicle_type",
    "violation_type",
    "offence_code",
    "device_id",
    "created_by_id",
    "police_station",
    "junction_name",
    "updated_vehicle_number",
    "updated_vehicle_type",
    "validation_status",
]


def _parse_datetime(col: str) -> pl.Expr:
    """Parses ISO-ish timestamps with variable fractional-second precision and a
    UTC offset. Polars' auto-format detection (`strict=False` with no format)
    is unreliable across mixed-precision real data, so we try explicit formats
    in order and coalesce to the first one that matches each row:
      1. '2024-01-14 09:48:46.490255+00:00'  (fractional seconds, colon offset)
      2. '2024-01-14 09:48:46.490255+00'     (fractional seconds, bare offset)
      3. '2024-01-14 09:48:46+00:00'         (no fractional seconds, colon offset)
      4. '2024-01-14 09:48:46+00'            (no fractional seconds, bare offset)
    Any row matching none of these becomes null rather than raising, since
    several of these columns are mostly-null by design (e.g. closed_datetime).
    """
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    target = pl.Datetime(time_unit="us", time_zone="UTC")
    attempts = [
        s.str.strptime(target, format="%Y-%m-%d %H:%M:%S%.f%:z", strict=False),
        s.str.strptime(target, format="%Y-%m-%d %H:%M:%S%.f%z", strict=False),
        s.str.strptime(target, format="%Y-%m-%d %H:%M:%S%.f%#z", strict=False),
        s.str.strptime(target, format="%Y-%m-%d %H:%M:%S%:z", strict=False),
        s.str.strptime(target, format="%Y-%m-%d %H:%M:%S%z", strict=False),
        s.str.strptime(target, format="%Y-%m-%d %H:%M:%S%#z", strict=False),
    ]
    return pl.coalesce(attempts).alias(col)


def load_raw(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(
        csv_path,
        infer_schema_length=10000,
        null_values=["", "No Value", "NaN", "nan"],
        try_parse_dates=False,
    )

    # Enforce string dtypes explicitly (CSV inference sometimes guesses wrong
    # for sparsely-populated columns like description/center_code).
    for col in STRING_COLS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8))

    # Parse all timestamp-like columns
    df = df.with_columns([_parse_datetime(c) for c in DATETIME_COLS if c in df.columns])

    # data_sent_to_scita -> bool
    if "data_sent_to_scita" in df.columns:
        df = df.with_columns(pl.col("data_sent_to_scita").cast(pl.Boolean, strict=False))

    # center_code -> float (already numeric in most rows)
    if "center_code" in df.columns:
        df = df.with_columns(pl.col("center_code").cast(pl.Float64, strict=False))

    return df


def clean(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Drops rows with null/out-of-Bengaluru-bounds lat/lng or missing
    created_datetime (our hypertable partition key). Returns (clean_df, stats)."""
    n_in = df.height

    bounds_mask = (
        pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
        & pl.col("latitude").is_between(settings.bbox_lat_min, settings.bbox_lat_max)
        & pl.col("longitude").is_between(settings.bbox_lng_min, settings.bbox_lng_max)
        & pl.col("created_datetime").is_not_null()
        & pl.col("id").is_not_null()
    )

    dropped = df.filter(~bounds_mask)
    clean_df = df.filter(bounds_mask)

    null_counts = {c: int(df[c].null_count()) for c in df.columns}

    stats = {
        "rows_in": n_in,
        "rows_out": clean_df.height,
        "rows_dropped": dropped.height,
        "null_counts": null_counts,
    }
    return clean_df, stats


def main(csv_path: str | None = None) -> int:
    csv_path = csv_path or settings.raw_violations_csv
    if not Path(csv_path).exists():
        print(f"❌ Raw CSV not found at {csv_path}. Set RAW_VIOLATIONS_CSV or pass a path.")
        return 1

    print(f"Loading {csv_path} with Polars...")
    df = load_raw(csv_path)
    clean_df, stats = clean(df)

    out_dir = Path(settings.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "violations_clean.parquet"
    clean_df.write_parquet(out_path)

    print("\n--- Ingestion Summary ---")
    print(f"Rows in:      {stats['rows_in']}")
    print(f"Rows out:     {stats['rows_out']}")
    print(f"Rows dropped: {stats['rows_dropped']}")
    print("Null counts per column:")
    for col, n in stats["null_counts"].items():
        print(f"  {col}: {n}")
    print(f"\n✅ Wrote {out_path} ({clean_df.height} rows, {clean_df.width} cols)")
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(arg))
