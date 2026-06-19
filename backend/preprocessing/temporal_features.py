"""
Layer 2 — Temporal Features

IMPORTANT — read before changing rush-hour logic:

created_datetime is confirmed genuine UTC (full-dataset hour histogram
peaks at UTC hour 5, which converts to ~05:30-11:30 IST). That converted
window does NOT match conventional traffic rush hour (7-10AM / 5-9PM) —
it looks far more like a data-entry/processing shift pattern than the
actual time violations occurred. Evidence: counts collapse to near-zero
from 15:30-23:30 IST (bottoming at 16 records around 19:30 IST), which
is exactly inside the evening rush window the original build guide
assumed. Using the conventional definition here would make is_rush_hour
*anti*-correlated with violation volume in this dataset.

Per project decision: is_rush_hour is defined from the OBSERVED peak
window (05:30-11:30 IST), not the textbook definition. If this dataset
is ever swapped for one with real violation-occurrence timestamps, this
window must be re-derived — don't carry it over blindly.
"""

import polars as pl

IST = "Asia/Kolkata"

RUSH_START_MIN = 5 * 60 + 30  # 05:30 IST — observed peak start
RUSH_END_MIN = 11 * 60 + 30  # 11:30 IST — observed peak end

# Bengaluru / Karnataka major holidays — hardcoded, only covers years
# present in this dataset (2023-2024). Extend before reusing on newer data.
BENGALURU_HOLIDAYS = [
    "2023-01-26",  # Republic Day
    "2023-03-22",  # Ugadi
    "2023-08-15",  # Independence Day
    "2023-10-24",  # Dussehra
    "2023-11-01",  # Kannada Rajyotsava
    "2023-11-12",  # Diwali
    "2024-01-26",  # Republic Day
    "2024-04-09",  # Ugadi
    "2024-08-15",  # Independence Day
    "2024-10-12",  # Dussehra
    "2024-11-01",  # Kannada Rajyotsava
    "2024-11-01",  # Diwali (placeholder — confirm exact 2024 date if used for modeling)
]


def add_temporal_features(df: pl.DataFrame, ts_col: str = "created_datetime") -> pl.DataFrame:
    ist = pl.col(ts_col).dt.convert_time_zone(IST)
    holiday_dates = pl.Series(sorted(set(BENGALURU_HOLIDAYS))).str.to_date()

    df = df.with_columns(
        ist.alias("created_datetime_ist"),
        ist.dt.hour().alias("hour"),
        ist.dt.weekday().alias("day_of_week"),  # ISO: Mon=1 ... Sun=7
        ist.dt.month().alias("month"),
    )

    minute_of_day = (
    pl.col("hour").cast(pl.Int32) * 60
    + pl.col("created_datetime_ist").dt.minute().cast(pl.Int32)
    )

    df = df.with_columns(
        (pl.col("day_of_week") >= 6).alias("is_weekend"),
        minute_of_day.is_between(RUSH_START_MIN, RUSH_END_MIN).alias("is_rush_hour"),
        pl.col("created_datetime_ist").dt.date().is_in(holiday_dates).alias("is_holiday"),
    )

    df = df.with_columns(
        pl.when(pl.col("hour").is_between(5, 11))
        .then(pl.lit("Morning"))
        .when(pl.col("hour").is_between(12, 16))
        .then(pl.lit("Afternoon"))
        .when(pl.col("hour").is_between(17, 20))
        .then(pl.lit("Evening"))
        .otherwise(pl.lit("Night"))
        .alias("shift")
    )

    return df


if __name__ == "__main__":
    IN_PATH = "data/processed/violations_geosnapped.parquet"
    OUT_PATH = "data/processed/violations_temporal.parquet"

    print(f"Loading {IN_PATH} ...")
    df = pl.read_parquet(IN_PATH)
    df = add_temporal_features(df)
    df.write_parquet(OUT_PATH)
    print(f"✅ Wrote {OUT_PATH} ({df.height} rows)")
    print(df.group_by("shift").len().sort("shift"))
    print(f"Rush-hour rows: {df['is_rush_hour'].sum()} / {df.height}")
    print(f"Holiday rows: {df['is_holiday'].sum()} / {df.height}")