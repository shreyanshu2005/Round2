"""
backend/ingestion/validator.py

Pandera schema validation for the violations dataset, matched to the ACTUAL
production CSV columns (NOT the originally-assumed violation_id/timestamp/
junction_id/fine_amount/officer_id schema — that schema does not exist in
the real data; see CONTEXT HANDOFF notes).

Real columns (298,450 rows sampled):
id, latitude, longitude, location, vehicle_number, vehicle_type, description,
violation_type, offence_code, created_datetime, closed_datetime,
modified_datetime, device_id, created_by_id, center_code, police_station,
data_sent_to_scita, junction_name, action_taken_timestamp,
data_sent_to_scita_timestamp, updated_vehicle_number, updated_vehicle_type,
validation_status, validation_timestamp

Usage:
    python backend/ingestion/validator.py
Validates data/processed/violations_clean.parquet and exits non-zero on
schema failure.
"""
from __future__ import annotations

import sys

import pandera.polars as pa
import polars as pl
from pandera.polars import DataFrameSchema, Column
from pandera import Check

from backend.core.config import settings

# Bengaluru bounding box (degrees)
LAT_MIN, LAT_MAX = settings.bbox_lat_min, settings.bbox_lat_max
LNG_MIN, LNG_MAX = settings.bbox_lng_min, settings.bbox_lng_max

VALID_VALIDATION_STATUSES = {"approved", "rejected", "created1", None}

violations_schema = DataFrameSchema(
    {
        "id": Column(pl.Utf8, Check.str_startswith("FKID"), nullable=False, unique=True),
        "latitude": Column(pl.Float64, Check.in_range(LAT_MIN, LAT_MAX), nullable=False),
        "longitude": Column(pl.Float64, Check.in_range(LNG_MIN, LNG_MAX), nullable=False),
        "location": Column(pl.Utf8, nullable=True),
        "vehicle_number": Column(pl.Utf8, nullable=False),
        "vehicle_type": Column(pl.Utf8, nullable=False),
        "violation_type": Column(pl.Utf8, nullable=False),   # JSON-array-as-string
        "offence_code": Column(pl.Utf8, nullable=False),      # JSON-array-as-string
        "created_datetime": Column(pl.Datetime(time_zone="UTC"), nullable=False),
        "closed_datetime": Column(pl.Datetime(time_zone="UTC"), nullable=True),
        "modified_datetime": Column(pl.Datetime(time_zone="UTC"), nullable=True),
        "device_id": Column(pl.Utf8, nullable=False),
        "created_by_id": Column(pl.Utf8, nullable=True),
        "center_code": Column(pl.Float64, nullable=True),
        "police_station": Column(pl.Utf8, nullable=True),
        "data_sent_to_scita": Column(pl.Boolean, nullable=True),
        "junction_name": Column(pl.Utf8, nullable=True),
        "action_taken_timestamp": Column(pl.Datetime(time_zone="UTC"), nullable=True),
        "data_sent_to_scita_timestamp": Column(pl.Datetime(time_zone="UTC"), nullable=True),
        "updated_vehicle_number": Column(pl.Utf8, nullable=True),
        "updated_vehicle_type": Column(pl.Utf8, nullable=True),
        "validation_status": Column(pl.Utf8, nullable=True),
        "validation_timestamp": Column(pl.Datetime(time_zone="UTC"), nullable=True),
    },
    strict=False,  # allow extra derived columns (e.g. ingestion adds none yet, but be permissive)
    coerce=False,
)


def validate(df: pl.DataFrame) -> pl.DataFrame:
    """Validates df against violations_schema. Raises pandera.errors.SchemaError on failure."""
    return violations_schema.validate(df)


def main() -> int:
    clean_path = f"{settings.processed_dir}/violations_clean.parquet"
    try:
        df = pl.read_parquet(clean_path)
    except FileNotFoundError:
        print(f"❌ {clean_path} not found. Run batch_loader.py first.")
        return 1

    try:
        validate(df)
    except pa.errors.SchemaError as e:
        print("❌ Schema validation FAILED")
        print(e)
        return 1

    print(f"✅ Schema validation passed — {df.height} rows, {df.width} columns, 0 errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
