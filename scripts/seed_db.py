"""
scripts/seed_db.py

Reads data/processed/violations_clean.parquet and bulk-inserts into the
`violations` Postgres table (created by infra/init.sql) using
psycopg2.extras.execute_values in batches of 10,000 for speed.

Usage:
    python scripts/seed_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import psycopg2
from psycopg2.extras import execute_values

from backend.core.config import settings

BATCH_SIZE = 10_000

# Column order must match the INSERT statement below
COLUMNS = [
    "id", "latitude", "longitude", "location", "vehicle_number", "vehicle_type",
    "violation_type", "offence_code", "created_datetime", "closed_datetime",
    "modified_datetime", "device_id", "created_by_id", "center_code",
    "police_station", "data_sent_to_scita", "junction_name",
    "action_taken_timestamp", "data_sent_to_scita_timestamp",
    "updated_vehicle_number", "updated_vehicle_type", "validation_status",
    "validation_timestamp",
]

INSERT_SQL = f"""
    INSERT INTO violations (
        {", ".join(COLUMNS)}, geom
    ) VALUES %s
    ON CONFLICT (id) DO NOTHING
"""

VALUE_TEMPLATE = (
    "(" + ", ".join(["%s"] * len(COLUMNS)) +
    ", ST_SetSRID(ST_MakePoint(%s, %s), 4326))"
)


def row_to_values(row: dict) -> tuple:
    base = tuple(row[c] for c in COLUMNS)
    # geom built from (longitude, latitude)
    return base + (row["longitude"], row["latitude"])


def main() -> int:
    clean_path = Path(settings.processed_dir) / "violations_clean.parquet"
    if not clean_path.exists():
        print(f"❌ {clean_path} not found. Run batch_loader.py first.")
        return 1

    df = pl.read_parquet(clean_path)
    rows = df.select(COLUMNS).to_dicts()
    total = len(rows)
    print(f"Seeding {total} rows into `violations`...")

    conn = psycopg2.connect(settings.database_url.replace("postgresql+psycopg2", "postgresql"))
    conn.autocommit = False
    inserted = 0
    try:
        with conn.cursor() as cur:
            for i in range(0, total, BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                values = [row_to_values(r) for r in batch]
                execute_values(cur, INSERT_SQL, values, template=VALUE_TEMPLATE, page_size=BATCH_SIZE)
                inserted += len(batch)
                print(f"  ...{inserted}/{total}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"✅ Seed complete: {inserted} rows committed to violations table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
