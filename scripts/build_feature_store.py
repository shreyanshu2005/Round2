"""
Layer 2 — Orchestrator

Runs the full preprocessing pipeline in memory and writes
data/processed/feature_store.parquet:

    violations_clean.parquet
        -> junction_metadata (built once, cached)
        -> geo_snap
        -> temporal_features
        -> deduplication (flag only)
        -> feature_engineering
        -> feature_store.parquet

junction_metadata.parquet is built automatically on first run if it
doesn't exist yet; subsequent runs reuse the cached version. Delete it
manually if you need to rebuild (e.g. after the raw data changes).

Run:
    python scripts/build_feature_store.py
"""

import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.preprocessing.deduplication import flag_duplicates
from backend.preprocessing.feature_engineering import build_features
from backend.preprocessing.geo_snap import geo_snap
from backend.preprocessing.junction_metadata_builder import build_junction_metadata
from backend.preprocessing.temporal_features import add_temporal_features

CLEAN_PATH = "data/processed/violations_clean.parquet"
META_PATH = "data/processed/junction_metadata.parquet"
OUT_PATH = "data/processed/feature_store.parquet"


def main():
    print(f"Loading {CLEAN_PATH} ...")
    df = pl.read_parquet(CLEAN_PATH)

    if os.path.exists(META_PATH):
        print(f"Loading cached {META_PATH} ...")
        meta = pl.read_parquet(META_PATH)
    else:
        print("No cached junction metadata found — building from scratch ...")
        meta = build_junction_metadata(df)
        meta.write_parquet(META_PATH)
        print(f"✅ Wrote {META_PATH} ({meta.height} junctions)")

    print("Running geo-snap ...")
    df = geo_snap(df, meta)

    print("Adding temporal features ...")
    df = add_temporal_features(df)

    print("Flagging duplicates ...")
    df = flag_duplicates(df)

    print("Running feature engineering ...")
    df = build_features(df, meta)

    df.write_parquet(OUT_PATH)
    print(f"✅ Wrote {OUT_PATH} ({df.height} rows, {len(df.columns)} cols)")

    snap_cov = df["junction_id_snapped"].is_not_null().sum() / df.height
    print(f"Snap coverage: {snap_cov:.1%}")
    print(f"Rush-hour rows: {df['is_rush_hour'].sum()} / {df.height}")
    print(f"Duplicate-flagged rows: {df['is_duplicate'].sum()} / {df.height}")
    print(f"Multi-offence rows (n_offences > 1): {(df['n_offences'] > 1).sum()} / {df.height}")


if __name__ == "__main__":
    main()