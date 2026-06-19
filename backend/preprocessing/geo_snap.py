"""
Layer 2 — Geo-Snap

  - Rows with a real "BTP### - Name" junction_name get their own code
    directly: junction_id_snapped = the BTP code, snap_distance_m = 0.
    No KD-tree needed — these are already ground truth.
  - All other rows ("No Junction" + the handful of null junction_name
    rows — ~147,717 / 298,282 on the full dataset) are snapped via
    KD-tree, matched in UTM 43N meters (not raw lat/lng) against the
    junction centroids from junction_metadata_builder.py. Snaps beyond
    SNAP_REJECT_M are rejected (junction_id_snapped set to null) rather
    than forced to a possibly-wrong nearest junction.

Run standalone:
    python backend/preprocessing/geo_snap.py
Or call geo_snap(df, meta) directly from build_feature_store.py.
"""

import polars as pl
from pyproj import Transformer
from scipy.spatial import KDTree

CLEAN_PATH = "data/processed/violations_clean.parquet"
META_PATH = "data/processed/junction_metadata.parquet"
OUT_PATH = "data/processed/violations_geosnapped.parquet"

SNAP_REJECT_M = 500.0
JUNCTION_RE = r"^(BTP\d{3}) - (.+)$"

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


def geo_snap(df: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    named_mask = (
        df["junction_name"].is_not_null()
        & (df["junction_name"] != "No Junction")
        & df["junction_name"].str.contains(JUNCTION_RE)
    )

    direct = df.filter(named_mask).with_columns(
        pl.col("junction_name").str.extract(JUNCTION_RE, 1).alias("junction_id_snapped"),
        pl.lit(0.0).alias("snap_distance_m"),
    )

    needs_snap = df.filter(~named_mask)

    if needs_snap.height > 0:
        lats = needs_snap["latitude"].to_list()
        lngs = needs_snap["longitude"].to_list()
        x, y = _to_utm.transform(lngs, lats)

        tree = KDTree(
            list(zip(meta["centroid_utm_x"].to_list(), meta["centroid_utm_y"].to_list()))
        )
        dist, idx = tree.query(list(zip(x, y)))

        junction_ids = meta["junction_id"].to_list()
        snapped_ids = [junction_ids[i] for i in idx]

        needs_snap = needs_snap.with_columns(
            pl.Series("junction_id_snapped", snapped_ids),
            pl.Series("snap_distance_m", dist),
        ).with_columns(
            pl.when(pl.col("snap_distance_m") > SNAP_REJECT_M)
            .then(None)
            .otherwise(pl.col("junction_id_snapped"))
            .alias("junction_id_snapped")
        )
    else:
        needs_snap = needs_snap.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("junction_id_snapped"),
            pl.lit(None, dtype=pl.Float64).alias("snap_distance_m"),
        )

    result = pl.concat([direct, needs_snap], how="vertical_relaxed")

    n_snapped = result["junction_id_snapped"].is_not_null().sum()
    coverage = n_snapped / result.height
    print(f"Snap coverage: {coverage:.1%} ({n_snapped}/{result.height})")

    return result


if __name__ == "__main__":
    print(f"Loading {CLEAN_PATH} and {META_PATH} ...")
    df = pl.read_parquet(CLEAN_PATH)
    meta = pl.read_parquet(META_PATH)
    result = geo_snap(df, meta)
    result.write_parquet(OUT_PATH)
    print(f"✅ Wrote {OUT_PATH} ({result.height} rows)")