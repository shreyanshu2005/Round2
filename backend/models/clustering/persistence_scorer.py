"""
Cluster Persistence Scorer.

Splits feature_store into monthly slices, runs HDBSCAN on each,
then measures how consistently each spatial zone remains a hotspot
across months. Score 0-1:
  > 0.7  → structural hotspot (chronic infra problem)
  < 0.3  → transient hotspot (event-driven, one-time patrol)

Strategy
--------
Rather than trying to match cluster IDs across months (they change), we
work on a 50m-radius grid cell basis:
  1. Quantise UTM coords to 50m grid cells.
  2. For each month, collect which grid cells are inside any HDBSCAN cluster.
  3. persistence_score(cell) = fraction of months cell was in a cluster.
  4. For the final model's clusters, persistence_score(cluster) =
     mean persistence score of its core grid cells.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import hdbscan
import joblib
from pyproj import Transformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURE_STORE = BASE_DIR / "data" / "processed" / "feature_store.parquet"
PERSISTENCE_PATH = BASE_DIR / "data" / "processed" / "cluster_persistence.parquet"

_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
GRID_SIZE_M = 50      # 50-metre grid cells
MIN_CLUSTER_SIZE = 30  # smaller than main model — monthly slices are smaller


def _latlon_to_utm(lat: np.ndarray, lon: np.ndarray):
    utm_x, utm_y = _TRANSFORMER.transform(lon, lat)
    return utm_x, utm_y


def _to_grid_cell(utm_x: np.ndarray, utm_y: np.ndarray, cell_size: int = GRID_SIZE_M):
    """Quantise UTM coords to grid cell (ix, iy) tuple."""
    ix = (utm_x // cell_size).astype(int)
    iy = (utm_y // cell_size).astype(int)
    return ix, iy


def _clustered_cells_for_slice(df_month: pl.DataFrame) -> set:
    """Return set of (ix, iy) grid cells that are inside any HDBSCAN cluster."""
    if len(df_month) < MIN_CLUSTER_SIZE * 2:
        return set()

    lat = df_month["latitude"].to_numpy()
    lon = df_month["longitude"].to_numpy()
    utm_x, utm_y = _latlon_to_utm(lat, lon)
    coords = np.column_stack([utm_x, utm_y])

    model = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(coords)

    in_cluster = labels >= 0
    if not in_cluster.any():
        return set()

    ix, iy = _to_grid_cell(utm_x[in_cluster], utm_y[in_cluster])
    return set(zip(ix.tolist(), iy.tolist()))


def compute_persistence(
    df: pl.DataFrame | None = None,
    save: bool = True,
) -> pl.DataFrame:
    """
    Compute per-grid-cell persistence scores across all months in the data.
    Returns a DataFrame with columns: grid_ix, grid_iy, persistence_score.
    """
    if df is None:
        df = pl.read_parquet(FEATURE_STORE)

    # Need created_datetime as date for monthly grouping
    if "created_datetime" not in df.columns:
        raise ValueError("feature_store must have 'created_datetime' column")

    df = df.filter(
        pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
        & pl.col("latitude").is_between(12.8, 13.2)
        & pl.col("longitude").is_between(77.4, 77.8)
    )

    # Add year-month string for slicing
    df = df.with_columns(
        pl.col("created_datetime").dt.strftime("%Y-%m").alias("ym")
    )

    months = df["ym"].unique().sort().to_list()
    logger.info(f"Computing persistence over {len(months)} months …")

    # Count how many months each cell was active
    cell_month_count: dict[tuple, int] = {}
    for ym in months:
        slice_df = df.filter(pl.col("ym") == ym)
        cells = _clustered_cells_for_slice(slice_df)
        for cell in cells:
            cell_month_count[cell] = cell_month_count.get(cell, 0) + 1

    if not cell_month_count:
        logger.warning("No clustered cells found across any month.")
        return pl.DataFrame({"grid_ix": [], "grid_iy": [], "persistence_score": []})

    n_months = len(months)
    records = [
        {"grid_ix": k[0], "grid_iy": k[1], "persistence_score": v / n_months}
        for k, v in cell_month_count.items()
    ]
    persistence_df = pl.DataFrame(records).with_columns(
        pl.col("persistence_score").cast(pl.Float64)
    )

    if save:
        PERSISTENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        persistence_df.write_parquet(PERSISTENCE_PATH)
        logger.info(f"Persistence scores saved → {PERSISTENCE_PATH}")

    structural = (persistence_df["persistence_score"] > 0.7).sum()
    transient = (persistence_df["persistence_score"] < 0.3).sum()
    logger.info(
        f"Grid cells: {len(persistence_df):,} | "
        f"Structural (>0.7): {structural:,} | "
        f"Transient (<0.3): {transient:,}"
    )
    return persistence_df


def attach_persistence_to_clusters(
    df_with_clusters: pl.DataFrame,
    persistence_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Given the clustered dataframe (with cluster_id column), compute
    mean persistence_score per cluster_id and join back.

    Returns df_with_clusters with added 'cluster_persistence_score' column.
    """
    if persistence_df is None:
        if PERSISTENCE_PATH.exists():
            persistence_df = pl.read_parquet(PERSISTENCE_PATH)
        else:
            logger.warning("No persistence parquet found — running compute_persistence …")
            persistence_df = compute_persistence(df_with_clusters)

    # Compute grid cell for each row in clustered df
    lat = df_with_clusters["latitude"].to_numpy()
    lon = df_with_clusters["longitude"].to_numpy()
    utm_x, utm_y = _latlon_to_utm(lat, lon)
    ix, iy = _to_grid_cell(utm_x, utm_y)

    df_with_clusters = df_with_clusters.with_columns([
        pl.Series("grid_ix", ix.tolist()),
        pl.Series("grid_iy", iy.tolist()),
    ])

    # Join persistence score onto each row
    df_with_clusters = df_with_clusters.join(
        persistence_df, on=["grid_ix", "grid_iy"], how="left"
    ).with_columns(
        pl.col("persistence_score").fill_null(0.0).alias("cell_persistence_score")
    ).drop("persistence_score")

    # Aggregate per cluster → mean of cell scores
    cluster_persist = (
        df_with_clusters
        .filter(pl.col("cluster_id") >= 0)
        .group_by("cluster_id")
        .agg(pl.col("cell_persistence_score").mean().alias("cluster_persistence_score"))
    )

    df_with_clusters = df_with_clusters.join(
        cluster_persist, on="cluster_id", how="left"
    ).with_columns(
        pl.col("cluster_persistence_score").fill_null(0.0)
    )

    return df_with_clusters


def load_persistence() -> pl.DataFrame:
    if not PERSISTENCE_PATH.exists():
        raise FileNotFoundError(f"No persistence data at {PERSISTENCE_PATH}. Run compute_persistence() first.")
    return pl.read_parquet(PERSISTENCE_PATH)


if __name__ == "__main__":
    persistence_df = compute_persistence()
    print(persistence_df.sort("persistence_score", descending=True).head(20))