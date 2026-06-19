"""
HDBSCAN hotspot clustering model.

Reads from data/processed/feature_store.parquet (real schema).
Projects lat/lng → UTM Zone 43N (EPSG:32643) before clustering.
Outputs: cluster_id, cluster_probability per row.
"""

import os
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import hdbscan
from pyproj import Transformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
FEATURE_STORE = BASE_DIR / "data" / "processed" / "feature_store.parquet"
MODEL_DIR = BASE_DIR / "models" / "saved" / "clustering"
MODEL_PATH = MODEL_DIR / "hdbscan_model.joblib"

# UTM Zone 43N (Bengaluru)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


def latlon_to_utm(lat: np.ndarray, lon: np.ndarray):
    """Project lat/lon arrays to UTM Zone 43N (metres)."""
    utm_x, utm_y = _TRANSFORMER.transform(lon, lat)
    return utm_x, utm_y


def load_feature_store(path: Path = FEATURE_STORE) -> pl.DataFrame:
    df = pl.read_parquet(path)
    # Drop rows without valid coordinates
    df = df.filter(
        pl.col("latitude").is_not_null()
        & pl.col("longitude").is_not_null()
        & (pl.col("latitude").is_between(12.8, 13.2))
        & (pl.col("longitude").is_between(77.4, 77.8))
    )
    return df


def build_coords(df: pl.DataFrame) -> np.ndarray:
    """Return (N, 2) UTM coordinate array."""
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    utm_x, utm_y = latlon_to_utm(lat, lon)
    return np.column_stack([utm_x, utm_y])


def train(
    df: pl.DataFrame | None = None,
    min_cluster_size: int = 50,
    min_samples: int = 5,
    save: bool = True,
) -> tuple[hdbscan.HDBSCAN, pl.DataFrame]:
    """
    Train HDBSCAN on the feature store.
    Returns (fitted model, df with cluster_id and cluster_probability columns).
    """
    if df is None:
        logger.info("Loading feature store …")
        df = load_feature_store()

    logger.info(f"Clustering {len(df):,} rows …")
    coords = build_coords(df)

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,       # needed for approximate_predict later
    )
    model.fit(coords)

    labels = model.labels_                          # -1 = noise
    probs = model.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).sum() / len(labels)
    logger.info(f"Clusters found: {n_clusters}  |  Noise ratio: {noise_ratio:.1%}")

    df = df.with_columns([
        pl.Series("cluster_id", labels.astype(int)),
        pl.Series("cluster_probability", probs.astype(float)),
    ])

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved → {MODEL_PATH}")

    return model, df


def load_model() -> hdbscan.HDBSCAN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run train() first.")
    return joblib.load(MODEL_PATH)


def predict(coords_utm: np.ndarray, model: hdbscan.HDBSCAN | None = None):
    """
    Assign new points (UTM coords, shape N×2) to existing clusters.
    Returns (cluster_ids, probabilities).
    """
    if model is None:
        model = load_model()
    labels, probs = hdbscan.approximate_predict(model, coords_utm)
    return labels, probs


def cluster_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Return per-cluster summary: centroid, violation count, top offence types.
    Excludes noise (cluster_id == -1).
    """
    # Explode violation_type if it exists as a list column
    work = df.filter(pl.col("cluster_id") >= 0)

    summary = work.group_by("cluster_id").agg([
        pl.col("latitude").mean().alias("centroid_lat"),
        pl.col("longitude").mean().alias("centroid_lng"),
        pl.count("id").alias("violation_count"),
        pl.col("cluster_probability").mean().alias("mean_cluster_probability"),
    ]).sort("violation_count", descending=True)

    return summary


if __name__ == "__main__":
    model, df_with_clusters = train()
    summary = cluster_summary(df_with_clusters)
    print(summary.head(15))