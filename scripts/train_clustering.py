"""
Layer 3 training script.

Runs:
  1. HDBSCAN clustering on feature_store.parquet
  2. Persistence scoring (monthly stability)
  3. Attaches persistence scores to cluster output
  4. Writes clustered_feature_store.parquet (feature_store + cluster columns)
  5. Logs run to MLflow

Usage:
    python scripts/train_clustering.py
"""

import sys
import logging
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from backend.models.clustering.hdbscan_model import (
    load_feature_store,
    train,
    cluster_summary,
)
from backend.models.clustering.persistence_scorer import (
    compute_persistence,
    attach_persistence_to_clusters,
)
from backend.models.registry.mlflow_utils import log_clustering_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "data" / "processed" / "clustered_feature_store.parquet"


def main():
    logger.info("═" * 60)
    logger.info("BTIP Layer 3 — HDBSCAN Clustering & Persistence Scoring")
    logger.info("═" * 60)

    # 1. Train HDBSCAN
    logger.info("\n[1/4] Training HDBSCAN …")
    df = load_feature_store()
    model, df_clustered = train(df=df, min_cluster_size=50, min_samples=5, save=True)

    labels = df_clustered["cluster_id"].to_numpy()
    n_clusters = int((labels >= 0).sum())  # distinct is done below
    import numpy as np
    n_unique = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise_ratio = float((labels == -1).sum() / len(labels))
    logger.info(f"  Unique clusters: {n_unique}  |  Noise ratio: {noise_ratio:.1%}")

    # 2. Persistence scoring
    logger.info("\n[2/4] Computing monthly persistence scores …")
    persistence_df = compute_persistence(df=df, save=True)
    mean_persist = float(persistence_df["persistence_score"].mean()) if len(persistence_df) > 0 else 0.0
    logger.info(f"  Mean persistence score: {mean_persist:.3f}")

    # 3. Attach persistence to clustered df
    logger.info("\n[3/4] Attaching persistence scores to clusters …")
    df_final = attach_persistence_to_clusters(df_clustered, persistence_df)

    # 4. Save enriched feature store
    logger.info(f"\n[4/4] Saving clustered feature store → {OUT_PATH}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.write_parquet(OUT_PATH)
    logger.info(f"  Rows: {len(df_final):,}  |  Columns: {len(df_final.columns)}")

    # 5. Cluster summary
    summary = cluster_summary(df_final)
    logger.info("\nTop 10 clusters by violation count:")
    print(summary.head(10))

    # 6. Log to MLflow
    logger.info("\nLogging to MLflow …")
    try:
        run_id = log_clustering_run(
            model=model,
            n_clusters=n_unique,
            noise_ratio=noise_ratio,
            mean_persistence_score=mean_persist,
        )
        logger.info(f"  MLflow run_id: {run_id}")
    except Exception as e:
        logger.warning(f"  MLflow logging failed (non-fatal): {e}")

    # 7. Verification summary
    logger.info("\n" + "═" * 60)
    logger.info("Layer 3 COMPLETE — Verification:")
    logger.info(f"  ✓ Unique clusters:        {n_unique}")
    logger.info(f"  ✓ Noise ratio:            {noise_ratio:.1%}  (target < 40%)")
    logger.info(f"  ✓ Mean persistence score: {mean_persist:.3f}")
    structural = (persistence_df["persistence_score"] > 0.7).sum() if len(persistence_df) > 0 else 0
    logger.info(f"  ✓ Structural hotspot cells (>0.7): {structural:,}")
    logger.info(f"  ✓ Output: {OUT_PATH}")
    logger.info("═" * 60)

    if n_unique < 10:
        logger.warning("⚠ Fewer than 10 clusters found. Consider reducing min_cluster_size.")
    if noise_ratio > 0.4:
        logger.warning("⚠ Noise ratio > 40%. Consider reducing min_cluster_size or min_samples.")


if __name__ == "__main__":
    main()