"""
MLflow utility helpers shared across all BTIP model layers.

Usage (clustering):
    from backend.models.registry.mlflow_utils import log_clustering_run, load_model

Environment variable MLFLOW_TRACKING_URI controls the backend.
Defaults to file-based ./mlruns (no MLflow server needed for local dev).
"""

import os
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.pyfunc

logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(Path(__file__).resolve().parents[3] / "mlruns"))
mlflow.set_tracking_uri(TRACKING_URI)


# ── Experiment helpers ────────────────────────────────────────────────────────

def get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info(f"Created MLflow experiment '{name}' (id={exp_id})")
    else:
        exp_id = exp.experiment_id
    return exp_id


# ── Clustering ────────────────────────────────────────────────────────────────

def log_clustering_run(
    model,                          # fitted hdbscan.HDBSCAN
    n_clusters: int,
    noise_ratio: float,
    mean_persistence_score: float,
    params: dict[str, Any] | None = None,
    model_path: str | None = None,
) -> str:
    """Log a clustering run and return the run_id."""
    exp_id = get_or_create_experiment("btip-clustering")
    with mlflow.start_run(experiment_id=exp_id) as run:
        # Params
        default_params = {
            "min_cluster_size": model.min_cluster_size,
            "min_samples": model.min_samples,
            "metric": model.metric,
            "cluster_selection_method": model.cluster_selection_method,
        }
        mlflow.log_params({**default_params, **(params or {})})

        # Metrics
        mlflow.log_metrics({
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "mean_persistence_score": mean_persistence_score,
        })

        # Model artifact
        mlflow.sklearn.log_model(model, artifact_path="hdbscan_model")

        run_id = run.info.run_id
        logger.info(f"MLflow clustering run logged: {run_id}")
        return run_id


# ── Risk / XGBoost / LightGBM ─────────────────────────────────────────────────

def log_risk_run(
    model,
    model_name: str,       # "lightgbm" or "xgboost"
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
) -> str:
    exp_id = get_or_create_experiment("btip-risk-prediction")
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        mlflow.log_params(params or {})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        run_id = run.info.run_id
        logger.info(f"MLflow risk run ({model_name}) logged: {run_id}")
        return run_id


# ── Generic model loader ──────────────────────────────────────────────────────

def load_model(run_id: str, artifact_path: str = "model"):
    """Load a model logged with mlflow.sklearn from a specific run."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)


def get_best_run(experiment_name: str, metric: str, ascending: bool = True) -> mlflow.entities.Run | None:
    """Return the run with the best (min or max) value of `metric`."""
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    order = "ASC" if ascending else "DESC"
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if runs.empty:
        return None
    return mlflow.get_run(runs.iloc[0]["run_id"])