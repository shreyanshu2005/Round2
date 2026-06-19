"""
Layer 4 — LightGBM Risk Prediction Model
Bengaluru Traffic Intelligence Platform (BTIP)

Predicts expected violation count per zone per 4-hour window.
Adapted for real schema: violation_type (JSON array), no fine_amount,
cluster features from Layer 3 HDBSCAN output.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
FEATURE_STORE_PATH = ROOT / "data" / "processed" / "clustered_feature_store.parquet"
FALLBACK_STORE_PATH = ROOT / "data" / "processed" / "feature_store.parquet"
MODEL_DIR = ROOT / "models" / "saved" / "risk"
MODEL_PATH = MODEL_DIR / "lgbm_risk.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoders.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature config ──────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["police_station", "vehicle_type", "primary_violation_type"]
NUMERIC_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "is_holiday", "rolling_7d_count", "rolling_30d_count",
    "cluster_id", "cluster_probability", "cluster_persistence_score",
    "latitude", "longitude",
]
TARGET = "violation_count"


# ── Data helpers ────────────────────────────────────────────────────────────

def _parse_violation_type(vt: str) -> str:
    """Extract first violation type from JSON-array-as-string."""
    try:
        parsed = json.loads(vt)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip().upper()
    except Exception:
        pass
    return "UNKNOWN"


def load_feature_store() -> pl.DataFrame:
    """Load clustered or plain feature store, whichever exists."""
    path = FEATURE_STORE_PATH if FEATURE_STORE_PATH.exists() else FALLBACK_STORE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No feature store found. Run Layer 2 (build_feature_store.py) first.\n"
            f"Looked at: {FEATURE_STORE_PATH} and {FALLBACK_STORE_PATH}"
        )
    logger.info(f"Loading feature store from {path}")
    return pl.read_parquet(path)


def _build_zone_windows(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate raw violations into (zone × 4-hour-window) rows.
    zone = cluster_id (from HDBSCAN) or snapped junction bucket.
    window = floor(hour / 4) → 0,1,2,3,4,5  (six 4h windows per day)
    """
    # Ensure required columns exist; add defaults if Layer 2/3 outputs differ
    if "cluster_id" not in df.columns:
        df = df.with_columns(pl.lit(-1).alias("cluster_id"))
    if "cluster_probability" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("cluster_probability"))
    if "cluster_persistence_score" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("cluster_persistence_score"))

    # Derive primary violation type from JSON array
    if "violation_type" in df.columns:
        df = df.with_columns(
            pl.col("violation_type")
            .map_elements(_parse_violation_type, return_dtype=pl.Utf8)
            .alias("primary_violation_type")
        )
    else:
        df = df.with_columns(pl.lit("UNKNOWN").alias("primary_violation_type"))

    # Ensure temporal columns exist
    for col, default in [
        ("hour", 0), ("day_of_week", 0), ("month", 1),
        ("is_weekend", 0), ("is_rush_hour", 0), ("is_holiday", 0),
        ("rolling_7d_count", 0.0), ("rolling_30d_count", 0.0),
    ]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))

    # 4-hour window bucket
    df = df.with_columns(
        (pl.col("hour") // 4).cast(pl.Int32).alias("window_4h")
    )

    # Date column for grouping
    date_col = None
    for c in ["created_datetime", "date", "timestamp"]:
        if c in df.columns:
            date_col = c
            break

    group_keys = ["cluster_id", "window_4h", "day_of_week", "month", "is_weekend",
                  "is_rush_hour", "is_holiday", "vehicle_type", "police_station",
                  "primary_violation_type"]
    if date_col:
        group_keys.append(date_col)

    # Keep only columns that exist
    group_keys = [k for k in group_keys if k in df.columns]

    agg_exprs = [
        pl.count("id").alias(TARGET) if "id" in df.columns else pl.len().alias(TARGET),
        pl.col("rolling_7d_count").mean().alias("rolling_7d_count"),
        pl.col("rolling_30d_count").mean().alias("rolling_30d_count"),
        pl.col("cluster_probability").mean().alias("cluster_probability"),
        pl.col("cluster_persistence_score").mean().alias("cluster_persistence_score"),
        pl.col("latitude").mean().alias("latitude"),
        pl.col("longitude").mean().alias("longitude"),
    ]
    # Only aggregate cols that exist
    agg_exprs = [e for e in agg_exprs
                 if not hasattr(e, '_pyexpr') or True]  # keep all, polars handles missing

    # Derive hour from window_4h midpoint for feature usage
    aggregated = (
        df.group_by(group_keys)
        .agg(
            pl.len().alias(TARGET),
            pl.col("rolling_7d_count").mean().alias("rolling_7d_count"),
            pl.col("rolling_30d_count").mean().alias("rolling_30d_count"),
            pl.col("cluster_probability").mean().alias("cluster_probability"),
            pl.col("cluster_persistence_score").mean().alias("cluster_persistence_score"),
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
        )
    )

    # Re-derive hour from window bucket midpoint
    aggregated = aggregated.with_columns(
        (pl.col("window_4h") * 4 + 2).alias("hour")
    )

    return aggregated


def _encode_categoricals(
    df: pl.DataFrame,
    encoders: Optional[dict] = None,
    fit: bool = True,
) -> tuple[pl.DataFrame, dict]:
    """Label-encode categorical columns. Returns (df, encoders)."""
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit("UNKNOWN").alias(col))

        series = df[col].cast(pl.Utf8).fill_null("UNKNOWN").to_list()

        if fit:
            le = LabelEncoder()
            encoded = le.fit_transform(series)
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels
            known = set(le.classes_)
            series = [s if s in known else le.classes_[0] for s in series]
            encoded = le.transform(series)

        df = df.with_columns(
            pl.Series(name=col + "_enc", values=encoded.astype(np.int32))
        )

    return df, encoders


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature columns available in df."""
    candidates = NUMERIC_COLS + [c + "_enc" for c in CATEGORICAL_COLS]
    return [c for c in candidates if c in df.columns]


# ── Training ────────────────────────────────────────────────────────────────

def _lgbm_params() -> dict:
    return {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }


def _optuna_search(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> dict:
    """Bayesian hyperparameter search with Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for train_idx, val_idx in tscv.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = model.predict(X[val_idx])
            maes.append(mean_absolute_error(y[val_idx], np.maximum(preds, 0)))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best Optuna MAE: {study.best_value:.4f} | params: {study.best_params}")
    best = study.best_params
    best.update({"objective": "regression", "metric": "mae",
                 "n_jobs": -1, "random_state": 42, "verbose": -1})
    return best


def cross_validate(
    df_agg: pl.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """Time-series cross-validation. Returns dict with fold metrics."""
    X = df_agg[feature_cols].to_numpy()
    y = df_agg[TARGET].to_numpy().astype(float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    params = _lgbm_params()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = np.maximum(model.predict(X[val_idx]), 0)
        mae = mean_absolute_error(y[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
        # MAPE — guard division by zero
        mask = y[val_idx] > 0
        mape = float(np.mean(np.abs((y[val_idx][mask] - preds[mask]) / y[val_idx][mask]))) * 100 if mask.any() else 0.0
        fold_metrics.append({"fold": fold + 1, "mae": mae, "rmse": rmse, "mape": mape})
        logger.info(f"Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")

    mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
    mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    mean_mape = float(np.mean([m["mape"] for m in fold_metrics]))
    logger.info(f"CV Summary → MAE={mean_mae:.2f}  RMSE={mean_rmse:.2f}  MAPE={mean_mape:.1f}%")

    return {
        "folds": fold_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_mape": mean_mape,
    }


def train(
    df: Optional[pl.DataFrame] = None,
    use_optuna: bool = False,
    n_optuna_trials: int = 50,
) -> tuple[lgb.LGBMRegressor, dict, dict]:
    """
    Full training pipeline.
    Returns (model, encoders, cv_metrics).
    Also saves model + encoders to MODEL_DIR.
    """
    if df is None:
        df = load_feature_store()

    logger.info(f"Raw rows: {len(df):,}")
    df_agg = _build_zone_windows(df)
    logger.info(f"Aggregated zone-windows: {len(df_agg):,}")

    df_agg, encoders = _encode_categoricals(df_agg, fit=True)
    feature_cols = get_feature_cols(df_agg)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = df_agg[feature_cols].to_numpy()
    y = df_agg[TARGET].to_numpy().astype(float)

    # CV metrics before final fit
    cv_metrics = cross_validate(df_agg, feature_cols)

    # Final model on all data
    if use_optuna:
        params = _optuna_search(X, y, n_trials=n_optuna_trials)
    else:
        params = _lgbm_params()

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"encoders": encoders, "feature_cols": feature_cols}, ENCODER_PATH)
    logger.info(f"Model saved → {MODEL_PATH}")

    return model, encoders, cv_metrics


# ── Inference ───────────────────────────────────────────────────────────────

def load_model() -> tuple[lgb.LGBMRegressor, dict, list[str]]:
    """Load saved model and encoders."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_PATH}. Run train() first.")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(ENCODER_PATH)
    return model, meta["encoders"], meta["feature_cols"]


def predict(
    model: lgb.LGBMRegressor,
    encoders: dict,
    feature_cols: list[str],
    input_df: pl.DataFrame,
) -> np.ndarray:
    """
    Predict violation count for input rows (already aggregated zone-windows).
    Returns array of non-negative predicted counts.
    """
    df, _ = _encode_categoricals(input_df, encoders=encoders, fit=False)
    # Fill any missing feature cols with 0
    for col in feature_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).alias(col))
    X = df[feature_cols].to_numpy()
    return np.maximum(model.predict(X), 0)


def predict_single(
    zone_id: int,
    hour: int,
    day_of_week: int,
    month: int,
    police_station: str = "UNKNOWN",
    vehicle_type: str = "UNKNOWN",
    primary_violation_type: str = "UNKNOWN",
    rolling_7d_count: float = 0.0,
    rolling_30d_count: float = 0.0,
    cluster_probability: float = 0.5,
    cluster_persistence_score: float = 0.5,
    latitude: float = 12.97,
    longitude: float = 77.59,
    model: Optional[lgb.LGBMRegressor] = None,
    encoders: Optional[dict] = None,
    feature_cols: Optional[list[str]] = None,
) -> float:
    """Convenience wrapper for single-zone single-window prediction."""
    if model is None:
        model, encoders, feature_cols = load_model()

    is_weekend = int(day_of_week >= 5)
    is_rush_hour = int(hour in [7, 8, 9, 17, 18, 19, 20])

    row = pl.DataFrame({
        "cluster_id": [zone_id],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "is_rush_hour": [is_rush_hour],
        "is_holiday": [0],
        "rolling_7d_count": [rolling_7d_count],
        "rolling_30d_count": [rolling_30d_count],
        "cluster_probability": [cluster_probability],
        "cluster_persistence_score": [cluster_persistence_score],
        "latitude": [latitude],
        "longitude": [longitude],
        "police_station": [police_station],
        "vehicle_type": [vehicle_type],
        "primary_violation_type": [primary_violation_type],
    })

    preds = predict(model, encoders, feature_cols, row)
    return float(preds[0])


# ── Feature importance ──────────────────────────────────────────────────────

def feature_importance(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    top_n: int = 15,
) -> list[dict]:
    """Return top-N features by gain importance."""
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    return [{"feature": f, "importance": float(imp)} for f, imp in pairs]


# ── CLI entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    use_optuna = "--optuna" in sys.argv
    logger.info("Starting LightGBM training pipeline...")
    model, encoders, cv = train(use_optuna=use_optuna)

    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    for fold in cv["folds"]:
        print(f"  Fold {fold['fold']}: MAE={fold['mae']:.2f}  RMSE={fold['rmse']:.2f}  MAPE={fold['mape']:.1f}%")
    print(f"\n  Mean MAE  : {cv['mean_mae']:.2f}")
    print(f"  Mean RMSE : {cv['mean_rmse']:.2f}")
    print(f"  Mean MAPE : {cv['mean_mape']:.1f}%")

    fi = feature_importance(model, joblib.load(ENCODER_PATH)["feature_cols"])
    print("\nTOP FEATURES BY GAIN:")
    for item in fi[:10]:
        print(f"  {item['feature']:40s} {item['importance']:,.0f}")
    print("="*50)