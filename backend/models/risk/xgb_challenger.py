"""
Layer 4 — XGBoost Challenger + Ensemble Blender
Bengaluru Traffic Intelligence Platform (BTIP)

Trains XGBoost on the same feature set as LightGBM.
Final ensemble prediction = 0.6 * lgbm_pred + 0.4 * xgb_pred
(weights recalibrated from inverse CV MAE when both models are available).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from backend.models.risk.lgbm_risk import (
    _build_zone_windows,
    _encode_categoricals,
    get_feature_cols,
    load_feature_store,
    load_model as load_lgbm,
    TARGET,
)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "saved" / "risk"
XGB_MODEL_PATH = MODEL_DIR / "xgb_risk.joblib"
ENSEMBLE_META_PATH = MODEL_DIR / "ensemble_weights.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── XGBoost config ──────────────────────────────────────────────────────────

def _xgb_params() -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
    }


# ── Training ────────────────────────────────────────────────────────────────

def cross_validate_xgb(
    df_agg: pl.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """Time-series CV for XGBoost. Returns fold metrics + mean MAE."""
    X = df_agg[feature_cols].to_numpy()
    y = df_agg[TARGET].to_numpy().astype(float)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    params = _xgb_params()

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        # NEW — move early_stopping_rounds to constructor
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = np.maximum(model.predict(X[val_idx]), 0)
        mae = mean_absolute_error(y[val_idx], preds)
        rmse = np.sqrt(mean_squared_error(y[val_idx], preds))
        mask = y[val_idx] > 0
        mape = float(np.mean(np.abs((y[val_idx][mask] - preds[mask]) / y[val_idx][mask]))) * 100 if mask.any() else 0.0
        fold_metrics.append({"fold": fold + 1, "mae": mae, "rmse": rmse, "mape": mape})
        logger.info(f"[XGB] Fold {fold+1}: MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")

    mean_mae = float(np.mean([m["mae"] for m in fold_metrics]))
    mean_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
    mean_mape = float(np.mean([m["mape"] for m in fold_metrics]))
    logger.info(f"[XGB] CV → MAE={mean_mae:.2f}  RMSE={mean_rmse:.2f}  MAPE={mean_mape:.1f}%")

    return {
        "folds": fold_metrics,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_mape": mean_mape,
    }


def train(
    df: Optional[pl.DataFrame] = None,
) -> tuple[xgb.XGBRegressor, dict, dict]:
    """
    Train XGBoost challenger.
    Reuses the same aggregated windows and encoders as LightGBM.
    Returns (xgb_model, encoders, cv_metrics).
    """
    if df is None:
        df = load_feature_store()

    logger.info(f"[XGB] Raw rows: {len(df):,}")
    df_agg = _build_zone_windows(df)
    logger.info(f"[XGB] Aggregated zone-windows: {len(df_agg):,}")

    df_agg, encoders = _encode_categoricals(df_agg, fit=True)
    feature_cols = get_feature_cols(df_agg)

    X = df_agg[feature_cols].to_numpy()
    y = df_agg[TARGET].to_numpy().astype(float)

    cv_metrics = cross_validate_xgb(df_agg, feature_cols)

    params = _xgb_params()
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)

    joblib.dump(model, XGB_MODEL_PATH)
    logger.info(f"[XGB] Model saved → {XGB_MODEL_PATH}")

    return model, encoders, cv_metrics


# ── Load ────────────────────────────────────────────────────────────────────

def load_model() -> xgb.XGBRegressor:
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"No saved XGB model at {XGB_MODEL_PATH}. Run train() first.")
    return joblib.load(XGB_MODEL_PATH)


# ── Ensemble ────────────────────────────────────────────────────────────────

def compute_ensemble_weights(
    lgbm_mae: float,
    xgb_mae: float,
) -> tuple[float, float]:
    """
    Derive ensemble weights from inverse CV MAE.
    Lower MAE → higher weight.
    """
    inv_lgbm = 1.0 / max(lgbm_mae, 1e-6)
    inv_xgb = 1.0 / max(xgb_mae, 1e-6)
    total = inv_lgbm + inv_xgb
    w_lgbm = round(inv_lgbm / total, 4)
    w_xgb = round(inv_xgb / total, 4)
    logger.info(f"Ensemble weights — LightGBM: {w_lgbm:.3f}  XGBoost: {w_xgb:.3f}")
    return w_lgbm, w_xgb


def save_ensemble_weights(w_lgbm: float, w_xgb: float) -> None:
    joblib.dump({"lgbm": w_lgbm, "xgb": w_xgb}, ENSEMBLE_META_PATH)
    logger.info(f"Ensemble weights saved → {ENSEMBLE_META_PATH}")


def load_ensemble_weights() -> tuple[float, float]:
    if not ENSEMBLE_META_PATH.exists():
        logger.warning("No ensemble weights found. Using defaults (0.6 / 0.4).")
        return 0.6, 0.4
    meta = joblib.load(ENSEMBLE_META_PATH)
    return meta["lgbm"], meta["xgb"]


def ensemble_predict(
    X: np.ndarray,
    lgbm_model,
    xgb_model: xgb.XGBRegressor,
    w_lgbm: float = 0.6,
    w_xgb: float = 0.4,
) -> np.ndarray:
    """
    Weighted ensemble prediction.
    Both models must be trained on the same feature set.
    """
    lgbm_preds = np.maximum(lgbm_model.predict(X), 0)
    xgb_preds = np.maximum(xgb_model.predict(X), 0)
    blended = w_lgbm * lgbm_preds + w_xgb * xgb_preds
    return np.maximum(blended, 0)


def ensemble_predict_with_uncertainty(
    X: np.ndarray,
    lgbm_model,
    xgb_model: xgb.XGBRegressor,
    w_lgbm: float = 0.6,
    w_xgb: float = 0.4,
) -> dict:
    """
    Returns blended prediction + individual model predictions
    (spread between models used as a proxy uncertainty signal).
    """
    lgbm_preds = np.maximum(lgbm_model.predict(X), 0)
    xgb_preds = np.maximum(xgb_model.predict(X), 0)
    blended = w_lgbm * lgbm_preds + w_xgb * xgb_preds

    spread = np.abs(lgbm_preds - xgb_preds)
    # Approximate P10/P90 as blended ± 1.28 * spread (loosely normal-ish)
    p10 = np.maximum(blended - 1.28 * spread, 0)
    p90 = blended + 1.28 * spread

    return {
        "p50": blended,
        "p10": p10,
        "p90": p90,
        "lgbm": lgbm_preds,
        "xgb": xgb_preds,
        "spread": spread,
    }


# ── CLI entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    logger.info("Training XGBoost challenger...")
    xgb_model, _, xgb_cv = train()

    # Load LightGBM CV metrics for weight computation
    lgbm_meta_path = MODEL_DIR / "lgbm_cv_metrics.joblib"
    if lgbm_meta_path.exists():
        lgbm_cv = joblib.load(lgbm_meta_path)
        w_lgbm, w_xgb = compute_ensemble_weights(
            lgbm_cv["mean_mae"], xgb_cv["mean_mae"]
        )
    else:
        w_lgbm, w_xgb = 0.6, 0.4
        logger.warning("LightGBM CV metrics not found. Using default weights 0.6/0.4.")

    save_ensemble_weights(w_lgbm, w_xgb)

    print("\n" + "="*50)
    print("XGB CROSS-VALIDATION RESULTS")
    print("="*50)
    for fold in xgb_cv["folds"]:
        print(f"  Fold {fold['fold']}: MAE={fold['mae']:.2f}  RMSE={fold['rmse']:.2f}  MAPE={fold['mape']:.1f}%")
    print(f"\n  Mean MAE  : {xgb_cv['mean_mae']:.2f}")
    print(f"  Mean RMSE : {xgb_cv['mean_rmse']:.2f}")
    print(f"  Mean MAPE : {xgb_cv['mean_mape']:.1f}%")
    print(f"\n  Ensemble  : LightGBM×{w_lgbm:.3f} + XGBoost×{w_xgb:.3f}")
    print("="*50)