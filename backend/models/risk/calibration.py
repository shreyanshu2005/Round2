"""
Layer 4 — Platt Scaling Calibration
Bengaluru Traffic Intelligence Platform (BTIP)

Maps raw violation count predictions → calibrated Risk Score 0–100.
Uses a sigmoid (Platt) fit on held-out fold predictions.

Risk Score semantics:
  0–33   → Low risk   (green)
  34–66  → Medium risk (amber)
  67–100 → High risk   (red)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # stable sigmoid
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "saved" / "risk"
CALIBRATOR_PATH = MODEL_DIR / "platt_calibrator.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Risk band thresholds
LOW_THRESHOLD = 33
HIGH_THRESHOLD = 67


# ── Sigmoid model ───────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt sigmoid: sigma(a*x + b)."""
    return expit(a * x + b)


class PlattCalibrator:
    """
    Fits a sigmoid on raw predictions to produce calibrated scores in [0, 1].
    Scores are then scaled to [0, 100].

    The target used during fitting is a normalised violation rate:
      target = clip(raw_violations / p95_violations, 0, 1)
    This maps the distribution so sigmoid saturation roughly corresponds to
    the 95th-percentile violation rate → avoids a few extreme events dominating.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0
        self.p95: float = 1.0  # 95th percentile of training violation counts
        self._fitted: bool = False

    def fit(
        self,
        raw_preds: np.ndarray,
        y_true: np.ndarray,
    ) -> "PlattCalibrator":
        """
        Fit Platt scaling.

        raw_preds : model output (predicted violation counts, ≥ 0)
        y_true    : actual violation counts from held-out fold
        """
        raw_preds = np.maximum(raw_preds, 0).astype(float)
        y_true = np.maximum(y_true, 0).astype(float)

        # Normalise targets to [0, 1] using 95th percentile cap
        # NEW
        p99 = float(np.percentile(y_true, 99))
        self.p95 = max(p99, 10.0)   # floor at 10 so sigmoid isn't compressed
        y_norm = np.clip(y_true / self.p95, 0.0, 1.0)

        # Normalise predictions by the same scale for consistent fitting
        x_norm = np.clip(raw_preds / self.p95, 0.0, 5.0)  # allow extrapolation

        try:
            popt, _ = curve_fit(
                _sigmoid,
                x_norm,
                y_norm,
                p0=[1.0, 0.0],
                maxfev=5000,
            )
            self.a, self.b = float(popt[0]), float(popt[1])
        except RuntimeError:
            logger.warning("curve_fit did not converge. Using identity sigmoid (a=1, b=0).")
            self.a, self.b = 1.0, 0.0

        self._fitted = True
        logger.info(f"PlattCalibrator fitted: a={self.a:.4f}  b={self.b:.4f}  p95={self.p95:.2f}")
        return self

    def predict_score(self, raw_preds: np.ndarray) -> np.ndarray:
        """
        Convert raw violation-count predictions → Risk Score [0, 100].
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call .fit() first.")
        raw_preds = np.maximum(raw_preds, 0).astype(float)
        x_norm = raw_preds / self.p95
        prob = _sigmoid(x_norm, self.a, self.b)
        scores = np.clip(prob * 100.0, 0.0, 100.0)
        return scores

    def predict_score_with_bands(
        self,
        raw_preds: np.ndarray,
        p10_raw: Optional[np.ndarray] = None,
        p90_raw: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Returns P10/P50/P90 risk scores.
        If p10_raw / p90_raw not provided, uses ±20% of p50 as approximation.
        """
        p50 = self.predict_score(raw_preds)

        if p10_raw is not None:
            p10 = self.predict_score(p10_raw)
        else:
            p10 = np.clip(p50 * 0.8, 0.0, 100.0)

        if p90_raw is not None:
            p90 = self.predict_score(p90_raw)
        else:
            p90 = np.clip(p50 * 1.2, 0.0, 100.0)

        return {"p10": p10, "p50": p50, "p90": p90}

    def risk_label(self, score: float) -> str:
        if score <= LOW_THRESHOLD:
            return "LOW"
        elif score <= HIGH_THRESHOLD:
            return "MEDIUM"
        return "HIGH"

    def risk_color(self, score: float) -> str:
        if score <= LOW_THRESHOLD:
            return "#00A86B"   # green
        elif score <= HIGH_THRESHOLD:
            return "#FFB020"   # amber
        return "#FF4444"       # red


# ── Fit from CV folds ───────────────────────────────────────────────────────

def fit_calibrator_from_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> PlattCalibrator:
    """
    Fit Platt calibrator on out-of-fold predictions.
    This avoids leaking training data into the calibration.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros(len(y))
    oof_actual = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        m.fit(X[train_idx], y[train_idx],
              eval_set=[(X[val_idx], y[val_idx])],
              callbacks=[__import__("lightgbm").early_stopping(50, verbose=False),
                         __import__("lightgbm").log_evaluation(-1)])
        oof_preds[val_idx] = np.maximum(m.predict(X[val_idx]), 0)
        oof_actual[val_idx] = y[val_idx]

    calibrator = PlattCalibrator()
    calibrator.fit(oof_preds, oof_actual)
    return calibrator


# ── Save / load ─────────────────────────────────────────────────────────────

def save_calibrator(calibrator: PlattCalibrator) -> None:
    joblib.dump(calibrator, CALIBRATOR_PATH)
    logger.info(f"Calibrator saved → {CALIBRATOR_PATH}")


def load_calibrator() -> PlattCalibrator:
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"No calibrator at {CALIBRATOR_PATH}. Run train_calibrator() first."
        )
    return joblib.load(CALIBRATOR_PATH)


def train_calibrator(
    model=None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> PlattCalibrator:
    """
    Full calibration training pipeline.
    If model/X/y not supplied, loads from saved LightGBM artefacts.
    """
    if model is None or X is None or y is None:
        from backend.models.risk.lgbm_risk import (
            load_model, load_feature_store, _build_zone_windows,
            _encode_categoricals, get_feature_cols, TARGET,
        )
        model, encoders, feature_cols = load_model()
        df = load_feature_store()
        df_agg = _build_zone_windows(df)
        df_agg, _ = _encode_categoricals(df_agg, encoders=encoders, fit=False)
        feature_cols = get_feature_cols(df_agg)
        X = df_agg[feature_cols].to_numpy()
        y = df_agg[TARGET].to_numpy().astype(float)

    calibrator = fit_calibrator_from_cv(model, X, y)
    save_calibrator(calibrator)
    return calibrator


# ── Convenience helpers ──────────────────────────────────────────────────────

def score_to_label(score: float) -> str:
    if score <= LOW_THRESHOLD:
        return "LOW"
    elif score <= HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def score_to_color(score: float) -> str:
    if score <= LOW_THRESHOLD:
        return "#00A86B"
    elif score <= HIGH_THRESHOLD:
        return "#FFB020"
    return "#FF4444"


# ── CLI entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    logger.info("Training Platt calibrator...")
    calibrator = train_calibrator()

    # Demo: show how some raw counts map to risk scores
    test_counts = np.array([0, 5, 10, 20, 35, 50, 75, 100, 150])
    scores = calibrator.predict_score(test_counts)

    print("\n" + "="*50)
    print("CALIBRATION DEMO — Raw Count → Risk Score")
    print("="*50)
    for count, score in zip(test_counts, scores):
        label = calibrator.risk_label(score)
        bar = "█" * int(score / 5)
        print(f"  violations={count:4d}  →  score={score:5.1f}  [{label:6s}]  {bar}")
    print("="*50)
    print(f"\n  Sigmoid params: a={calibrator.a:.4f}  b={calibrator.b:.4f}")
    print(f"  p95 violation count: {calibrator.p95:.1f}")