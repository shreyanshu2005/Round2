"""
Layer 4 — SHAP Explainer
Bengaluru Traffic Intelligence Platform (BTIP)

Computes SHAP values for every risk prediction.
Returns top-5 features with human-readable names, magnitude, and direction.
Works with both LightGBM and XGBoost models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import polars as pl
import shap

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models" / "saved" / "risk"
SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer.joblib"

# Human-readable feature name mapping
FEATURE_LABELS: dict[str, str] = {
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
    "is_weekend": "Weekend",
    "is_rush_hour": "Rush Hour",
    "is_holiday": "Public Holiday",
    "rolling_7d_count": "7-Day Violation Trend",
    "rolling_30d_count": "30-Day Violation Trend",
    "cluster_id": "Hotspot Zone",
    "cluster_probability": "Zone Hotspot Confidence",
    "cluster_persistence_score": "Hotspot Persistence (Structural)",
    "latitude": "Location (Latitude)",
    "longitude": "Location (Longitude)",
    "police_station_enc": "Police Station",
    "vehicle_type_enc": "Vehicle Type",
    "primary_violation_type_enc": "Offence Type",
    "window_4h": "4-Hour Window",
}


def _readable(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


# ── Build & save explainer ──────────────────────────────────────────────────

def build_explainer(
    model: Any,
    X_background: Optional[np.ndarray] = None,
    n_background: int = 200,
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for an LightGBM or XGBoost model.
    Optionally pass a background dataset for expected value computation.
    """
    if X_background is not None and len(X_background) > n_background:
        # Use a random subset as background — speeds up SHAP significantly
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_background), size=n_background, replace=False)
        background = X_background[idx]
    else:
        background = X_background

    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    return explainer


def save_explainer(explainer: shap.TreeExplainer) -> None:
    joblib.dump(explainer, SHAP_EXPLAINER_PATH)
    logger.info(f"SHAP explainer saved → {SHAP_EXPLAINER_PATH}")


def load_explainer() -> shap.TreeExplainer:
    if not SHAP_EXPLAINER_PATH.exists():
        raise FileNotFoundError(
            f"No saved SHAP explainer at {SHAP_EXPLAINER_PATH}. "
            "Run build_and_save_explainer() first."
        )
    return joblib.load(SHAP_EXPLAINER_PATH)


def build_and_save_explainer(
    model: Any,
    X_background: Optional[np.ndarray] = None,
) -> shap.TreeExplainer:
    explainer = build_explainer(model, X_background)
    save_explainer(explainer)
    return explainer


# ── SHAP value computation ──────────────────────────────────────────────────

def explain_single(
    explainer: shap.TreeExplainer,
    x: np.ndarray,
    feature_cols: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Compute SHAP values for a single feature vector x (shape: [n_features]).
    Returns top_n features sorted by absolute SHAP impact.

    Output format:
      [{"feature": "Rush Hour", "raw_feature": "is_rush_hour",
        "value": 1.0, "impact": 3.2, "direction": "+"},  ...]
    """
    x_2d = x.reshape(1, -1)
    try:
        shap_values = explainer.shap_values(x_2d)
        # LightGBM regression returns array directly; handle tree ensembles
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        sv = shap_values[0]  # shape: [n_features]
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Returning empty explanations.")
        return []

    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[::-1][:top_n]

    explanations = []
    for idx in top_idx:
        impact = float(sv[idx])
        feature = feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}"
        value = float(x[idx])
        explanations.append({
            "feature": _readable(feature),
            "raw_feature": feature,
            "value": value,
            "impact": round(impact, 4),
            "direction": "+" if impact >= 0 else "-",
        })

    return explanations


def explain_batch(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_cols: list[str],
    top_n: int = 5,
) -> list[list[dict]]:
    """
    Compute SHAP explanations for a batch of rows.
    Returns list of explanation lists, one per row.
    """
    try:
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as e:
        logger.warning(f"Batch SHAP failed: {e}. Returning empty explanations.")
        return [[] for _ in range(len(X))]

    results = []
    for i, sv in enumerate(shap_values):
        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:top_n]
        row_explanations = []
        for idx in top_idx:
            impact = float(sv[idx])
            feature = feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}"
            value = float(X[i, idx])
            row_explanations.append({
                "feature": _readable(feature),
                "raw_feature": feature,
                "value": value,
                "impact": round(impact, 4),
                "direction": "+" if impact >= 0 else "-",
            })
        results.append(row_explanations)

    return results


def get_global_importance(
    explainer: shap.TreeExplainer,
    X: np.ndarray,
    feature_cols: list[str],
    sample_size: int = 500,
) -> list[dict]:
    """
    Global feature importance via mean absolute SHAP values.
    Uses a sample for speed.
    """
    if len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    try:
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    except Exception as e:
        logger.error(f"Global SHAP failed: {e}")
        return []

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    pairs = sorted(
        zip(feature_cols, mean_abs),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"feature": _readable(f), "raw_feature": f, "mean_abs_shap": round(float(v), 4)}
        for f, v in pairs
    ]


# ── Convenience: explain from raw input dict ────────────────────────────────

def explain_from_dict(
    input_dict: dict,
    feature_cols: list[str],
    explainer: Optional[shap.TreeExplainer] = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Explain a prediction given a raw input dict (API-friendly).
    Missing features are filled with 0.
    """
    if explainer is None:
        explainer = load_explainer()

    x = np.array([float(input_dict.get(col, 0)) for col in feature_cols])
    return explain_single(explainer, x, feature_cols, top_n=top_n)


# ── CLI entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

    from backend.models.risk.lgbm_risk import load_model, load_feature_store, _build_zone_windows, _encode_categoricals, get_feature_cols

    logger.info("Loading model and feature store...")
    model, encoders, feature_cols = load_model()
    df = load_feature_store()
    df_agg = _build_zone_windows(df)
    df_agg, _ = _encode_categoricals(df_agg, encoders=encoders, fit=False)
    feat_cols = get_feature_cols(df_agg)
    X = df_agg[feat_cols].to_numpy()

    logger.info(f"Building SHAP explainer on {min(200, len(X))} background samples...")
    explainer = build_and_save_explainer(model, X_background=X)

    # Test on first row
    sample_x = X[0]
    explanation = explain_single(explainer, sample_x, feat_cols)
    print("\nSample explanation (first row):")
    for e in explanation:
        arrow = "▲" if e["direction"] == "+" else "▼"
        print(f"  {arrow} {e['feature']:40s} impact={e['impact']:+.3f}  value={e['value']}")

    # Global importance
    global_imp = get_global_importance(explainer, X, feat_cols)
    print("\nGlobal feature importance (mean |SHAP|):")
    for g in global_imp[:10]:
        bar = "█" * int(g["mean_abs_shap"] * 5)
        print(f"  {g['feature']:40s} {g['mean_abs_shap']:.4f}  {bar}")