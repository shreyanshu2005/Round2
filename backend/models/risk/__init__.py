"""
Risk Prediction Module
Bengaluru Traffic Intelligence Platform (BTIP)

Exports:
- LightGBM risk model
- XGBoost challenger model
- Platt calibrator
- SHAP explainability utilities
"""

# LightGBM
from .lgbm_risk import (
    train as train_lgbm,
    load_model as load_lgbm_model,
    predict,
    predict_single,
    feature_importance,
)

# Calibration
from .calibration import (
    PlattCalibrator,
    train_calibrator,
    load_calibrator,
    score_to_label,
    score_to_color,
)

# SHAP
from .shap_explainer import (
    build_explainer,
    build_and_save_explainer,
    load_explainer,
    explain_single,
    explain_batch,
    get_global_importance,
)

# XGBoost + Ensemble
from .xgb_challenger import (
    train as train_xgb,
    load_model as load_xgb_model,
    ensemble_predict,
    ensemble_predict_with_uncertainty,
    compute_ensemble_weights,
)

__all__ = [
    # LightGBM
    "train_lgbm",
    "load_lgbm_model",
    "predict",
    "predict_single",
    "feature_importance",

    # Calibration
    "PlattCalibrator",
    "train_calibrator",
    "load_calibrator",
    "score_to_label",
    "score_to_color",

    # SHAP
    "build_explainer",
    "build_and_save_explainer",
    "load_explainer",
    "explain_single",
    "explain_batch",
    "get_global_importance",

    # XGB
    "train_xgb",
    "load_xgb_model",
    "ensemble_predict",
    "ensemble_predict_with_uncertainty",
    "compute_ensemble_weights",
]