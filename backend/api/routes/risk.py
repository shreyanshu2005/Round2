"""
backend/api/routes/risk.py

Layer 4 — Risk Prediction API
Bengaluru Traffic Intelligence Platform (BTIP)

GET /api/v1/risk?zone_id=<int>&shift=<Morning|Afternoon|Evening|Night>&date=<YYYY-MM-DD>

Pipeline per request:
  1. Look up pre-computed zone (cluster_id) context — rolling counts, cluster
     probability/persistence, centroid lat/lng, dominant categoricals — from
     the feature store (computed once at process start, not per-request).
  2. Overlay hour/day_of_week/month/is_weekend/is_rush_hour derived from the
     requested shift + date.
  3. Run the LightGBM + XGBoost ensemble (backend.models.risk.xgb_challenger).
  4. Platt-calibrate the ensemble's P10/P50/P90 raw counts to a 0-100 Risk
     Score (backend.models.risk.calibration).
  5. Attach top-5 SHAP explanations (backend.models.risk.shap_explainer).

Mount in backend/main.py with:
    from backend.api.routes import risk
    app.include_router(risk.router, prefix="/api/v1")
"""
from __future__ import annotations

import logging
from datetime import date as date_cls
from functools import lru_cache

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.models.risk import calibration, lgbm_risk, shap_explainer, xgb_challenger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])

# Shift -> representative hour. Mirrors the Morning/Afternoon/Evening/Night
# shift buckets defined in Layer 2's temporal_features.py.
SHIFT_HOUR_MAP: dict[str, int] = {
    "Morning": 8,      # 06:00-11:59
    "Afternoon": 14,   # 12:00-16:59
    "Evening": 18,     # 17:00-20:59
    "Night": 23,       # 21:00-05:59
}
RUSH_HOURS = {7, 8, 9, 17, 18, 19, 20}


# ── Response models ──────────────────────────────────────────────────────

class ShapExplanationOut(BaseModel):
    feature: str
    raw_feature: str
    value: float
    impact: float
    direction: str


class ConfidenceBandOut(BaseModel):
    p10: float
    p50: float
    p90: float


class RiskResponse(BaseModel):
    zone_id: int
    shift: str
    date: date_cls
    risk_score: float = Field(..., ge=0, le=100)
    risk_label: str
    confidence_band: ConfidenceBandOut
    predicted_violations: float
    model_spread: float
    shap_explanations: list[ShapExplanationOut]


# ── Model bundle — loaded once per process, not per-request ─────────────

class RiskModelBundle:
    """Holds every trained Layer 4 artefact plus a pre-computed per-zone
    context table, so a request only needs zone_id/shift/date."""

    def __init__(self) -> None:
        self.lgbm_model, self.encoders, self.feature_cols = lgbm_risk.load_model()
        self.xgb_model = xgb_challenger.load_model()
        self.w_lgbm, self.w_xgb = xgb_challenger.load_ensemble_weights()
        self.calibrator = calibration.load_calibrator()
        self.explainer = self._load_or_build_explainer()
        self.zone_context = self._build_zone_context_table()

    def _load_or_build_explainer(self):
        try:
            return shap_explainer.load_explainer()
        except FileNotFoundError:
            logger.warning(
                "No saved SHAP explainer found — building one from the feature "
                "store on first request (this is a one-time cost)."
            )
            df = lgbm_risk.load_feature_store()
            df_agg = lgbm_risk._build_zone_windows(df)
            df_agg, _ = lgbm_risk._encode_categoricals(df_agg, encoders=self.encoders, fit=False)
            X_bg = df_agg[self.feature_cols].to_numpy()
            return shap_explainer.build_and_save_explainer(self.lgbm_model, X_background=X_bg)

    def _build_zone_context_table(self) -> dict[int, dict]:
        df = lgbm_risk.load_feature_store()
        df_agg = lgbm_risk._build_zone_windows(df)

        grouped = df_agg.group_by("cluster_id").agg(
            pl.col("rolling_7d_count").mean().alias("rolling_7d_count"),
            pl.col("rolling_30d_count").mean().alias("rolling_30d_count"),
            pl.col("cluster_probability").mean().alias("cluster_probability"),
            pl.col("cluster_persistence_score").mean().alias("cluster_persistence_score"),
            pl.col("latitude").mean().alias("latitude"),
            pl.col("longitude").mean().alias("longitude"),
            pl.col("police_station").mode().first().alias("police_station"),
            pl.col("vehicle_type").mode().first().alias("vehicle_type"),
            pl.col("primary_violation_type").mode().first().alias("primary_violation_type"),
        )

        context: dict[int, dict] = {}
        for row in grouped.iter_rows(named=True):
            context[int(row["cluster_id"])] = row
        return context


@lru_cache(maxsize=1)
def get_model_bundle() -> RiskModelBundle:
    """Process-wide singleton. FileNotFoundError propagates to the caller,
    which turns it into a 503 (training pipeline hasn't been run yet)."""
    return RiskModelBundle()


# ── Feature vector construction ──────────────────────────────────────────

def _build_feature_row(
    bundle: RiskModelBundle, zone_id: int, shift: str, the_date: date_cls
) -> pl.DataFrame:
    if zone_id not in bundle.zone_context:
        raise HTTPException(
            status_code=404,
            detail=f"zone_id={zone_id} not found (no HDBSCAN cluster with this id in the trained feature store).",
        )
    ctx = bundle.zone_context[zone_id]

    hour = SHIFT_HOUR_MAP[shift]
    day_of_week = the_date.weekday()  # Monday=0 .. Sunday=6
    month = the_date.month
    is_weekend = int(day_of_week >= 5)
    is_rush_hour = int(hour in RUSH_HOURS)
    is_holiday = 0  # Bengaluru holiday calendar lives in Layer 2; default here

    return pl.DataFrame({
        "cluster_id": [zone_id],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "month": [month],
        "is_weekend": [is_weekend],
        "is_rush_hour": [is_rush_hour],
        "is_holiday": [is_holiday],
        "rolling_7d_count": [ctx["rolling_7d_count"]],
        "rolling_30d_count": [ctx["rolling_30d_count"]],
        "cluster_probability": [ctx["cluster_probability"]],
        "cluster_persistence_score": [ctx["cluster_persistence_score"]],
        "latitude": [ctx["latitude"]],
        "longitude": [ctx["longitude"]],
        "police_station": [ctx["police_station"] or "UNKNOWN"],
        "vehicle_type": [ctx["vehicle_type"] or "UNKNOWN"],
        "primary_violation_type": [ctx["primary_violation_type"] or "UNKNOWN"],
    })


# ── Core query logic (plain callable — safe for both REST and GraphQL) ────

def _query_risk(
    zone_id: int,
    shift: str,
    date: date_cls,
) -> RiskResponse:
    """
    Shared risk-scoring logic. No FastAPI Query() wrappers — plain Python
    arguments only, so this can be called directly (e.g. from GraphQL
    resolvers in backend/api/graphql/schema.py) as well as from the REST
    route below.
    """
    if shift not in SHIFT_HOUR_MAP:
        raise HTTPException(
            status_code=422,
            detail=f"shift must be one of {list(SHIFT_HOUR_MAP)}, got '{shift}'.",
        )

    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Risk models not trained yet — run lgbm_risk.train(), "
                    f"xgb_challenger.train(), and calibration.train_calibrator() first. ({exc})",
        )

    row = _build_feature_row(bundle, zone_id, shift, date)
    row_encoded, _ = lgbm_risk._encode_categoricals(row, encoders=bundle.encoders, fit=False)
    for col in bundle.feature_cols:
        if col not in row_encoded.columns:
            row_encoded = row_encoded.with_columns(pl.lit(0).alias(col))
    X = row_encoded[bundle.feature_cols].to_numpy()

    uncertainty = xgb_challenger.ensemble_predict_with_uncertainty(
        X, bundle.lgbm_model, bundle.xgb_model, bundle.w_lgbm, bundle.w_xgb
    )
    predicted_violations = float(uncertainty["p50"][0])
    spread = float(uncertainty["spread"][0])

    bands = bundle.calibrator.predict_score_with_bands(
        raw_preds=uncertainty["p50"],
        p10_raw=uncertainty["p10"],
        p90_raw=uncertainty["p90"],
    )
    risk_score = float(bands["p50"][0])
    risk_label = bundle.calibrator.risk_label(risk_score)

    shap_items = shap_explainer.explain_single(
        bundle.explainer, X[0], bundle.feature_cols, top_n=5
    )

    return RiskResponse(
        zone_id=zone_id,
        shift=shift,
        date=date,
        risk_score=round(risk_score, 2),
        risk_label=risk_label,
        confidence_band=ConfidenceBandOut(
            p10=round(float(bands["p10"][0]), 2),
            p50=round(risk_score, 2),
            p90=round(float(bands["p90"][0]), 2),
        ),
        predicted_violations=round(predicted_violations, 2),
        model_spread=round(spread, 2),
        shap_explanations=[ShapExplanationOut(**s) for s in shap_items],
    )


# ── Route ──────────────────────────────────────────────────────────────────

@router.get("", response_model=RiskResponse)
def get_risk(
    zone_id: int = Query(..., description="HDBSCAN cluster_id from Layer 3"),
    shift: str = Query(..., description="Morning | Afternoon | Evening | Night"),
    date: date_cls = Query(..., description="YYYY-MM-DD"),
) -> RiskResponse:
    return _query_risk(zone_id=zone_id, shift=shift, date=date)


# Plain-callable alias for GraphQL resolvers (backend/api/graphql/schema.py)
# or any other internal caller that needs risk data without going through
# FastAPI's request/Query machinery.
get_risk_data = _query_risk