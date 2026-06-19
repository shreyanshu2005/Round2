"""
backend/api/routes/recommendations.py
---------------------------------------
GET /api/v1/recommendations

Two-stage patrol allocation:
  1. ILP (PuLP/CBC) — deterministic, always-on primary allocation.
  2. RL (PPO, optional) — advisory delta on top of ILP. If RL inference
     fails for any reason (model not trained, deps missing, bad shape),
     we silently fall back to ILP-only and flag it in the response.

Risk scores are pulled from the Layer 4 risk model (lgbm_risk.py +
calibration.py) directly — Layer 4's /risk HTTP route may not exist yet
per the Current State doc, so this route calls the model module in-process
rather than over HTTP, to avoid taking on that dependency.

SHAP explanations reuse Layer 4's shap_explainer.py per recommended zone.

Response shape
--------------
{
  "shift": "Evening",
  "date": "2024-01-15",
  "total_officers": 20,
  "officers_allocated": 20,
  "solver_status": "Optimal",
  "rl_available": true,
  "recommendations": [
    {
      "zone_id": "J11",
      "n_officers": 2,
      "risk_score": 97.0,
      "congestion_score": 81.4,
      "expected_reduction_pct": 63.2,
      "advisory_delta": 1,
      "confidence": "high",
      "shap_explanations": [{"feature": "...", "impact": 0.0, "direction": "+"}]
    },
    ...
  ]
}
"""

from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.decision.ilp_optimizer import (
    ILPOptimizer,
    ZoneInput,
    deterrence_factor,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

MAX_TOTAL_OFFICERS = 200


# ── Response models ──────────────────────────────────────────────────────────

class ShapExplanation(BaseModel):
    feature: str
    impact: float
    direction: str  # "+" or "-"


class RecommendationItem(BaseModel):
    zone_id: str
    n_officers: int
    risk_score: float
    congestion_score: float
    expected_reduction_pct: float
    advisory_delta: Optional[int] = None
    confidence: str
    shap_explanations: list[ShapExplanation] = []


class RecommendationsResponse(BaseModel):
    shift: str
    date: str
    total_officers: int
    officers_allocated: int
    solver_status: str
    rl_available: bool
    recommendations: list[RecommendationItem]


# ── Data fetch helpers ────────────────────────────────────────────────────────

def _confidence_label(risk_score: float) -> str:
    if risk_score >= 67:
        return "high"
    if risk_score >= 34:
        return "medium"
    return "low"


def _get_risk_and_shap_for_shift(shift: str, date: str) -> tuple[dict[str, float], dict[str, list[dict]]]:
    """
    Fetch per-zone calibrated risk scores + top-5 SHAP explanations for the
    given shift/date by calling Layer 4's modules in-process.

    Calls backend.models.risk.lgbm_risk / calibration / shap_explainer
    directly. If any of those raise (e.g. model artefacts not trained yet),
    the exception propagates and the route returns 503 — there is no
    meaningful recommendation without a risk model.
    """
    from backend.models.risk import lgbm_risk, calibration, shap_explainer

    df = lgbm_risk.load_feature_store()
    df_agg = lgbm_risk._build_zone_windows(df)

    model, encoders, feature_cols = lgbm_risk.load_model()
    df_enc, _ = lgbm_risk._encode_categoricals(df_agg, encoders=encoders, fit=False)

    # Filter to the requested shift if the column exists; else use all zones
    # for that date as a best-effort fallback.
    if "shift" in df_enc.columns:
        df_enc = df_enc.filter(df_enc["shift"] == shift)

    if df_enc.height == 0:
        raise ValueError(f"No zone-window rows found for shift={shift}, date={date}")

    X = df_enc[feature_cols].to_numpy()
    zone_ids = [str(z) for z in df_enc["junction_id_snapped"].to_list()] if (
        "junction_id_snapped" in df_enc.columns
    ) else [str(i) for i in range(df_enc.height)]

    raw_preds = model.predict(X).clip(min=0)
    calibrator = calibration.load_calibrator()
    risk_scores_arr = calibrator.predict_score(raw_preds)

    risk_scores = dict(zip(zone_ids, risk_scores_arr.tolist()))

    explainer = shap_explainer.build_explainer(model, X_background=X)
    shap_by_zone: dict[str, list[dict]] = {}
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    for i, zone_id in enumerate(zone_ids):
        row_shap = shap_values[i]
        top5_idx = abs(row_shap).argsort()[::-1][:5]
        shap_by_zone[zone_id] = [
            {
                "feature": shap_explainer._readable(feature_cols[j]),
                "impact": float(row_shap[j]),
                "direction": "+" if row_shap[j] >= 0 else "-",
            }
            for j in top5_idx
        ]

    return risk_scores, shap_by_zone


def _get_congestion_scores() -> dict[str, float]:
    """Best-effort load of Layer 6 congestion scores. Returns {} if missing."""
    opt = ILPOptimizer()
    return opt._load_congestion_scores()


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=RecommendationsResponse)
async def get_recommendations(
    shift: str = Query(..., description="Morning | Afternoon | Evening | Night"),
    date: date_type = Query(..., description="YYYY-MM-DD"),
    total_officers: int = Query(20, ge=1, le=MAX_TOTAL_OFFICERS),
) -> RecommendationsResponse:
    """
    Two-stage patrol recommendation: ILP primary allocation + optional RL
    advisory delta + per-zone SHAP explanations from the Layer 4 risk model.
    """
    # 1. Risk scores + SHAP (Layer 4)
    try:
        risk_scores, shap_by_zone = _get_risk_and_shap_for_shift(shift, str(date))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Risk model not trained yet (Layer 4 incomplete): {e}",
        )
    except Exception as e:
        logger.exception("Failed to fetch risk scores for recommendations")
        raise HTTPException(status_code=500, detail=f"Risk scoring failed: {e}")

    # 2. Congestion scores (Layer 6) — optional, degrades gracefully
    congestion_scores = _get_congestion_scores()

    # 3. ILP primary allocation
    opt = ILPOptimizer()
    zones = opt.build_zone_inputs(risk_scores, congestion_scores)
    allocation, meta = opt.optimize(zones, total_officers)

    # 4. RL advisory delta — best-effort, never blocks the response
    rl_available = False
    advisory_deltas: dict[str, int] = {}
    try:
        from backend.decision.rl_agent import load_agent, predict_delta

        rl_model = load_agent()
        advisory_deltas = predict_delta(rl_model, risk_scores, congestion_scores, allocation)
        rl_available = True
    except Exception as e:
        logger.info("RL advisory unavailable, falling back to ILP-only: %s", e)

    # 5. Assemble response — only zones that actually received officers
    zone_by_id = {z.zone_id: z for z in zones}
    items: list[RecommendationItem] = []
    for zone_id, n_officers in allocation.items():
        if n_officers <= 0:
            continue
        zone = zone_by_id[zone_id]
        reduction_pct = round(deterrence_factor(n_officers, opt.deterrence_k) * 100, 1)
        shap_list = [
            ShapExplanation(**s) for s in shap_by_zone.get(zone_id, [])
        ]
        items.append(
            RecommendationItem(
                zone_id=zone_id,
                n_officers=n_officers,
                risk_score=round(zone.risk_score, 1),
                congestion_score=round(zone.congestion_score, 1),
                expected_reduction_pct=reduction_pct,
                advisory_delta=advisory_deltas.get(zone_id) if rl_available else None,
                confidence=_confidence_label(zone.risk_score),
                shap_explanations=shap_list,
            )
        )

    items.sort(key=lambda r: r.risk_score, reverse=True)

    return RecommendationsResponse(
        shift=shift,
        date=str(date),
        total_officers=total_officers,
        officers_allocated=sum(allocation.values()),
        solver_status=str(meta.get("status", "unknown")),
        rl_available=rl_available,
        recommendations=items,
    )