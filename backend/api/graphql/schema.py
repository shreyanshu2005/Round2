"""
backend/api/graphql/schema.py
Strawberry GraphQL schema for BTIP. Resolvers delegate to the same
service-layer logic used by the REST routes (no duplicated business logic).
"""
from __future__ import annotations

from typing import List, Optional

import strawberry

# NOTE: these imports point at the service-layer functions that the REST
# routes (backend/api/routes/*.py) already call. GraphQL resolvers should
# NEVER reimplement query logic — they call the same functions REST does.
from backend.api.routes.hotspots import get_hotspots_data
from backend.api.routes.forecast import get_forecast_data
from backend.api.routes.risk import get_risk_data
from backend.api.routes.recommendations import get_recommendations_data


@strawberry.type
class Hotspot:
    cluster_id: int
    centroid_lat: float
    centroid_lng: float
    violation_count: int
    cluster_probability: float
    persistence_score: float
    top_offence_types: List[str]


@strawberry.type
class ConfidencePoint:
    timestamp: str
    p10: float
    p50: float
    p90: float


@strawberry.type
class Forecast:
    junction_id: str
    horizon: str
    points: List[ConfidencePoint]


@strawberry.type
class ShapExplanation:
    feature: str
    impact: float
    direction: str


@strawberry.type
class RiskScore:
    zone_id: str
    risk_score: float
    confidence_p10: float
    confidence_p50: float
    confidence_p90: float
    predicted_violations: float
    shap_explanations: List[ShapExplanation]


@strawberry.type
class Recommendation:
    zone_id: str
    n_officers: int
    risk_score: float
    expected_reduction_pct: float
    confidence: float
    shap_explanations: List[ShapExplanation]


@strawberry.type
class Query:
    @strawberry.field
    async def hotspots(
        self, bbox: Optional[str] = None, date_range: Optional[str] = None
    ) -> List[Hotspot]:
        rows = await get_hotspots_data(bbox=bbox, date_range=date_range)
        return [Hotspot(**r) for r in rows]

    @strawberry.field
    async def forecast(self, junction_id: str, horizon: str = "24h") -> Forecast:
        data = await get_forecast_data(junction_id=junction_id, horizon=horizon)
        return Forecast(
            junction_id=junction_id,
            horizon=horizon,
            points=[ConfidencePoint(**p) for p in data["points"]],
        )

    @strawberry.field
    async def risk_score(self, zone_id: str, shift: str, date: str) -> RiskScore:
        data = await get_risk_data(zone_id=zone_id, shift=shift, date=date)
        return RiskScore(
            zone_id=zone_id,
            risk_score=data["risk_score"],
            confidence_p10=data["confidence_band"]["p10"],
            confidence_p50=data["confidence_band"]["p50"],
            confidence_p90=data["confidence_band"]["p90"],
            predicted_violations=data["predicted_violations"],
            shap_explanations=[ShapExplanation(**s) for s in data["shap_explanations"]],
        )

    @strawberry.field
    async def recommendation(
        self, shift: str, date: str, total_officers: int
    ) -> List[Recommendation]:
        rows = await get_recommendations_data(
            shift=shift, date=date, total_officers=total_officers
        )
        return [
            Recommendation(
                zone_id=r["zone_id"],
                n_officers=r["n_officers"],
                risk_score=r["risk_score"],
                expected_reduction_pct=r["expected_reduction_pct"],
                confidence=r["confidence"],
                shap_explanations=[ShapExplanation(**s) for s in r["shap_explanations"]],
            )
            for r in rows
        ]


schema = strawberry.Schema(query=Query)