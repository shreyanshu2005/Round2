"""
backend/api/graphql/schema.py
--------------------------------
Strawberry GraphQL schema mirroring the REST routes: hotspots, forecast,
riskScore, recommendation. Each resolver calls the SAME underlying logic
as its REST counterpart (no duplicated business logic) so the two APIs
never drift apart.

Mounted at /graphql in main.py via strawberry.fastapi.GraphQLRouter.

Example query
-------------
  query {
    hotspots(limit: 5) {
      clusterId
      centroidLat
      centroidLng
      violationCount
      persistenceScore
    }
  }
"""

from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Optional

import strawberry

logger = logging.getLogger(__name__)


# ── GraphQL types ─────────────────────────────────────────────────────────────

@strawberry.type
class Hotspot:
    cluster_id: str
    centroid_lat: float
    centroid_lng: float
    violation_count: int
    cluster_probability: float
    persistence_score: float


@strawberry.type
class ForecastPoint:
    timestamp: str
    p10: float
    p50: float
    p90: float


@strawberry.type
class ShapExplanationGQL:
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
    shap_explanations: list[ShapExplanationGQL]


@strawberry.type
class Recommendation:
    zone_id: str
    n_officers: int
    risk_score: float
    congestion_score: float
    expected_reduction_pct: float
    confidence: str


# ── Query root ────────────────────────────────────────────────────────────────

@strawberry.type
class Query:
    @strawberry.field
    def hotspots(self, limit: int = 10) -> list[Hotspot]:
        """
        Mirrors GET /api/v1/hotspots. Calls the same DB query logic as the
        REST route (backend/api/routes/hotspots.py) — see that module for
        the underlying SQL.
        """
        try:
            from backend.api.routes.hotspots import fetch_hotspots_sync
            rows = fetch_hotspots_sync(limit=limit)
            return [
                Hotspot(
                    cluster_id=str(r["cluster_id"]),
                    centroid_lat=float(r["centroid_lat"]),
                    centroid_lng=float(r["centroid_lng"]),
                    violation_count=int(r["violation_count"]),
                    cluster_probability=float(r.get("cluster_probability", 0.0)),
                    persistence_score=float(r.get("persistence_score", 0.0)),
                )
                for r in rows
            ]
        except Exception as e:
            logger.warning("GraphQL hotspots resolver failed: %s", e)
            return []

    @strawberry.field
    def forecast(self, junction_id: str, horizon: str = "24h") -> list[ForecastPoint]:
        """Mirrors GET /api/v1/forecast?junction_id=...&horizon=..."""
        try:
            from backend.api.routes.forecast import fetch_forecast_sync
            points = fetch_forecast_sync(junction_id=junction_id, horizon=horizon)
            return [
                ForecastPoint(timestamp=p["ts"], p10=p["p10"], p50=p["p50"], p90=p["p90"])
                for p in points
            ]
        except Exception as e:
            logger.warning("GraphQL forecast resolver failed: %s", e)
            return []

    @strawberry.field
    def risk_score(self, zone_id: str, shift: str, date: str) -> Optional[RiskScore]:
        """Mirrors GET /api/v1/risk?zone_id=...&shift=...&date=..."""
        try:
            from backend.api.routes.risk import fetch_risk_score_sync
            r = fetch_risk_score_sync(zone_id=zone_id, shift=shift, date=date)
            if r is None:
                return None
            return RiskScore(
                zone_id=r["zone_id"],
                risk_score=r["risk_score"],
                confidence_p10=r["confidence_band"]["p10"],
                confidence_p50=r["confidence_band"]["p50"],
                confidence_p90=r["confidence_band"]["p90"],
                predicted_violations=r["predicted_violations"],
                shap_explanations=[
                    ShapExplanationGQL(**s) for s in r.get("shap_explanations", [])
                ],
            )
        except Exception as e:
            logger.warning("GraphQL riskScore resolver failed: %s", e)
            return None

    @strawberry.field
    def recommendations(
        self, shift: str, date: str, total_officers: int = 20
    ) -> list[Recommendation]:
        """Mirrors GET /api/v1/recommendations?shift=...&date=...&total_officers=..."""
        try:
            import asyncio
            from backend.api.routes.recommendations import get_recommendations

            result = asyncio.get_event_loop().run_until_complete(
                get_recommendations(shift=shift, date=date_type.fromisoformat(date), total_officers=total_officers)
            )
            return [
                Recommendation(
                    zone_id=r.zone_id,
                    n_officers=r.n_officers,
                    risk_score=r.risk_score,
                    congestion_score=r.congestion_score,
                    expected_reduction_pct=r.expected_reduction_pct,
                    confidence=r.confidence,
                )
                for r in result.recommendations
            ]
        except Exception as e:
            logger.warning("GraphQL recommendations resolver failed: %s", e)
            return []


schema = strawberry.Schema(query=Query)