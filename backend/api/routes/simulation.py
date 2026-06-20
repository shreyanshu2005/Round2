"""
backend/api/routes/simulation.py
-----------------------------------
POST /api/v1/simulation

Runs a what-if patrol deployment scenario through the digital twin
(deterrence + graph diffusion) and returns before/after metrics with a
P10/P50/P90 confidence band.

Request body
------------
{
  "zone_allocations": [{"zone_id": "J11", "n_officers": 3}, ...],
  "shift": "Evening",
  "date": "2024-01-15",
  "window_hours": 4          // optional, default 4
}

Response shape
---------------
{
  "total_violations_before": 100.0,
  "total_violations_after": 70.2,
  "reduction_pct": 29.8,
  "congestion_improvement_pct": 27.9,
  "affected_junction_count": 4,
  "confidence_band": {"p10": 28.4, "p50": 29.7, "p90": 31.0},
  "latency_seconds": 0.02,
  "per_junction": [...]
}
"""

from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.simulation.digital_twin import DigitalTwin, MAX_TOTAL_OFFICERS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["simulation"])

# Module-level singleton — avoids reloading the OSM graph / deterrence
# params on every request (graph load is expensive, see Layer 6 notes).
_twin: Optional[DigitalTwin] = None


def _get_twin() -> DigitalTwin:
    global _twin
    if _twin is None:
        _twin = DigitalTwin()
    return _twin


# ── Request / response models ─────────────────────────────────────────────────

class ZoneAllocationInput(BaseModel):
    zone_id: str
    n_officers: int = Field(ge=0, le=10)


class SimulationRequest(BaseModel):
    zone_allocations: list[ZoneAllocationInput]
    shift: str
    date: date_type
    window_hours: int = Field(default=4, ge=1, le=24)

    @field_validator("zone_allocations")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("zone_allocations must contain at least one zone")
        return v


class ConfidenceBand(BaseModel):
    p10: float
    p50: float
    p90: float


class PerJunctionResult(BaseModel):
    junction_id: str
    n_officers: int
    violation_rate_before: float
    violation_rate_after: float
    congestion_score_before: float
    congestion_score_after: float
    reduction_pct: float
    spillover_received_pct: float
    is_directly_patrolled: bool


class SimulationResponse(BaseModel):
    total_violations_before: float
    total_violations_after: float
    reduction_pct: float
    congestion_improvement_pct: float
    affected_junction_count: int
    confidence_band: Optional[ConfidenceBand]
    window_hours: int
    shift: str
    date: str
    latency_seconds: float
    per_junction: list[PerJunctionResult]


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=SimulationResponse)
async def run_simulation(body: SimulationRequest) -> SimulationResponse:
    """
    Run a digital-twin what-if scenario for the given officer allocation.

    Validates total officers <= 50 (Build Guide limit). Delegates to
    DigitalTwin.run_scenario() for deterrence + graph diffusion + Monte
    Carlo confidence band.
    """
    zone_allocations = {z.zone_id: z.n_officers for z in body.zone_allocations}
    total_officers = sum(zone_allocations.values())

    if total_officers > MAX_TOTAL_OFFICERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Total officers ({total_officers}) exceeds the "
                f"{MAX_TOTAL_OFFICERS}-officer simulation limit."
            ),
        )
    if total_officers == 0:
        raise HTTPException(
            status_code=400,
            detail="zone_allocations must include at least one zone with n_officers > 0",
        )

    twin = _get_twin()
    try:
        result = twin.run_scenario(
            zone_allocations=zone_allocations,
            shift=body.shift,
            date=str(body.date),
            window_hours=body.window_hours,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Digital twin simulation failed")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")

    return SimulationResponse(**result)