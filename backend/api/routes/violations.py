"""
backend/api/routes/violations.py
---------------------------------
GET /api/v1/violations

Filters: date_range, junction_id, offence_type, bbox (geo bounding box)
Pagination: limit / offset

Uses SQLAlchemy + PostGIS ST_Within for bbox filtering. Assumes the
`violations` table has a PostGIS geometry column `geom` (POINT, SRID 4326)
created in Layer 1, plus indexes on `timestamp` and `junction_id`
(added in this layer — see migration note at bottom of file).

Response shape
--------------
{
  "total": 298234,
  "limit": 100,
  "offset": 0,
  "results": [
    {
      "violation_id": "...",
      "timestamp": "2024-01-15T08:30:00",
      "latitude": 12.93,
      "longitude": 77.61,
      "junction_id": "J123",
      "offence_type": "signal_jump",
      "vehicle_type": "two_wheeler",
      "fine_amount": 500,
      "officer_id": "O45",
      "validation_status": "approved"
    },
    ...
  ]
}
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.core.database import get_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/violations", tags=["violations"])

MAX_LIMIT = 1000


# ── Response models ──────────────────────────────────────────────────────────

class ViolationRecord(BaseModel):
    violation_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    junction_id: Optional[str] = None
    offence_type: Optional[str] = None
    vehicle_type: Optional[str] = None
    fine_amount: Optional[float] = None
    officer_id: Optional[str] = None
    validation_status: Optional[str] = None


class ViolationsResponse(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[ViolationRecord]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_bbox(bbox: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """
    bbox query param format: "min_lng,min_lat,max_lng,max_lat"
    Returns None if not provided. Raises HTTPException on malformed input.
    """
    if not bbox:
        return None
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
        min_lng, min_lat, max_lng, max_lat = parts
        if not (-180 <= min_lng <= 180 and -180 <= max_lng <= 180):
            raise ValueError
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError
        return min_lng, min_lat, max_lng, max_lat
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="bbox must be 'min_lng,min_lat,max_lng,max_lat' with valid coordinate ranges",
        )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("", response_model=ViolationsResponse)
async def get_violations(
    date_from: Optional[datetime] = Query(None, description="Inclusive start of date_range"),
    date_to: Optional[datetime] = Query(None, description="Inclusive end of date_range"),
    junction_id: Optional[str] = Query(None),
    offence_type: Optional[str] = Query(None),
    bbox: Optional[str] = Query(
        None, description="min_lng,min_lat,max_lng,max_lat"
    ),
    limit: int = Query(100, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
) -> ViolationsResponse:
    """
    Filtered, paginated violation records.

    bbox filtering uses ST_Within(geom, ST_MakeEnvelope(...)) so it can use
    the PostGIS GiST index on `geom`. timestamp/junction_id filters use the
    btree indexes added in this layer.
    """
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=400, detail="date_from must be <= date_to")

    bbox_tuple = _parse_bbox(bbox)

    where_clauses: list[str] = []
    params: dict = {"limit": limit, "offset": offset}

    if date_from:
        where_clauses.append("timestamp >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where_clauses.append("timestamp <= :date_to")
        params["date_to"] = date_to
    if junction_id:
        where_clauses.append("junction_id = :junction_id")
        params["junction_id"] = junction_id
    if offence_type:
        where_clauses.append("offence_type = :offence_type")
        params["offence_type"] = offence_type
    if bbox_tuple:
        min_lng, min_lat, max_lng, max_lat = bbox_tuple
        where_clauses.append(
            "ST_Within(geom, ST_MakeEnvelope(:min_lng, :min_lat, :max_lng, :max_lat, 4326))"
        )
        params.update(
            min_lng=min_lng, min_lat=min_lat, max_lng=max_lng, max_lat=max_lat
        )

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_sql = text(f"SELECT COUNT(*) FROM violations {where_sql}")
    select_sql = text(
        f"""
        SELECT
            violation_id,
            timestamp,
            ST_Y(geom) AS latitude,
            ST_X(geom) AS longitude,
            junction_id,
            offence_type,
            vehicle_type,
            fine_amount,
            officer_id,
            validation_status
        FROM violations
        {where_sql}
        ORDER BY timestamp DESC
        LIMIT :limit OFFSET :offset
        """
    )

    engine = get_engine()
    async with engine.connect() as conn:
        total = (await conn.execute(count_sql, params)).scalar_one()
        rows = (await conn.execute(select_sql, params)).mappings().all()

    results = [ViolationRecord(**dict(row)) for row in rows]

    logger.info(
        "GET /violations  filters=%s  total=%d  returned=%d",
        {k: v for k, v in params.items() if k not in ("limit", "offset")},
        total,
        len(results),
    )

    return ViolationsResponse(total=total, limit=limit, offset=offset, results=results)


# ── One-time migration note (run manually or via Alembic) ─────────────────────
#
# CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations (timestamp);
# CREATE INDEX IF NOT EXISTS idx_violations_junction_id ON violations (junction_id);
# CREATE INDEX IF NOT EXISTS idx_violations_geom ON violations USING GIST (geom);
