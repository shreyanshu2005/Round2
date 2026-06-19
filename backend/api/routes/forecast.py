"""
forecast.py — FastAPI route for /api/v1/forecast
--------------------------------------------------
Routes:
  GET /forecast               — 24h or 7d P10/P50/P90 forecast for a junction
  GET /forecast/risk-calendar — 7×4 shift-risk grid for a junction
  GET /forecast/top-junctions — List of top-20 LSTM-served junctions

Model routing:
  junction in top-20 AND LSTM model available → LSTM
  otherwise → Prophet
"""

import logging
import warnings
from typing import Literal, Optional

import numpy as np
import polars as pl
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["Forecast"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    ts: str
    p10: float
    p50: float
    p90: float
    source: str  # 'lstm' | 'prophet' | 'global'


class ForecastResponse(BaseModel):
    junction_id: str
    horizon: str
    model_used: str
    points: list[ForecastPoint]


class CalendarCell(BaseModel):
    day: str
    shift: str
    p50: float


class RiskCalendarResponse(BaseModel):
    junction_id: str
    calendar: list[list[CalendarCell]]  # 7 days × 4 shifts


# ---------------------------------------------------------------------------
# Lazy model loaders (avoid startup cost)
# ---------------------------------------------------------------------------

_prophet_module = None
_lstm_module = None
_top20_cache: Optional[list[str]] = None
_feature_store: Optional[pl.DataFrame] = None

FEATURE_STORE_PATH = "data/processed/feature_store.parquet"


def _get_prophet():
    global _prophet_module
    if _prophet_module is None:
        from backend.models.forecasting import prophet_forecast
        _prophet_module = prophet_forecast
    return _prophet_module


def _get_lstm():
    global _lstm_module
    if _lstm_module is None:
        from backend.models.forecasting import lstm_forecast
        _lstm_module = lstm_forecast
    return _lstm_module


def _get_top20() -> list[str]:
    global _top20_cache
    if _top20_cache is None:
        lstm = _get_lstm()
        _top20_cache = lstm.get_top20_junctions(FEATURE_STORE_PATH)
    return _top20_cache


def _get_recent_series(junction_id: str, lookback: int = 336) -> np.ndarray:
    """Load the most recent `lookback` hourly violation counts for a junction."""
    global _feature_store
    if _feature_store is None:
        try:
            _feature_store = pl.read_parquet(FEATURE_STORE_PATH)
        except Exception:
            return np.zeros(lookback, dtype=np.float32)

    df = _feature_store
    junction_col = "junction_id_snapped" if "junction_id_snapped" in df.columns else "junction_name"
    ts_col = "created_datetime" if "created_datetime" in df.columns else "timestamp"

    jdf = (
        df.filter(pl.col(junction_col).cast(pl.Utf8) == str(junction_id))
        .with_columns(pl.col(ts_col).cast(pl.Datetime).dt.truncate("1h").alias("hour_bucket"))
        .group_by("hour_bucket")
        .len()
        .sort("hour_bucket")
        .tail(lookback)
    )
    if len(jdf) == 0:
        return np.zeros(lookback, dtype=np.float32)
    return jdf["len"].to_numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=ForecastResponse)
async def get_forecast(
    junction_id: str = Query(..., description="Junction identifier"),
    horizon: Literal["24h", "7d"] = Query("24h", description="Forecast horizon: 24h or 7d"),
):
    """
    Return P10/P50/P90 hourly forecast for a junction.

    - Top-20 junctions served by LSTM (quantile regression)
    - All others served by Prophet (holiday-aware)
    """
    horizon_hours = 24 if horizon == "24h" else 168

    try:
        top20 = _get_top20()
        use_lstm = str(junction_id) in top20

        if use_lstm:
            lstm = _get_lstm()
            recent = _get_recent_series(junction_id)
            points_raw = lstm.forecast(junction_id, recent, horizon_hours=horizon_hours)

            if not points_raw:  # LSTM returned empty (e.g. horizon mismatch)
                use_lstm = False

        if not use_lstm:
            prophet = _get_prophet()
            points_raw = prophet.forecast(junction_id, horizon_hours=horizon_hours)

        if not points_raw:
            raise HTTPException(status_code=500, detail="Forecast generation failed — check model artifacts")

        model_used = "lstm" if use_lstm else points_raw[0].get("source", "prophet")
        points = [ForecastPoint(**p) for p in points_raw]

        # Final sanity check: P10 ≤ P50 ≤ P90
        for pt in points:
            pt.p10 = min(pt.p10, pt.p50)
            pt.p90 = max(pt.p90, pt.p50)

        return ForecastResponse(
            junction_id=str(junction_id),
            horizon=horizon,
            model_used=model_used,
            points=points,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet: {e}. Run scripts/train_all_models.py first.",
        )
    except Exception as e:
        logger.exception(f"Forecast error for junction {junction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-calendar", response_model=RiskCalendarResponse)
async def get_risk_calendar(
    junction_id: str = Query(..., description="Junction identifier"),
):
    """
    Return 7-day × 4-shift risk calendar (P50 violation count per cell).
    Useful for risk heatmap grid on the Forecast Dashboard.
    """
    try:
        prophet = _get_prophet()
        raw_calendar = prophet.risk_calendar(junction_id)

        calendar = [
            [CalendarCell(**cell) for cell in row]
            for row in raw_calendar
        ]

        return RiskCalendarResponse(junction_id=str(junction_id), calendar=calendar)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Risk calendar error for junction {junction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-junctions")
async def get_top_junctions():
    """Return the list of top-20 junctions served by the LSTM model."""
    try:
        return {"top_junctions": _get_top20(), "model": "lstm"}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))