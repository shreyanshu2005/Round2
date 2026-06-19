"""
prophet_forecast.py
--------------------
Per-junction Prophet models with Bengaluru holiday regressors.
Outputs P10/P50/P90 confidence bands for 24h and 7-day horizons.

Key design decisions:
- One Prophet model per junction, trained in parallel via joblib
- Junctions with < 30 observations fall back to a global model
- Serialised to models/saved/prophet/junction_{id}.pkl (one file per junction)
- Global fallback model at models/saved/prophet/global_model.pkl
"""

import os
import pickle
import warnings
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bengaluru public holiday calendar (hardcoded – extend as needed)
# ---------------------------------------------------------------------------
BENGALURU_HOLIDAYS = pd.DataFrame(
    {
        "holiday": [
            "New Year's Day", "Makar Sankranti", "Republic Day",
            "Ugadi", "Good Friday", "Dr. Ambedkar Jayanti",
            "May Day", "Independence Day", "Ganesh Chaturthi",
            "Mahalaya Amavasya", "Gandhi Jayanti", "Navami",
            "Dussehra", "Diwali", "Kannada Rajyotsava",
            "Christmas Day",
        ],
        "ds": pd.to_datetime([
            "2023-01-01", "2023-01-14", "2023-01-26",
            "2023-03-22", "2023-04-07", "2023-04-14",
            "2023-05-01", "2023-08-15", "2023-09-19",
            "2023-10-14", "2023-10-02", "2023-10-23",
            "2023-10-24", "2023-11-12", "2023-11-01",
            "2023-12-25",
        ]),
    }
)

# Duplicate for 2024 (shift by 1 year)
_h24 = BENGALURU_HOLIDAYS.copy()
_h24["ds"] = _h24["ds"] + pd.DateOffset(years=1)
BENGALURU_HOLIDAYS = pd.concat([BENGALURU_HOLIDAYS, _h24], ignore_index=True)

MODEL_DIR = Path("models/saved/prophet")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_RECORDS_FOR_JUNCTION_MODEL = 30
UNCERTAINTY_SAMPLES = 500  # prophet samples for CI bands


# ---------------------------------------------------------------------------
# Helper: aggregate feature_store to junction-level hourly series
# ---------------------------------------------------------------------------

def _load_junction_series(feature_store_path: str) -> dict[str, pd.DataFrame]:
    """
    Returns dict mapping junction_id → DataFrame with columns [ds, y, is_rush_hour].
    Uses Polars for loading then converts to pandas for Prophet.
    """
    try:
        import polars as pl
        df = pl.read_parquet(feature_store_path)
        # Ensure timestamp column is parsed
        if "created_datetime" in df.columns:
            ts_col = "created_datetime"
        elif "timestamp" in df.columns:
            ts_col = "timestamp"
        else:
            raise ValueError("No timestamp column found in feature store")

        df = df.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("ts"))
        df = df.with_columns(pl.col("ts").dt.truncate("1h").alias("hour_bucket"))

        junction_col = "junction_id_snapped" if "junction_id_snapped" in df.columns else "junction_name"

        agg = (
            df.group_by([junction_col, "hour_bucket"])
            .agg([
                pl.len().alias("y"),
                pl.col("is_rush_hour").mean().alias("is_rush_hour"),
            ])
            .sort("hour_bucket")
        )
        agg_pd = agg.to_pandas()
        agg_pd = agg_pd.rename(columns={"hour_bucket": "ds", junction_col: "junction_id"})
        agg_pd["ds"] = pd.to_datetime(agg_pd["ds"])

        result = {}
        for jid, grp in agg_pd.groupby("junction_id"):
            result[str(jid)] = grp[["ds", "y", "is_rush_hour"]].reset_index(drop=True)
        return result
    except Exception as e:
        logger.error(f"Failed to load junction series: {e}")
        raise


# ---------------------------------------------------------------------------
# Single junction model trainer
# ---------------------------------------------------------------------------

def _train_single(
    junction_id: str,
    series: pd.DataFrame,
    uncertainty_samples: int = UNCERTAINTY_SAMPLES,
) -> Optional[object]:
    """Train a Prophet model for one junction. Returns model or None on failure."""
    try:
        from prophet import Prophet  # lazy import to avoid top-level cost
    except ImportError:
        logger.error("prophet package not installed. Run: pip install prophet")
        return None

    if len(series) < MIN_RECORDS_FOR_JUNCTION_MODEL:
        logger.debug(f"Junction {junction_id}: only {len(series)} records — skipping (will use global fallback)")
        return None

    try:
        m = Prophet(
            uncertainty_samples=uncertainty_samples,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=BENGALURU_HOLIDAYS,
        )
        m.add_regressor("is_rush_hour")
        m.fit(series.rename(columns={"y": "y"}))  # Prophet expects 'ds','y'
        return m
    except Exception as e:
        logger.warning(f"Prophet fit failed for junction {junction_id}: {e}")
        return None


def _train_global(all_series: dict[str, pd.DataFrame]) -> Optional[object]:
    """Train a single global model on all junctions combined (fallback)."""
    try:
        from prophet import Prophet
    except ImportError:
        return None

    combined = pd.concat(all_series.values(), ignore_index=True)
    hourly = combined.groupby("ds").agg({"y": "sum", "is_rush_hour": "mean"}).reset_index()
    hourly = hourly.sort_values("ds")

    try:
        m = Prophet(
            uncertainty_samples=UNCERTAINTY_SAMPLES,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=BENGALURU_HOLIDAYS,
        )
        m.add_regressor("is_rush_hour")
        m.fit(hourly)
        return m
    except Exception as e:
        logger.warning(f"Global Prophet fit failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API: train all models
# ---------------------------------------------------------------------------

def train_all(
    feature_store_path: str = "data/processed/feature_store.parquet",
    n_jobs: int = -1,
) -> dict:
    """
    Train per-junction Prophet models + global fallback.
    Serialises each model to MODEL_DIR/junction_{id}.pkl.
    Returns summary dict.
    """
    logger.info("Loading junction time series from feature store...")
    junction_series = _load_junction_series(feature_store_path)
    logger.info(f"Found {len(junction_series)} junctions")

    def _train_and_save(jid: str, series: pd.DataFrame):
        model = _train_single(jid, series)
        if model is not None:
            path = MODEL_DIR / f"junction_{jid}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            return jid, True
        return jid, False

    results = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
        delayed(_train_and_save)(jid, s) for jid, s in junction_series.items()
    )

    trained = [jid for jid, ok in results if ok]
    skipped = [jid for jid, ok in results if not ok]

    logger.info(f"Trained per-junction models: {len(trained)}, skipped (sparse): {len(skipped)}")

    # Global fallback
    logger.info("Training global fallback Prophet model...")
    global_model = _train_global(junction_series)
    if global_model:
        with open(MODEL_DIR / "global_model.pkl", "wb") as f:
            pickle.dump(global_model, f)
        logger.info("Global model saved.")
    else:
        logger.warning("Global model training failed.")

    return {
        "trained_junctions": len(trained),
        "skipped_junctions": len(skipped),
        "global_model": global_model is not None,
    }


# ---------------------------------------------------------------------------
# Public API: load model
# ---------------------------------------------------------------------------

def _load_model(junction_id: str):
    """Load per-junction model, or fall back to global."""
    path = MODEL_DIR / f"junction_{junction_id}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f), "junction"
    global_path = MODEL_DIR / "global_model.pkl"
    if global_path.exists():
        with open(global_path, "rb") as f:
            return pickle.load(f), "global"
    raise FileNotFoundError(
        f"No Prophet model found for junction {junction_id} and no global fallback. "
        "Run train_all() first."
    )


# ---------------------------------------------------------------------------
# Public API: forecast
# ---------------------------------------------------------------------------

def forecast(
    junction_id: str,
    horizon_hours: int = 24,
    rush_hour_value: float = 0.0,
) -> list[dict]:
    """
    Return a list of {ts, p10, p50, p90} dicts for the next `horizon_hours` hours.

    Args:
        junction_id: Junction identifier (string).
        horizon_hours: 24 for 24h, 168 for 7d.
        rush_hour_value: Regressor value for future hours (0 or 1).

    Returns:
        List of dicts with keys: ts (ISO str), p10, p50, p90.
    """
    model, source = _load_model(str(junction_id))

    future = model.make_future_dataframe(periods=horizon_hours, freq="h", include_history=False)
    # Mark future hours as rush hour based on hour of day
    future["is_rush_hour"] = future["ds"].dt.hour.isin([7, 8, 9, 17, 18, 19, 20]).astype(float)

    forecast_df = model.predict(future)

    # Extract confidence columns
    # Prophet provides yhat_lower / yhat_upper for ~80% CI
    # We approximate P10/P90 from the uncertainty samples via simulation percentiles
    results = []
    for _, row in forecast_df.tail(horizon_hours).iterrows():
        p50 = max(0.0, float(row["yhat"]))
        # yhat_lower ≈ P10, yhat_upper ≈ P90 (Prophet 80% interval, close enough)
        p10 = max(0.0, float(row["yhat_lower"]))
        p90 = max(0.0, float(row["yhat_upper"]))
        # Enforce ordering
        p10 = min(p10, p50)
        p90 = max(p90, p50)
        results.append({
            "ts": row["ds"].isoformat(),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "source": source,
        })

    return results


def risk_calendar(junction_id: str) -> list[list[dict]]:
    """
    Return 7×4 grid of P50 risk for a given junction.
    Rows = days (Mon-Sun), Columns = shifts (Morning/Afternoon/Evening/Night).
    """
    SHIFT_HOURS = {
        "Morning": (6, 12),
        "Afternoon": (12, 17),
        "Evening": (17, 21),
        "Night": (21, 6),
    }
    forecast_7d = forecast(junction_id, horizon_hours=168)
    df = pd.DataFrame(forecast_7d)
    df["ts"] = pd.to_datetime(df["ts"])
    df["day"] = df["ts"].dt.day_name()
    df["hour"] = df["ts"].dt.hour

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    shifts = ["Morning", "Afternoon", "Evening", "Night"]

    calendar = []
    for day in days:
        row = []
        day_df = df[df["day"] == day]
        for shift in shifts:
            sh, eh = SHIFT_HOURS[shift]
            if sh < eh:
                mask = (day_df["hour"] >= sh) & (day_df["hour"] < eh)
            else:  # Night wraps midnight
                mask = (day_df["hour"] >= sh) | (day_df["hour"] < eh)
            avg_p50 = float(day_df.loc[mask, "p50"].mean()) if mask.any() else 0.0
            row.append({"day": day, "shift": shift, "p50": round(avg_p50, 2)})
        calendar.append(row)

    return calendar


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = train_all()
    print(f"\nTraining complete: {summary}")