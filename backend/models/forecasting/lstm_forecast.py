"""
lstm_forecast.py
-----------------
Multi-junction LSTM for the top-20 highest-volume junctions.
Uses quantile (pinball) loss to produce P10/P50/P90 confidence bands.
Falls back to Prophet for junctions not in the top-20.

Architecture:
  - 2-layer LSTM, hidden_size=128
  - Lookback window: 14 days (336 hourly steps)
  - Output heads: 3 quantiles × horizon steps
  - Quantile regression via pinball loss
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/saved/lstm/lstm_model.pt")
SCALER_PATH = Path("models/saved/lstm/scaler.pkl")
TOP20_PATH = Path("models/saved/lstm/top20_junctions.pkl")

LOOKBACK = 336       # 14 days × 24h
HIDDEN_SIZE = 128
NUM_LAYERS = 2
QUANTILES = [0.1, 0.5, 0.9]
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
DEVICE_PREFERENCE = "cuda"  # falls back to cpu automatically


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None


def _pinball_loss(pred, target, quantiles):
    """
    Compute pinball (quantile) loss for multi-quantile output.
    pred shape: (batch, horizon, n_quantiles)
    target shape: (batch, horizon)
    """
    import torch
    loss = 0.0
    for i, q in enumerate(quantiles):
        err = target - pred[:, :, i]
        loss += torch.mean(torch.max(q * err, (q - 1) * err))
    return loss / len(quantiles)


# ---------------------------------------------------------------------------
# LSTM Model definition
# ---------------------------------------------------------------------------

def _build_model(input_size: int, horizon: int, n_quantiles: int = 3):
    """Dynamically import torch and build the LSTM model."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        raise

    class LSTMForecaster(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, horizon, n_quantiles):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0,
            )
            self.head = nn.Linear(hidden_size, horizon * n_quantiles)
            self.horizon = horizon
            self.n_quantiles = n_quantiles

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]           # (batch, hidden)
            pred = self.head(last)          # (batch, horizon * n_quantiles)
            return pred.view(-1, self.horizon, self.n_quantiles)

    return LSTMForecaster(input_size, HIDDEN_SIZE, NUM_LAYERS, horizon, n_quantiles)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_sequences(series: np.ndarray, lookback: int, horizon: int):
    """
    Create sliding window (X, y) pairs from a 1-D time series.
    X: (N, lookback), y: (N, horizon)
    """
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API: identify top-20 junctions
# ---------------------------------------------------------------------------

def get_top20_junctions(feature_store_path: str = "data/processed/feature_store.parquet") -> list[str]:
    """Return list of top-20 junction IDs by total violation count."""
    try:
        import polars as pl
        df = pl.read_parquet(feature_store_path)
        junction_col = "junction_id_snapped" if "junction_id_snapped" in df.columns else "junction_name"
        top20 = (
            df.group_by(junction_col)
            .len()
            .sort("len", descending=True)
            .head(20)[junction_col]
            .to_list()
        )
        return [str(j) for j in top20]
    except Exception as e:
        logger.warning(f"Could not compute top-20 junctions: {e}")
        return []


# ---------------------------------------------------------------------------
# Public API: train LSTM
# ---------------------------------------------------------------------------

def train(
    feature_store_path: str = "data/processed/feature_store.parquet",
    horizon_hours: int = 168,   # 7 days
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Train LSTM on top-20 junctions and save model + scaler.
    Returns training summary dict.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import MinMaxScaler
        import pickle
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise

    device = _get_device()
    logger.info(f"LSTM training on device: {device}")

    # Identify top-20
    top20 = get_top20_junctions(feature_store_path)
    logger.info(f"Top-20 junctions: {top20}")

    # Load series for top-20
    import polars as pl
    df = pl.read_parquet(feature_store_path)
    junction_col = "junction_id_snapped" if "junction_id_snapped" in df.columns else "junction_name"
    ts_col = "created_datetime" if "created_datetime" in df.columns else "timestamp"

    df = df.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("ts"))
    df = df.with_columns(pl.col("ts").dt.truncate("1h").alias("hour_bucket"))
    df = df.filter(pl.col(junction_col).cast(pl.Utf8).is_in(top20))

    agg = (
        df.group_by([junction_col, "hour_bucket"])
        .len()
        .sort("hour_bucket")
    )
    agg_pd = agg.to_pandas()
    agg_pd = agg_pd.rename(columns={"hour_bucket": "ds", junction_col: "junction_id", "len": "y"})

    # Concatenate all junctions (multi-variate approach: one series per junction stacked)
    all_X, all_y = [], []
    scaler = MinMaxScaler()

    for jid in top20:
        jdf = agg_pd[agg_pd["junction_id"] == jid].sort_values("ds")
        if len(jdf) < LOOKBACK + horizon_hours:
            logger.debug(f"Junction {jid}: insufficient history for LSTM ({len(jdf)} hours)")
            continue
        series = jdf["y"].values.astype(np.float32).reshape(-1, 1)
        series_scaled = scaler.fit_transform(series).flatten()
        X, y = _build_sequences(series_scaled, LOOKBACK, horizon_hours)
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        logger.error("No junctions had enough data for LSTM training.")
        return {"trained": False, "reason": "insufficient data"}

    X_all = np.concatenate(all_X, axis=0)   # (N, lookback)
    y_all = np.concatenate(all_y, axis=0)   # (N, horizon)

    # Add feature dim for LSTM input
    X_t = torch.tensor(X_all[:, :, None], dtype=torch.float32)  # (N, lookback, 1)
    y_t = torch.tensor(y_all, dtype=torch.float32)               # (N, horizon)

    # Train/val split (time-aware: last 20% as validation)
    split = int(0.8 * len(X_t))
    X_train, X_val = X_t[:split], X_t[split:]
    y_train, y_val = y_t[:split], y_t[split:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    model = _build_model(input_size=1, horizon=horizon_hours, n_quantiles=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = _pinball_loss(pred, yb, QUANTILES)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = _pinball_loss(val_pred, y_val.to(device), QUANTILES).item()

        avg_train = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{epochs} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"  → Saved best model (val_loss={val_loss:.4f})")

    # Save scaler and top-20 list
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(TOP20_PATH, "wb") as f:
        pickle.dump(top20, f)

    logger.info(f"LSTM training complete. Best val_loss={best_val_loss:.4f}")
    return {
        "trained": True,
        "best_val_loss": best_val_loss,
        "top20_junctions": top20,
        "samples_used": len(X_all),
    }


# ---------------------------------------------------------------------------
# Public API: is_top20 / forecast
# ---------------------------------------------------------------------------

def is_top20_junction(junction_id: str) -> bool:
    """Check if junction_id is in the top-20 list (loaded from disk)."""
    if not TOP20_PATH.exists():
        return False
    import pickle
    with open(TOP20_PATH, "rb") as f:
        top20 = pickle.load(f)
    return str(junction_id) in [str(j) for j in top20]


def forecast(
    junction_id: str,
    recent_series: np.ndarray,
    horizon_hours: int = 24,
) -> list[dict]:
    """
    Run LSTM inference for one junction.

    Args:
        junction_id: Junction identifier.
        recent_series: 1-D array of recent hourly violation counts (length >= LOOKBACK).
        horizon_hours: 24 or 168.

    Returns:
        List of dicts with keys: ts (ISO str), p10, p50, p90.
    """
    try:
        import torch
        import pickle
    except ImportError:
        raise ImportError("PyTorch required for LSTM forecast")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("LSTM model not found. Run train() first.")
    if not SCALER_PATH.exists():
        raise FileNotFoundError("Scaler not found. Run train() first.")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    device = _get_device()
    horizon_hours = min(horizon_hours, 168)  # cap at 7d

    # Load model with correct horizon
    model = _build_model(input_size=1, horizon=horizon_hours, n_quantiles=3)
    state = torch.load(MODEL_PATH, map_location=device)
    # Horizon mismatch: load with ignore for head layer
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Rebuild with stored horizon
        logger.warning("LSTM horizon mismatch — using Prophet fallback shape")
        return []

    model.to(device).eval()

    # Prepare input
    if len(recent_series) < LOOKBACK:
        # Pad with zeros on the left
        pad = np.zeros(LOOKBACK - len(recent_series), dtype=np.float32)
        recent_series = np.concatenate([pad, recent_series.astype(np.float32)])

    series_scaled = scaler.transform(recent_series[-LOOKBACK:].reshape(-1, 1)).flatten()
    x = torch.tensor(series_scaled[None, :, None], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(x)  # (1, horizon, 3)

    pred_np = pred.cpu().numpy()[0]  # (horizon, 3)

    # Inverse transform
    p10_raw = scaler.inverse_transform(pred_np[:, 0].reshape(-1, 1)).flatten()
    p50_raw = scaler.inverse_transform(pred_np[:, 1].reshape(-1, 1)).flatten()
    p90_raw = scaler.inverse_transform(pred_np[:, 2].reshape(-1, 1)).flatten()

    # Generate future timestamps
    now = pd.Timestamp.now().floor("h") + pd.Timedelta(hours=1)
    future_ts = pd.date_range(now, periods=horizon_hours, freq="h")

    results = []
    for i, ts in enumerate(future_ts):
        p10 = max(0.0, float(p10_raw[i]))
        p50 = max(0.0, float(p50_raw[i]))
        p90 = max(0.0, float(p90_raw[i]))
        # Enforce monotonic ordering
        p10 = min(p10, p50)
        p90 = max(p90, p50)
        results.append({
            "ts": ts.isoformat(),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "source": "lstm",
        })

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = train()
    print(f"\nLSTM training complete: {summary}")