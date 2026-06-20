"""
scripts/train_all_models.py
End-to-end BTIP training pipeline: preprocessing -> HDBSCAN -> LightGBM ->
XGBoost -> Prophet -> LSTM, run sequentially with progress logging.

Each step is wrapped so a failure in one model does NOT crash the whole
pipeline — it's logged and the script continues to the next step, then
reports a final summary of what succeeded/failed.
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("btip.train_all")


@dataclass
class StepResult:
    name: str
    success: bool
    duration_s: float
    error: str = ""


@dataclass
class Pipeline:
    results: List[StepResult] = field(default_factory=list)

    def run_step(self, name: str, fn: Callable[[], None]) -> None:
        logger.info("▶ Starting: %s", name)
        start = time.time()
        try:
            fn()
            duration = time.time() - start
            logger.info("✔ Completed: %s (%.1fs)", name, duration)
            self.results.append(StepResult(name, True, duration))
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            duration = time.time() - start
            logger.error("✘ Failed: %s (%.1fs) — %s", name, duration, exc)
            self.results.append(StepResult(name, False, duration, str(exc)))

    def summary(self) -> str:
        lines = ["", "=" * 60, "TRAINING PIPELINE SUMMARY", "=" * 60]
        for r in self.results:
            status = "OK" if r.success else "FAILED"
            lines.append(f"  [{status:6}] {r.name:35} ({r.duration_s:.1f}s)")
            if not r.success:
                lines.append(f"           ↳ {r.error}")
        n_failed = sum(1 for r in self.results if not r.success)
        lines.append("=" * 60)
        lines.append(f"{len(self.results)} steps run, {n_failed} failed")
        lines.append("=" * 60)
        return "\n".join(lines)


def step_preprocessing():
    from scripts.build_feature_store import main as build_feature_store

    build_feature_store()


def step_hdbscan():
    from backend.models.clustering.hdbscan_model import train_hdbscan

    train_hdbscan()


def step_lightgbm():
    from backend.models.risk.lgbm_risk import train_lgbm

    train_lgbm()


def step_xgboost():
    from backend.models.risk.xgb_challenger import train_xgb

    train_xgb()


def step_prophet():
    from backend.models.forecasting.prophet_forecast import train_prophet_all_junctions

    train_prophet_all_junctions()


def step_lstm():
    from backend.models.forecasting.lstm_forecast import train_lstm_top20

    train_lstm_top20()


def main():
    pipeline = Pipeline()
    pipeline.run_step("Preprocessing / Feature Store", step_preprocessing)
    pipeline.run_step("HDBSCAN Clustering", step_hdbscan)
    pipeline.run_step("LightGBM Risk Model", step_lightgbm)
    pipeline.run_step("XGBoost Challenger", step_xgboost)
    pipeline.run_step("Prophet Forecasting", step_prophet)
    pipeline.run_step("LSTM Forecasting (top-20)", step_lstm)

    print(pipeline.summary())

    if any(not r.success for r in pipeline.results):
        sys.exit(1)


if __name__ == "__main__":
    main()