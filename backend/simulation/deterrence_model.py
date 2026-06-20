"""
backend/simulation/deterrence_model.py
-----------------------------------------
Logistic deterrence-decay model: how much officer presence reduces
violations at a zone, and how that effect decays over time once officers
leave.

Two parts:
  1. Officer-count deterrence: violation_reduction_pct = base_rate * (1 - exp(-k * n_officers))
     `k` is fit from historical data where officer count varied at the same
     zone/shift (via scipy.optimize.curve_fit). Falls back to a documented
     default if there isn't enough variation in the historical data.
  2. Time decay: deterrence effect decays exponentially after officers
     leave a zone — modelled with a 2-hour half-life by default, i.e.
     effect is ~negligible after ~4h without presence.

This reuses the same `deterrence_factor` shape as Layer 7's
`ilp_optimizer.py` (k defaults match) so ILP-recommended allocations and
simulated outcomes stay consistent with each other.

Usage
-----
  from backend.simulation.deterrence_model import DeterrenceModel
  dm = DeterrenceModel()
  dm.fit_k(feature_store_df)            # optional — uses historical variation
  reduction_pct = dm.reduction_pct(n_officers=3)
  effect_at_t = dm.time_decayed_effect(reduction_pct, hours_since_present=1.5)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STORE_PATH = REPO_ROOT / "data" / "processed" / "feature_store.parquet"
SAVED_PARAMS_PATH = REPO_ROOT / "models" / "saved" / "simulation" / "deterrence_params.json"

DEFAULT_K = 0.3            # used if historical fit is infeasible / insufficient data
DEFAULT_BASE_RATE = 1.0    # base_rate left at 1.0 unless calibrated — reduction_pct is then == deterrence curve
TIME_DECAY_HALF_LIFE_HOURS = 2.0   # effect halves every 2h without officer presence


@dataclass
class DeterrenceParams:
    k: float = DEFAULT_K
    base_rate: float = DEFAULT_BASE_RATE
    half_life_hours: float = TIME_DECAY_HALF_LIFE_HOURS
    fitted_from_data: bool = False
    n_samples_used: int = 0


class DeterrenceModel:
    """
    Fits and serves the officer-count -> violation-reduction curve, plus
    time-decay of that effect once officer presence ends.
    """

    def __init__(self, params: Optional[DeterrenceParams] = None):
        self.params = params or DeterrenceParams()

    # ── Fitting from historical data ────────────────────────────────────────

    def fit_k(
        self,
        df: Optional[pl.DataFrame] = None,
        feature_store_path: Path = FEATURE_STORE_PATH,
        min_samples: int = 20,
    ) -> DeterrenceParams:
        """
        Fit k from historical (officer_count, violation_reduction) pairs.

        Requires the feature store to have an `officer_count` column (per
        zone/shift) alongside violation counts, so we can measure shifts
        where officer presence varied and observe the resulting reduction
        relative to a zero-officer baseline at the same zone.

        If `officer_count` isn't present, or there's insufficient variation
        (< min_samples usable rows), falls back to DEFAULT_K and logs why —
        this is expected and documented in the Build Guide ("Default k=0.3
        if insufficient data").
        """
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            logger.warning("scipy not installed — using default deterrence k=%.2f", DEFAULT_K)
            return self.params

        if df is None:
            if not feature_store_path.exists():
                logger.warning(
                    "Feature store not found at %s — using default deterrence k=%.2f",
                    feature_store_path, DEFAULT_K,
                )
                return self.params
            df = pl.read_parquet(str(feature_store_path))

        if "officer_count" not in df.columns:
            logger.warning(
                "'officer_count' column not in feature store — historical "
                "deterrence fit unavailable, using default k=%.2f", DEFAULT_K,
            )
            return self.params

        zone_col = "junction_id_snapped" if "junction_id_snapped" in df.columns else "junction_name"
        if zone_col not in df.columns:
            logger.warning("No junction column found — using default k=%.2f", DEFAULT_K)
            return self.params

        # Aggregate: per zone, mean violation count at officer_count == 0
        # (baseline) vs at officer_count > 0, by officer_count bucket.
        agg = (
            df.filter(pl.col("officer_count").is_not_null())
            .group_by([zone_col, "officer_count"])
            .agg(pl.len().alias("violation_count"))
        )

        baseline = (
            agg.filter(pl.col("officer_count") == 0)
            .select([zone_col, pl.col("violation_count").alias("baseline_count")])
        )

        merged = agg.filter(pl.col("officer_count") > 0).join(baseline, on=zone_col, how="inner")
        merged = merged.filter(pl.col("baseline_count") > 0).with_columns(
            (1.0 - pl.col("violation_count") / pl.col("baseline_count")).alias("observed_reduction")
        ).filter(
            (pl.col("observed_reduction") >= -0.5) & (pl.col("observed_reduction") <= 1.0)
        )

        if merged.height < min_samples:
            logger.warning(
                "Only %d usable (officer_count, reduction) samples (< min_samples=%d) "
                "— using default k=%.2f", merged.height, min_samples, DEFAULT_K,
            )
            return self.params

        n_arr = merged["officer_count"].to_numpy().astype(float)
        reduction_arr = merged["observed_reduction"].to_numpy().astype(float)

        def _curve(n, k, base_rate):
            return base_rate * (1.0 - np.exp(-k * n))

        try:
            popt, _ = curve_fit(
                _curve, n_arr, reduction_arr, p0=[DEFAULT_K, DEFAULT_BASE_RATE],
                bounds=([0.001, 0.1], [5.0, 1.0]),
            )
            fitted_k, fitted_base_rate = popt
        except Exception as e:
            logger.warning(
                "curve_fit failed (%s) — using default k=%.2f", e, DEFAULT_K,
            )
            return self.params

        self.params = DeterrenceParams(
            k=float(fitted_k),
            base_rate=float(fitted_base_rate),
            half_life_hours=TIME_DECAY_HALF_LIFE_HOURS,
            fitted_from_data=True,
            n_samples_used=int(merged.height),
        )
        logger.info(
            "Deterrence model fitted from %d historical samples: k=%.3f, base_rate=%.3f",
            merged.height, fitted_k, fitted_base_rate,
        )
        return self.params

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path = SAVED_PARAMS_PATH) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.params.__dict__, indent=2))
        logger.info("Deterrence params saved → %s", path)

    @classmethod
    def load(cls, path: Path = SAVED_PARAMS_PATH) -> "DeterrenceModel":
        import json
        if not path.exists():
            logger.info("No saved deterrence params at %s — using defaults.", path)
            return cls()
        data = json.loads(path.read_text())
        return cls(DeterrenceParams(**data))

    # ── Core curve ───────────────────────────────────────────────────────────

    def reduction_pct(self, n_officers: float) -> float:
        """
        Instantaneous violation-reduction fraction [0, base_rate] for a given
        officer count, while officers are actively present.
        """
        n_officers = max(0.0, n_officers)
        return self.params.base_rate * (1.0 - math.exp(-self.params.k * n_officers))

    def time_decayed_effect(self, reduction_pct: float, hours_since_present: float) -> float:
        """
        Exponential time decay of the deterrence effect after officers leave
        a zone. Half-life default = 2h, i.e. effect is ~negligible
        (< 10% of original) after ~6-7h, ~25% remaining after 4h.

        hours_since_present = 0  -> full effect
        hours_since_present > 0 -> decayed effect (officers no longer present)
        """
        if hours_since_present <= 0:
            return reduction_pct
        decay_factor = 0.5 ** (hours_since_present / self.params.half_life_hours)
        return reduction_pct * decay_factor

    def effect_at_time(self, n_officers: float, hours_since_present: float) -> float:
        """Convenience: combine officer-count curve + time decay in one call."""
        base = self.reduction_pct(n_officers)
        return self.time_decayed_effect(base, hours_since_present)


# ── CLI entry-point / smoke test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    dm = DeterrenceModel()
    dm.fit_k()  # will fall back to default if feature store / officer_count absent
    dm.save()

    print("\n── Deterrence curve (officers -> reduction %) ──────────")
    for n in range(0, 6):
        print(f"  {n} officers: {dm.reduction_pct(n) * 100:.1f}% reduction")

    print("\n── Time decay (3 officers, reduction decays after they leave) ──")
    base = dm.reduction_pct(3)
    for h in [0, 1, 2, 4, 6]:
        print(f"  {h}h since present: {dm.time_decayed_effect(base, h) * 100:.1f}% reduction")