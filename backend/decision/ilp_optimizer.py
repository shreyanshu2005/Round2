"""
backend/decision/ilp_optimizer.py
----------------------------------
Stage 1 of the patrol optimization engine: a deterministic, explainable
Integer Linear Program (ILP) that allocates a fixed officer budget across
zones to maximize total expected risk reduction.

Objective
---------
    maximize  sum_i  risk_score[i] * deterrence_factor(x[i]) * weight_i

Since deterrence_factor(x[i]) = 1 - exp(-0.5 * x[i]) is non-linear in x[i],
we linearize it by precomputing the *marginal* gain of each additional
officer at each zone (0 -> 1, 1 -> 2, ... up to max_officers_per_zone) and
modelling allocation as a sum of binary "officer slot" variables. Marginal
gains are diminishing (concave), which guarantees the greedy/ILP slot
formulation is exact for this type of separable concave objective.

Inputs
------
  - risk_score[i]        : 0-100 calibrated risk score (Layer 4 — lgbm_risk +
                            calibration.py)
  - congestion_score[i]  : 0-100 congestion impact score (Layer 6 —
                            congestion_scorer.py), used as a secondary
                            tie-break weight so structurally critical
                            junctions are favoured among equal-risk zones.

Output
------
  dict[zone_id, int]  — officers allocated per zone, sums to total_officers.

Usage
-----
  from backend.decision.ilp_optimizer import ILPOptimizer
  opt = ILPOptimizer()
  allocation, meta = opt.optimize(total_officers=20, shift="Evening", date="2024-01-15")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import polars as pl
import pulp

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]  # btip-gridlock2/
CONGESTION_PATH = REPO_ROOT / "data" / "processed" / "junction_congestion_scores.parquet"

MAX_OFFICERS_PER_ZONE = 5
DEFAULT_K = 0.5  # deterrence decay constant: deterrence = 1 - exp(-k * n)


def deterrence_factor(n_officers: int, k: float = DEFAULT_K) -> float:
    """Diminishing-returns deterrence curve. n=0 -> 0.0, n=large -> ~1.0."""
    return 1.0 - math.exp(-k * n_officers)


@dataclass
class ZoneInput:
    zone_id: str
    risk_score: float            # 0-100, from Layer 4 risk API
    congestion_score: float = 0.0  # 0-100, from Layer 6 congestion_scorer
    risk_weight: float = 0.85    # blend weight: risk vs congestion in objective
    meta: dict = field(default_factory=dict)


class ILPOptimizer:
    """
    PuLP-based ILP officer allocator.

    Decision variables: x[i] = integer officers assigned to zone i, 0 <= x[i] <= max_officers_per_zone.
    Linearized via marginal-gain binary slots so PuLP/CBC solves it as a
    pure (and exact, given concavity) 0/1 knapsack-style ILP.
    """

    def __init__(
        self,
        max_officers_per_zone: int = MAX_OFFICERS_PER_ZONE,
        deterrence_k: float = DEFAULT_K,
        congestion_path: Path = CONGESTION_PATH,
    ):
        self.max_officers_per_zone = max_officers_per_zone
        self.deterrence_k = deterrence_k
        self.congestion_path = congestion_path

    # ── Data assembly ────────────────────────────────────────────────────────

    def _load_congestion_scores(self) -> dict[str, float]:
        """zone_id -> congestion_score (0-100). Returns {} if file missing."""
        if not self.congestion_path.exists():
            logger.warning(
                "Congestion scores not found at %s — run congestion_scorer.py "
                "(Layer 6). Falling back to congestion_score=0 for all zones.",
                self.congestion_path,
            )
            return {}
        df = pl.read_parquet(str(self.congestion_path))
        return dict(zip(df["junction_id"].cast(pl.Utf8), df["congestion_score"]))

    def build_zone_inputs(
        self,
        risk_scores: dict[str, float],
        congestion_scores: Optional[dict[str, float]] = None,
    ) -> list[ZoneInput]:
        """
        Combine per-zone risk scores (required, from Layer 4) with congestion
        scores (optional, from Layer 6) into ZoneInput objects.
        """
        congestion_scores = (
            congestion_scores if congestion_scores is not None else self._load_congestion_scores()
        )
        zones = [
            ZoneInput(
                zone_id=str(zid),
                risk_score=float(rscore),
                congestion_score=float(congestion_scores.get(str(zid), 0.0)),
            )
            for zid, rscore in risk_scores.items()
        ]
        return zones

    # ── Objective weighting ──────────────────────────────────────────────────

    @staticmethod
    def _zone_weight(zone: ZoneInput) -> float:
        """
        Blended priority weight: mostly risk_score, with congestion_score as
        a tie-break/amplifier for structurally important junctions.
        """
        return (
            zone.risk_weight * zone.risk_score
            + (1.0 - zone.risk_weight) * zone.congestion_score
        )

    # ── ILP solve ─────────────────────────────────────────────────────────────

    def optimize(
        self,
        zones: list[ZoneInput],
        total_officers: int,
    ) -> tuple[dict[str, int], dict]:
        """
        Solve the ILP.

        Parameters
        ----------
        zones : list[ZoneInput] — candidate zones with risk/congestion scores
        total_officers : int — patrol budget for this shift

        Returns
        -------
        (allocation, meta)
          allocation : {zone_id: n_officers}, sums to <= total_officers
                       (== total_officers whenever enough zones/slots exist)
          meta       : solver status, objective value, per-zone risk reduction
        """
        if not zones:
            return {}, {"status": "no_zones", "objective": 0.0}

        prob = pulp.LpProblem("BTIP_Patrol_Allocation", pulp.LpMaximize)

        # Binary "slot" variables: slot[i][s] = 1 if zone i gets its s-th officer
        # (s = 1..max_officers_per_zone). Marginal value of slot s at zone i:
        #   weight_i * (deterrence(s) - deterrence(s-1))
        # which is positive and strictly decreasing in s (concavity), so an
        # optimal ILP solution will always fill lower slots before higher ones.
        slot_vars: dict[tuple[str, int], pulp.LpVariable] = {}
        marginal_value: dict[tuple[str, int], float] = {}

        for zone in zones:
            w = self._zone_weight(zone)
            prev_det = 0.0
            for s in range(1, self.max_officers_per_zone + 1):
                det = deterrence_factor(s, self.deterrence_k)
                marginal = w * (det - prev_det)
                prev_det = det
                var = pulp.LpVariable(f"slot_{zone.zone_id}_{s}", cat="Binary")
                slot_vars[(zone.zone_id, s)] = var
                marginal_value[(zone.zone_id, s)] = marginal

        # Objective: maximize total marginal value collected
        prob += pulp.lpSum(
            marginal_value[k] * v for k, v in slot_vars.items()
        ), "TotalRiskReduction"

        # Constraint 1: total officers used <= budget
        prob += (
            pulp.lpSum(slot_vars.values()) <= total_officers,
            "OfficerBudget",
        )

        # Constraint 2: ordering — slot s can only be used if slot s-1 is used
        # (enforces "fill lower slots first", though concavity already implies
        # this is optimal; constraint guards against degenerate ties).
        for zone in zones:
            for s in range(2, self.max_officers_per_zone + 1):
                prob += (
                    slot_vars[(zone.zone_id, s)] <= slot_vars[(zone.zone_id, s - 1)],
                    f"Order_{zone.zone_id}_{s}",
                )

        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            logger.warning(
                "ILP solver returned status=%s (expected Optimal). "
                "Relaxing: falling back to greedy allocation.",
                status,
            )
            return self._greedy_fallback(zones, total_officers), {
                "status": f"fallback_greedy ({status})",
                "objective": None,
            }

        allocation: dict[str, int] = {zone.zone_id: 0 for zone in zones}
        for (zone_id, s), var in slot_vars.items():
            if var.value() and var.value() > 0.5:
                allocation[zone_id] += 1

        objective_value = pulp.value(prob.objective)

        meta = {
            "status": status,
            "objective": objective_value,
            "total_officers_allocated": sum(allocation.values()),
            "deterrence_k": self.deterrence_k,
        }
        logger.info(
            "ILP solved: status=%s, officers_allocated=%d/%d, objective=%.2f",
            status, sum(allocation.values()), total_officers, objective_value or 0.0,
        )
        return allocation, meta

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _greedy_fallback(self, zones: list[ZoneInput], total_officers: int) -> dict[str, int]:
        """
        Greedy marginal-value allocator used only if the ILP solver fails to
        reach Optimal (e.g. CBC unavailable). Since the objective is
        separable and concave, greedy-by-marginal-value is itself optimal —
        this is a safe, demo-ready fallback, not an approximation.
        """
        allocation = {zone.zone_id: 0 for zone in zones}
        candidates = []
        for zone in zones:
            w = self._zone_weight(zone)
            prev_det = 0.0
            for s in range(1, self.max_officers_per_zone + 1):
                det = deterrence_factor(s, self.deterrence_k)
                candidates.append((w * (det - prev_det), zone.zone_id))
                prev_det = det
        candidates.sort(key=lambda c: c[0], reverse=True)

        remaining = total_officers
        for _, zone_id in candidates:
            if remaining <= 0:
                break
            if allocation[zone_id] >= self.max_officers_per_zone:
                continue
            allocation[zone_id] += 1
            remaining -= 1
        return allocation

    # ── Convenience: end-to-end from raw risk dict ───────────────────────────

    def allocate(
        self,
        risk_scores: dict[str, float],
        total_officers: int,
        congestion_scores: Optional[dict[str, float]] = None,
    ) -> tuple[dict[str, int], dict]:
        """One-shot: build zone inputs from risk_scores dict, then optimize."""
        zones = self.build_zone_inputs(risk_scores, congestion_scores)
        return self.optimize(zones, total_officers)


# ── CLI entry-point / smoke test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Smoke test with synthetic risk scores (use real /risk API output in prod)
    fake_risk = {f"J{i}": float(20 + (i * 7) % 80) for i in range(1, 21)}
    opt = ILPOptimizer()
    allocation, meta = opt.allocate(fake_risk, total_officers=20)

    print("\n── ILP allocation (synthetic smoke test) ──────────────")
    for zone_id, n in sorted(allocation.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"  {zone_id}: {n} officers  (risk={fake_risk[zone_id]:.1f})")
    print(f"\nMeta: {meta}")
    assert sum(allocation.values()) == 20, "ILP allocation must sum to total_officers"
    print("OK — allocation sums to total_officers.")