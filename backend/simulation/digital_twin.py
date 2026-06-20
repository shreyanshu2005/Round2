"""
backend/simulation/digital_twin.py
-------------------------------------
Orchestrates a full what-if patrol scenario: given an officer allocation
(typically the Layer 7 ILP/RL recommendation, or a manual override from the
Simulation Lab UI), compute:

  1. Direct deterrence reduction at each patrolled zone (deterrence_model.py)
  2. Graph-diffusion spillover relief to neighboring zones (graph_diffusion.py)
  3. Before/after state for every junction (violation_rate, congestion_score)
  4. A confidence band on the total reduction % via Monte Carlo noise on the
     deterrence k parameter (+/- 10%), per the Build Guide spec.

State representation
---------------------
  {junction_id: {violation_rate, congestion_score, n_officers,
                  reduction_pct, spillover_received}}

Usage
-----
  from backend.simulation.digital_twin import DigitalTwin
  twin = DigitalTwin()
  result = twin.run_scenario(
      zone_allocations={"J11": 3, "J5": 2},
      shift="Evening",
      date="2024-01-15",
  )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from backend.simulation.deterrence_model import DeterrenceModel
from backend.simulation.graph_diffusion import GraphDiffusion

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONGESTION_PATH = REPO_ROOT / "data" / "processed" / "junction_congestion_scores.parquet"

MAX_TOTAL_OFFICERS = 50
MONTE_CARLO_RUNS = 100
K_NOISE_PCT = 0.10  # +/-10% noise on deterrence k for confidence band


class DigitalTwin:
    """
    Agent-based-style digital twin: applies deterrence + graph diffusion to
    produce a before/after projection for a 4h what-if window.
    """

    def __init__(
        self,
        deterrence_model: Optional[DeterrenceModel] = None,
        graph_diffusion: Optional[GraphDiffusion] = None,
        congestion_path: Path = CONGESTION_PATH,
    ):
        self.deterrence_model = deterrence_model or DeterrenceModel.load()
        self.graph_diffusion = graph_diffusion or GraphDiffusion()
        self.congestion_path = congestion_path

    # ── Junction baseline state ─────────────────────────────────────────────

    def _load_baseline_state(self) -> dict[str, dict]:
        """
        Baseline per-junction state before simulation: violation_rate proxy
        (rolling_7d_mean from Layer 6 congestion scores) and congestion_score.
        Falls back to an empty baseline (zones default to 0) if Layer 6
        outputs aren't available yet.
        """
        if not self.congestion_path.exists():
            logger.warning(
                "Congestion scores not found at %s — baseline state will "
                "default to 0 for unlisted zones. Run congestion_scorer.py.",
                self.congestion_path,
            )
            return {}

        df = pl.read_parquet(str(self.congestion_path))
        baseline = {}
        for row in df.to_dicts():
            zid = str(row["junction_id"])
            baseline[zid] = {
                "violation_rate": float(row.get("rolling_7d_mean", 0.0) or 0.0),
                "congestion_score": float(row.get("congestion_score", 0.0) or 0.0),
            }
        return baseline

    # ── Core scenario runner ─────────────────────────────────────────────────

    def run_scenario(
        self,
        zone_allocations: dict[str, int],
        shift: str = "Evening",
        date: Optional[str] = None,
        window_hours: int = 4,
        max_hops: int = 2,
        run_confidence_band: bool = True,
    ) -> dict:
        """
        Run a single what-if scenario.

        Parameters
        ----------
        zone_allocations : {zone_id: n_officers} — typically from Layer 7's
                            ILP/RL recommendation, or a manual UI override.
        window_hours : simulation window (Build Guide default: 4h)
        max_hops : graph diffusion hop limit (Build Guide default: 2)

        Returns
        -------
        dict with before/after per-junction state, totals, and a P10/P50/P90
        confidence band on total_reduction_pct.
        """
        start_t = time.time()

        if sum(zone_allocations.values()) > MAX_TOTAL_OFFICERS:
            raise ValueError(
                f"Total officers ({sum(zone_allocations.values())}) exceeds "
                f"the {MAX_TOTAL_OFFICERS}-officer simulation limit."
            )

        baseline = self._load_baseline_state()

        # 1. Direct deterrence reduction per patrolled zone
        direct_reduction: dict[str, float] = {
            zid: self.deterrence_model.reduction_pct(n)
            for zid, n in zone_allocations.items()
            if n > 0
        }

        # 2. Graph diffusion spillover to neighbors
        spillover = self.graph_diffusion.propagate_multi(direct_reduction, max_hops=max_hops)

        # 3. Assemble before/after state for every affected junction
        affected_zones = set(direct_reduction) | set(spillover) | set(baseline)
        per_junction = []
        total_before = 0.0
        total_after = 0.0
        total_congestion_before = 0.0
        total_congestion_after = 0.0

        for zid in affected_zones:
            base_state = baseline.get(zid, {"violation_rate": 0.0, "congestion_score": 0.0})
            violation_rate_before = base_state["violation_rate"]
            congestion_before = base_state["congestion_score"]

            n_officers = zone_allocations.get(zid, 0)
            reduction = direct_reduction.get(zid, 0.0) or spillover.get(zid, 0.0)
            spillover_received = spillover.get(zid, 0.0) if zid not in direct_reduction else 0.0

            violation_rate_after = violation_rate_before * (1.0 - reduction)
            congestion_after = congestion_before * (1.0 - reduction)

            total_before += violation_rate_before
            total_after += violation_rate_after
            total_congestion_before += congestion_before
            total_congestion_after += congestion_after

            per_junction.append({
                "junction_id": zid,
                "n_officers": n_officers,
                "violation_rate_before": round(violation_rate_before, 2),
                "violation_rate_after": round(violation_rate_after, 2),
                "congestion_score_before": round(congestion_before, 2),
                "congestion_score_after": round(congestion_after, 2),
                "reduction_pct": round(reduction * 100, 1),
                "spillover_received_pct": round(spillover_received * 100, 1),
                "is_directly_patrolled": zid in direct_reduction,
            })

        per_junction.sort(key=lambda j: j["reduction_pct"], reverse=True)

        reduction_pct = (
            (1.0 - total_after / total_before) * 100 if total_before > 0 else 0.0
        )
        congestion_improvement_pct = (
            (1.0 - total_congestion_after / total_congestion_before) * 100
            if total_congestion_before > 0 else 0.0
        )

        # 4. Confidence band via Monte Carlo noise on deterrence k
        confidence_band = (
            self._confidence_band(zone_allocations, baseline, max_hops)
            if run_confidence_band else None
        )

        elapsed = time.time() - start_t
        if elapsed > 3.0:
            logger.warning(
                "Simulation took %.2fs (> 3s target for real-time demo).", elapsed
            )

        return {
            "total_violations_before": round(total_before, 1),
            "total_violations_after": round(total_after, 1),
            "reduction_pct": round(reduction_pct, 1),
            "congestion_improvement_pct": round(congestion_improvement_pct, 1),
            "affected_junction_count": len(affected_zones),
            "confidence_band": confidence_band,
            "window_hours": window_hours,
            "shift": shift,
            "date": date,
            "per_junction": per_junction,
            "latency_seconds": round(elapsed, 3),
        }

    # ── Monte Carlo confidence band ──────────────────────────────────────────

    def _confidence_band(
        self,
        zone_allocations: dict[str, int],
        baseline: dict[str, dict],
        max_hops: int,
        n_runs: int = MONTE_CARLO_RUNS,
    ) -> dict[str, float]:
        """
        Run the scenario n_runs times with +/-10% noise on the deterrence k
        parameter, returning P10/P50/P90 of total_reduction_pct.
        """
        rng = np.random.default_rng(42)
        base_k = self.deterrence_model.params.k
        results = []

        for _ in range(n_runs):
            noisy_k = base_k * (1.0 + rng.uniform(-K_NOISE_PCT, K_NOISE_PCT))
            noisy_model = DeterrenceModel(
                params=type(self.deterrence_model.params)(
                    k=noisy_k,
                    base_rate=self.deterrence_model.params.base_rate,
                    half_life_hours=self.deterrence_model.params.half_life_hours,
                )
            )

            direct_reduction = {
                zid: noisy_model.reduction_pct(n)
                for zid, n in zone_allocations.items() if n > 0
            }
            spillover = self.graph_diffusion.propagate_multi(direct_reduction, max_hops=max_hops)

            total_before, total_after = 0.0, 0.0
            for zid in set(direct_reduction) | set(spillover) | set(baseline):
                vr_before = baseline.get(zid, {"violation_rate": 0.0})["violation_rate"]
                reduction = direct_reduction.get(zid, 0.0) or spillover.get(zid, 0.0)
                total_before += vr_before
                total_after += vr_before * (1.0 - reduction)

            run_reduction = (1.0 - total_after / total_before) * 100 if total_before > 0 else 0.0
            results.append(run_reduction)

        results_arr = np.array(results)
        p10, p50, p90 = np.percentile(results_arr, [10, 50, 90])
        return {"p10": round(float(p10), 1), "p50": round(float(p50), 1), "p90": round(float(p90), 1)}


# ── CLI entry-point / smoke test ────────────────────────────────────────────

if __name__ == "__main__":
    import networkx as nx

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Synthetic graph + baseline for a self-contained smoke test
    G = nx.Graph()
    G.add_edge("J1", "J2", length=300.0)
    G.add_edge("J2", "J3", length=500.0)
    G.add_edge("J1", "J4", length=1000.0)

    class _FakeGI:
        def __init__(self, G):
            self.G = G

        def edge_weight(self, u, v):
            if not self.G.has_edge(u, v):
                return 0.0
            length = self.G[u][v].get("length", 100.0)
            return max(0.0, min(1.0, 1.0 - length / 2000.0))

    gd = GraphDiffusion(graph_intelligence=_FakeGI(G))
    dm = DeterrenceModel()  # default k, no historical fit needed for smoke test

    twin = DigitalTwin(deterrence_model=dm, graph_diffusion=gd)
    # Patch baseline loader for the smoke test (no real congestion parquet)
    twin._load_baseline_state = lambda: {
        "J1": {"violation_rate": 40.0, "congestion_score": 70.0},
        "J2": {"violation_rate": 25.0, "congestion_score": 55.0},
        "J3": {"violation_rate": 15.0, "congestion_score": 30.0},
        "J4": {"violation_rate": 20.0, "congestion_score": 40.0},
    }

    result = twin.run_scenario(zone_allocations={"J1": 3}, shift="Evening", date="2024-01-15")

    print("\n── Digital twin scenario: 3 officers at J1 ──────────────")
    print(f"Total before: {result['total_violations_before']}")
    print(f"Total after:  {result['total_violations_after']}")
    print(f"Reduction:    {result['reduction_pct']}%")
    print(f"Congestion improvement: {result['congestion_improvement_pct']}%")
    print(f"Confidence band: {result['confidence_band']}")
    print(f"Latency: {result['latency_seconds']}s")
    for j in result["per_junction"]:
        print(f"  {j}")

    assert result["confidence_band"]["p10"] < result["confidence_band"]["p50"] < result["confidence_band"]["p90"]
    print("\nOK — P10 < P50 < P90 holds.")