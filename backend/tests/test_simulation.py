"""
backend/tests/test_simulation.py
-----------------------------------
Layer 8 unit tests:
  - deterrence_model: monotonicity, time decay, fallback to default k
  - graph_diffusion: hop decay, 2-hop propagation, zero-reduction edge case
  - digital_twin: P10<P50<P90 confidence band, monotonicity (more officers
    -> more reduction), < 3s latency, total-officers limit enforced

Run with: pytest backend/tests/test_simulation.py -v
"""

from __future__ import annotations

import networkx as nx
import pytest

from backend.simulation.deterrence_model import DeterrenceModel, DeterrenceParams
from backend.simulation.digital_twin import DigitalTwin
from backend.simulation.graph_diffusion import GraphDiffusion


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _FakeGraphIntelligence:
    """Lightweight stand-in for Layer 6's GraphIntelligence, no OSM file needed."""

    def __init__(self, G: nx.Graph):
        self.G = G

    def edge_weight(self, u, v):
        if not self.G.has_edge(u, v):
            return 0.0
        length = self.G[u][v].get("length", 100.0)
        return max(0.0, min(1.0, 1.0 - length / 2000.0))


@pytest.fixture
def synthetic_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge("J1", "J2", length=300.0)
    G.add_edge("J2", "J3", length=500.0)
    G.add_edge("J1", "J4", length=1000.0)
    G.add_edge("J4", "J5", length=1500.0)
    return G


@pytest.fixture
def graph_diffusion(synthetic_graph) -> GraphDiffusion:
    return GraphDiffusion(graph_intelligence=_FakeGraphIntelligence(synthetic_graph))


@pytest.fixture
def deterrence_model() -> DeterrenceModel:
    return DeterrenceModel(DeterrenceParams(k=0.5, base_rate=1.0, half_life_hours=2.0))


@pytest.fixture
def synthetic_baseline() -> dict:
    return {
        "J1": {"violation_rate": 40.0, "congestion_score": 70.0},
        "J2": {"violation_rate": 25.0, "congestion_score": 55.0},
        "J3": {"violation_rate": 15.0, "congestion_score": 30.0},
        "J4": {"violation_rate": 20.0, "congestion_score": 40.0},
        "J5": {"violation_rate": 10.0, "congestion_score": 20.0},
    }


@pytest.fixture
def twin(deterrence_model, graph_diffusion, synthetic_baseline) -> DigitalTwin:
    t = DigitalTwin(deterrence_model=deterrence_model, graph_diffusion=graph_diffusion)
    t._load_baseline_state = lambda: synthetic_baseline
    return t


# ── Deterrence model ──────────────────────────────────────────────────────────

class TestDeterrenceModel:
    def test_zero_officers_zero_reduction(self, deterrence_model):
        assert deterrence_model.reduction_pct(0) == 0.0

    def test_reduction_monotonic_in_officers(self, deterrence_model):
        vals = [deterrence_model.reduction_pct(n) for n in range(0, 6)]
        assert vals == sorted(vals)

    def test_time_decay_reduces_effect(self, deterrence_model):
        base = deterrence_model.reduction_pct(3)
        decayed = deterrence_model.time_decayed_effect(base, hours_since_present=2.0)
        assert decayed < base
        assert decayed == pytest.approx(base * 0.5, rel=1e-6)  # half-life = 2h

    def test_zero_hours_no_decay(self, deterrence_model):
        base = deterrence_model.reduction_pct(3)
        assert deterrence_model.time_decayed_effect(base, hours_since_present=0) == base

    def test_fit_k_falls_back_gracefully_without_data(self):
        dm = DeterrenceModel()
        params = dm.fit_k(feature_store_path=__file__ and __import__("pathlib").Path("/nonexistent.parquet"))
        assert params.fitted_from_data is False
        assert params.k > 0


# ── Graph diffusion ───────────────────────────────────────────────────────────

class TestGraphDiffusion:
    def test_propagate_reaches_1_and_2_hop_neighbors(self, graph_diffusion):
        relief = graph_diffusion.propagate("J1", direct_reduction=0.6, max_hops=2)
        assert "J2" in relief  # 1-hop
        assert "J3" in relief  # 2-hop (via J2)
        assert "J4" in relief  # 1-hop

    def test_relief_decays_with_hops(self, graph_diffusion):
        relief = graph_diffusion.propagate("J1", direct_reduction=0.6, max_hops=2)
        assert relief["J2"] > relief["J3"]  # 1-hop > 2-hop

    def test_zero_reduction_yields_no_spillover(self, graph_diffusion):
        relief = graph_diffusion.propagate("J1", direct_reduction=0.0, max_hops=2)
        assert relief == {}

    def test_unknown_zone_yields_no_spillover(self, graph_diffusion):
        relief = graph_diffusion.propagate("J999", direct_reduction=0.6, max_hops=2)
        assert relief == {}

    def test_propagate_multi_excludes_directly_patrolled_zones(self, graph_diffusion):
        zone_reductions = {"J1": 0.6, "J2": 0.4}
        relief = graph_diffusion.propagate_multi(zone_reductions, max_hops=2)
        assert "J1" not in relief
        assert "J2" not in relief

    def test_propagate_multi_caps_relief_at_one(self, graph_diffusion):
        zone_reductions = {"J1": 0.99, "J4": 0.99}
        relief = graph_diffusion.propagate_multi(zone_reductions, max_hops=2)
        assert all(r <= 1.0 for r in relief.values())


# ── Digital twin ──────────────────────────────────────────────────────────────

class TestDigitalTwin:
    def test_confidence_band_ordering(self, twin):
        result = twin.run_scenario(zone_allocations={"J1": 3}, shift="Evening", date="2024-01-15")
        band = result["confidence_band"]
        assert band["p10"] <= band["p50"] <= band["p90"]

    def test_simulation_runs_under_3_seconds(self, twin):
        result = twin.run_scenario(zone_allocations={"J1": 3}, shift="Evening", date="2024-01-15")
        assert result["latency_seconds"] < 3.0

    def test_more_officers_never_decreases_reduction(self, twin):
        result_low = twin.run_scenario(
            zone_allocations={"J1": 1}, shift="Evening", date="2024-01-15", run_confidence_band=False
        )
        result_high = twin.run_scenario(
            zone_allocations={"J1": 4}, shift="Evening", date="2024-01-15", run_confidence_band=False
        )
        assert result_high["reduction_pct"] >= result_low["reduction_pct"]

    def test_total_officers_limit_enforced(self, twin):
        with pytest.raises(ValueError):
            twin.run_scenario(zone_allocations={"J1": 60}, shift="Evening", date="2024-01-15")

    def test_spillover_reaches_neighbors_not_directly_patrolled(self, twin):
        result = twin.run_scenario(
            zone_allocations={"J1": 3}, shift="Evening", date="2024-01-15", run_confidence_band=False
        )
        neighbor_rows = [j for j in result["per_junction"] if not j["is_directly_patrolled"]]
        assert len(neighbor_rows) > 0
        assert any(j["spillover_received_pct"] > 0 for j in neighbor_rows)

    def test_before_after_state_present_for_all_affected_junctions(self, twin):
        result = twin.run_scenario(
            zone_allocations={"J1": 2}, shift="Evening", date="2024-01-15", run_confidence_band=False
        )
        assert result["affected_junction_count"] == len(result["per_junction"])
        for j in result["per_junction"]:
            assert j["violation_rate_after"] <= j["violation_rate_before"]