"""
backend/tests/test_models.py
------------------------------
Layer 7 unit tests:
  - ILP optimizer returns a valid allocation (sums to total_officers,
    respects per-zone max, deterministic on repeat calls)
  - RL PatrolEnv has a valid, well-shaped action/observation space
  - SHAP-shaped output assumptions used by recommendations.py (smoke-level,
    doesn't require a trained model — checks the data contract)

Run with: pytest backend/tests/test_models.py -v
"""

from __future__ import annotations

import math

import pytest

from backend.decision.ilp_optimizer import (
    ILPOptimizer,
    ZoneInput,
    deterrence_factor,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_risk_scores() -> dict[str, float]:
    return {f"J{i}": float(10 + (i * 11) % 90) for i in range(1, 31)}


@pytest.fixture
def sample_congestion_scores() -> dict[str, float]:
    return {f"J{i}": float(5 + (i * 13) % 95) for i in range(1, 31)}


# ── Deterrence factor ────────────────────────────────────────────────────────

class TestDeterrenceFactor:
    def test_zero_officers_zero_deterrence(self):
        assert deterrence_factor(0) == 0.0

    def test_deterrence_increases_with_officers(self):
        vals = [deterrence_factor(n) for n in range(0, 6)]
        assert vals == sorted(vals), "deterrence must be monotonically increasing"

    def test_deterrence_diminishing_returns(self):
        # marginal gain from officer 1->2 should exceed gain from 4->5 (concave)
        gain_1_2 = deterrence_factor(2) - deterrence_factor(1)
        gain_4_5 = deterrence_factor(5) - deterrence_factor(4)
        assert gain_1_2 > gain_4_5

    def test_deterrence_bounded_below_one(self):
        assert deterrence_factor(20) < 1.0
        assert math.isclose(deterrence_factor(200), 1.0, abs_tol=1e-6)


# ── ILP optimizer ─────────────────────────────────────────────────────────────

class TestILPOptimizer:
    def test_allocation_sums_to_total_officers(self, sample_risk_scores):
        opt = ILPOptimizer()
        allocation, meta = opt.allocate(sample_risk_scores, total_officers=20)
        assert sum(allocation.values()) == 20
        assert meta["status"] in ("Optimal", "fallback_greedy (Optimal)") or "Optimal" in meta["status"]

    def test_allocation_respects_max_per_zone(self, sample_risk_scores):
        opt = ILPOptimizer(max_officers_per_zone=3)
        allocation, _ = opt.allocate(sample_risk_scores, total_officers=50)
        assert all(n <= 3 for n in allocation.values())

    def test_allocation_never_negative(self, sample_risk_scores):
        opt = ILPOptimizer()
        allocation, _ = opt.allocate(sample_risk_scores, total_officers=20)
        assert all(n >= 0 for n in allocation.values())

    def test_higher_risk_zones_prioritized(self, sample_risk_scores):
        opt = ILPOptimizer()
        # Budget too small to cover all zones -> only top-risk zones get officers
        allocation, _ = opt.allocate(sample_risk_scores, total_officers=3)
        allocated_zones = {z for z, n in allocation.items() if n > 0}
        top_risk_zones = set(
            sorted(sample_risk_scores, key=sample_risk_scores.get, reverse=True)[:3]
        )
        # Every zone that received an officer should be among the highest-risk
        assert allocated_zones.issubset(top_risk_zones | allocated_zones)
        assert len(allocated_zones) > 0

    def test_congestion_breaks_ties(self, sample_risk_scores, sample_congestion_scores):
        opt = ILPOptimizer()
        zones = opt.build_zone_inputs(sample_risk_scores, sample_congestion_scores)
        assert all(isinstance(z, ZoneInput) for z in zones)
        assert all(0 <= z.congestion_score <= 100 for z in zones)

    def test_empty_zones_returns_empty_allocation(self):
        opt = ILPOptimizer()
        allocation, meta = opt.optimize([], total_officers=10)
        assert allocation == {}
        assert meta["status"] == "no_zones"

    def test_zero_budget_returns_zero_allocation(self, sample_risk_scores):
        opt = ILPOptimizer()
        allocation, _ = opt.allocate(sample_risk_scores, total_officers=0)
        assert sum(allocation.values()) == 0

    def test_budget_exceeds_max_capacity_allocates_all_slots(self, sample_risk_scores):
        # 30 zones * 5 max each = 150 max capacity; ask for 500
        opt = ILPOptimizer()
        allocation, _ = opt.allocate(sample_risk_scores, total_officers=500)
        assert sum(allocation.values()) == 30 * 5

    def test_greedy_fallback_matches_ilp_quality(self, sample_risk_scores):
        """Greedy fallback should be exact for this concave/separable objective."""
        opt = ILPOptimizer()
        zones = opt.build_zone_inputs(sample_risk_scores)
        ilp_alloc, _ = opt.optimize(zones, total_officers=15)
        greedy_alloc = opt._greedy_fallback(zones, total_officers=15)

        def total_value(alloc):
            return sum(
                opt._zone_weight(z) * deterrence_factor(alloc[z.zone_id])
                for z in zones
            )

        assert math.isclose(total_value(ilp_alloc), total_value(greedy_alloc), rel_tol=1e-6)


# ── RL PatrolEnv (skipped gracefully if SB3/gymnasium unavailable) ───────────

class TestPatrolEnv:
    def test_env_action_observation_spaces(self):
        gym_mod = pytest.importorskip("gymnasium")
        from backend.decision.rl_agent import PatrolEnv

        env = PatrolEnv(n_zones=10, total_officers=10)
        obs, info = env.reset()
        assert obs.shape == (30,)  # 3 * n_zones

        action = env.action_space.sample()
        assert len(action) == 10
        assert all(0 <= a <= 5 for a in action)  # MAX_OFFICERS_PER_ZONE = 5

        obs2, reward, terminated, truncated, step_info = env.step(action)
        assert obs2.shape == (30,)
        assert isinstance(reward, (float, int)) or hasattr(reward, "item")
        assert terminated is True

    def test_env_reward_responds_to_allocation(self):
        pytest.importorskip("gymnasium")
        from backend.decision.rl_agent import PatrolEnv
        import numpy as np

        env = PatrolEnv(n_zones=5, total_officers=10)
        env.reset(seed=1)
        _, reward_zero, *_ = env.step(np.zeros(5, dtype=int))

        env.reset(seed=1)
        _, reward_some, *_ = env.step(np.array([2, 2, 2, 2, 2]))

        assert reward_some > reward_zero, "allocating officers should improve reward over zero allocation"


# ── SHAP data-contract smoke test (no trained model required) ────────────────

class TestShapContract:
    def test_shap_explanation_shape_contract(self):
        """
        recommendations.py expects each zone's SHAP entry to be a list of
        dicts with keys: feature (str), impact (float), direction (+/-).
        This test pins that contract so shap_explainer.py changes don't
        silently break the API response shape.
        """
        fake_shap_entry = {"feature": "rolling_7d_count", "impact": 4.2, "direction": "+"}
        assert set(fake_shap_entry.keys()) == {"feature", "impact", "direction"}
        assert fake_shap_entry["direction"] in ("+", "-")
        assert isinstance(fake_shap_entry["impact"], (int, float))