"""
backend/decision/rl_agent.py
------------------------------
Stage 2 of the patrol optimization engine: a PPO reinforcement-learning
agent (Stable-Baselines3) that learns an *advisory* re-allocation on top of
the ILP baseline (ilp_optimizer.py). RL is bonus/exploratory — the system
must remain fully functional via ILP alone if RL training or inference
fails (see `recommendations.py`).

Environment: PatrolEnv (custom gym.Env)
----------------------------------------
  state  = concat([risk_scores (n_zones), congestion_scores (n_zones),
                    current_allocation (n_zones)])           shape: (3*n_zones,)
  action = MultiDiscrete([max_officers+1] * n_zones)          per-zone officer count
  reward = violation_reduction - congestion_penalty

Notes
-----
  - Trained against a lightweight internal simulation (a simplified version
    of the Layer 8 digital twin's deterrence model) since the full digital
    twin doesn't exist until Layer 8. This is intentional and documented —
    rl_agent.py should be re-trained or fine-tuned once Layer 8's
    digital_twin.py is available, for a more faithful reward signal.
  - For the demo: ILP is the safe, always-working path. RL is bonus —
    recommendations.py gracefully falls back to ILP-only if RL inference
    raises any exception.

Usage
-----
  from backend.decision.rl_agent import PatrolEnv, train_agent, load_agent, predict_delta

  env = PatrolEnv(n_zones=20)
  model = train_agent(env, total_timesteps=100_000)
  model.save("models/saved/rl/patrol_ppo.zip")

  # at inference time
  model = load_agent()
  delta = predict_delta(model, risk_scores, congestion_scores, ilp_allocation)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_MODEL_PATH = REPO_ROOT / "models" / "saved" / "rl" / "patrol_ppo.zip"

MAX_OFFICERS_PER_ZONE = 5
DETERRENCE_K = 0.5


def _deterrence(n_officers: np.ndarray, k: float = DETERRENCE_K) -> np.ndarray:
    return 1.0 - np.exp(-k * n_officers)


# ── Lazy gym / SB3 imports ────────────────────────────────────────────────────
# gym + stable-baselines3 are heavy optional deps. Import lazily so the rest
# of the backend (ILP, API routes) works fine even if RL deps aren't
# installed in a given environment.

def _import_gym():
    try:
        import gymnasium as gym
        from gymnasium import spaces
        return gym, spaces
    except ImportError:
        import gym
        from gym import spaces
        return gym, spaces


class PatrolEnv:
    """
    Custom gym Env for patrol officer allocation.

    Built lazily (subclasses gym.Env at construction time) so that importing
    this module doesn't hard-require gym/gymnasium to be installed.
    """

    def __new__(cls, *args, **kwargs):
        gym, spaces = _import_gym()

        class _PatrolEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(
                self,
                n_zones: int = 20,
                total_officers: int = 20,
                max_officers_per_zone: int = MAX_OFFICERS_PER_ZONE,
                risk_scores: Optional[np.ndarray] = None,
                congestion_scores: Optional[np.ndarray] = None,
                seed: int = 42,
            ):
                super().__init__()
                self.n_zones = n_zones
                self.total_officers = total_officers
                self.max_officers_per_zone = max_officers_per_zone
                self.rng = np.random.default_rng(seed)

                self.risk_scores = (
                    risk_scores if risk_scores is not None
                    else self.rng.uniform(10, 90, size=n_zones).astype(np.float32)
                )
                self.congestion_scores = (
                    congestion_scores if congestion_scores is not None
                    else self.rng.uniform(10, 90, size=n_zones).astype(np.float32)
                )
                self.current_allocation = np.zeros(n_zones, dtype=np.float32)

                # Observation: [risk_scores, congestion_scores, current_allocation]
                self.observation_space = spaces.Box(
                    low=0.0, high=100.0, shape=(3 * n_zones,), dtype=np.float32
                )
                # Action: officers per zone, 0..max_officers_per_zone
                self.action_space = spaces.MultiDiscrete(
                    [max_officers_per_zone + 1] * n_zones
                )

            def _get_obs(self) -> np.ndarray:
                return np.concatenate(
                    [self.risk_scores, self.congestion_scores, self.current_allocation]
                ).astype(np.float32)

            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    self.rng = np.random.default_rng(seed)
                # Resample a fresh risk/congestion scenario each episode so the
                # policy generalizes across shifts rather than memorizing one.
                self.risk_scores = self.rng.uniform(10, 90, size=self.n_zones).astype(np.float32)
                self.congestion_scores = self.rng.uniform(10, 90, size=self.n_zones).astype(np.float32)
                self.current_allocation = np.zeros(self.n_zones, dtype=np.float32)
                return self._get_obs(), {}

            def step(self, action: np.ndarray):
                allocation = np.clip(
                    np.asarray(action, dtype=np.float32), 0, self.max_officers_per_zone
                )

                # Budget penalty: discourage exceeding total_officers, but allow
                # the agent to explore (soft constraint, not a hard mask) —
                # ILP already guarantees an exact-budget baseline; RL is advisory.
                officers_used = allocation.sum()
                budget_penalty = max(0.0, officers_used - self.total_officers) * 5.0

                det = _deterrence(allocation)
                violation_reduction = float(np.sum(self.risk_scores * det))

                # Congestion penalty: zones with high congestion but low
                # allocation continue to congest. Penalize unaddressed
                # congestion-weighted risk.
                unaddressed = self.congestion_scores * (1.0 - det)
                congestion_penalty = float(np.sum(unaddressed)) * 0.1

                reward = violation_reduction - congestion_penalty - budget_penalty
                self.current_allocation = allocation

                terminated = True  # single-step (one shift) episodic decision
                truncated = False
                info = {
                    "violation_reduction": violation_reduction,
                    "congestion_penalty": congestion_penalty,
                    "officers_used": float(officers_used),
                }
                return self._get_obs(), reward, terminated, truncated, info

        return _PatrolEnv(*args, **kwargs)


def train_agent(
    n_zones: int = 20,
    total_officers: int = 20,
    total_timesteps: int = 100_000,
    save_path: Path = RL_MODEL_PATH,
):
    """
    Train a PPO agent on PatrolEnv and save it.

    Requires stable-baselines3. Raises ImportError with a clear message if
    not installed — caller (e.g. a training script) should catch and skip
    gracefully for environments without RL deps.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 is required for RL training. "
            "Install with: pip install stable-baselines3 gymnasium"
        ) from e

    env = PatrolEnv(n_zones=n_zones, total_officers=total_officers)
    try:
        check_env(env, warn=True)
    except Exception as e:
        logger.warning("Gym env check raised a warning: %s", e)

    model = PPO("MlpPolicy", env, verbose=1, seed=42)
    logger.info("Training PPO agent for %d timesteps …", total_timesteps)
    model.learn(total_timesteps=total_timesteps)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("RL agent saved → %s", save_path)
    return model


def load_agent(load_path: Path = RL_MODEL_PATH):
    """Load a trained PPO agent. Raises FileNotFoundError if not trained yet."""
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError("stable-baselines3 is required to load the RL agent.") from e

    if not load_path.exists():
        raise FileNotFoundError(
            f"No trained RL agent at {load_path}. Run train_agent() first, "
            "or rely on ILP-only recommendations (graceful fallback)."
        )
    return PPO.load(str(load_path))


def predict_delta(
    model,
    risk_scores: dict[str, float],
    congestion_scores: dict[str, float],
    ilp_allocation: dict[str, int],
    max_officers_per_zone: int = MAX_OFFICERS_PER_ZONE,
) -> dict[str, int]:
    """
    Run inference with a trained PPO agent and return an "advisory_delta":
    the difference between the RL-suggested allocation and the ILP baseline,
    per zone. Positive = RL suggests more officers than ILP; negative = fewer.

    This is advisory only — never used to directly override ILP allocation
    in the API response (see recommendations.py).
    """
    zone_ids = list(risk_scores.keys())
    n_zones = len(zone_ids)

    risk_arr = np.array([risk_scores[z] for z in zone_ids], dtype=np.float32)
    cong_arr = np.array([congestion_scores.get(z, 0.0) for z in zone_ids], dtype=np.float32)
    alloc_arr = np.array([ilp_allocation.get(z, 0) for z in zone_ids], dtype=np.float32)

    obs = np.concatenate([risk_arr, cong_arr, alloc_arr]).astype(np.float32)

    action, _ = model.predict(obs, deterministic=True)
    rl_allocation = np.clip(action, 0, max_officers_per_zone)

    delta = {
        zone_ids[i]: int(rl_allocation[i] - alloc_arr[i]) for i in range(n_zones)
    }
    return delta


# ── CLI entry-point / smoke test ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    print("Smoke-testing PatrolEnv (no training, just reset/step) …")
    try:
        env = PatrolEnv(n_zones=10, total_officers=10)
        obs, info = env.reset()
        print(f"obs shape: {obs.shape}  (expected (30,) for n_zones=10)")
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        print(f"step OK — reward={reward:.2f}, info={info}")
        assert obs.shape == (30,)
        print("OK — PatrolEnv basic interface works.")
    except ImportError as e:
        print(f"SKIPPED — RL deps not installed: {e}")