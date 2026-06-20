"""
backend/simulation/graph_diffusion.py
----------------------------------------
Propagates congestion relief from a directly patrolled zone to its
neighboring junctions in the OSM road graph (Layer 6's
`graph_intelligence.py`). Relief decays by hop and by edge weight (inverse
road distance), so patrolling one busy junction also relieves nearby ones —
not just a flat multiplier.

Propagation rule (per Build Guide)
-----------------------------------
  relief_at_neighbor = direct_reduction * edge_weight * 0.3   (per hop)
  decays by 70% per additional hop (i.e. multiply by 0.3 again)

So for a 2-hop neighbor: relief = direct_reduction * edge_weight_1 * 0.3 * edge_weight_2 * 0.3

edge_weight comes from `GraphIntelligence.edge_weight(u, v)` (Layer 6):
  edge_weight = clip(1 - length_m / 2000, 0, 1)   — closer = stronger connection

Usage
-----
  from backend.simulation.graph_diffusion import GraphDiffusion
  gd = GraphDiffusion()
  spillover = gd.propagate(patrolled_zone="J11", direct_reduction=0.63, max_hops=2)
  # -> {"J12": 0.041, "J15": 0.018, ...}
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

HOP_DECAY = 0.3       # relief multiplier applied per hop (70% decay per hop)
DEFAULT_MAX_HOPS = 2


class GraphDiffusion:
    """
    Wraps Layer 6's `GraphIntelligence` to propagate direct violation-
    reduction effects from patrolled zones to their graph neighbors.
    """

    def __init__(self, graph_intelligence=None, hop_decay: float = HOP_DECAY):
        """
        Parameters
        ----------
        graph_intelligence : Optional[GraphIntelligence]
            Pass an already-loaded instance to avoid reloading the OSM graph
            for every simulation call (it's expensive — see Layer 6 notes).
            If None, lazily constructs and loads one on first use.
        """
        self._gi = graph_intelligence
        self.hop_decay = hop_decay

    @property
    def gi(self):
        if self._gi is None:
            from backend.decision.graph_intelligence import GraphIntelligence
            self._gi = GraphIntelligence()
            self._gi.load_graph()
        return self._gi

    # ── Core propagation ─────────────────────────────────────────────────────

    def propagate(
        self,
        patrolled_zone: str,
        direct_reduction: float,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> dict[str, float]:
        """
        Propagate relief from a single patrolled zone outward.

        Returns
        -------
        dict[neighbor_zone_id, relief_fraction] — relief received by each
        affected neighbor within max_hops. Does NOT include the patrolled
        zone itself (its effect is `direct_reduction`, applied separately).
        """
        if direct_reduction <= 0:
            return {}

        try:
            if patrolled_zone not in self.gi.G:
                logger.debug(
                    "Zone %s not found in OSM graph — no spillover propagated.",
                    patrolled_zone,
                )
                return {}
        except FileNotFoundError:
            logger.warning(
                "OSM graph unavailable — graph diffusion disabled for this "
                "simulation run (spillover = 0 for all neighbors)."
            )
            return {}

        relief: dict[str, float] = {}
        visited = {patrolled_zone}
        frontier = [(patrolled_zone, direct_reduction, 0)]  # (node, incoming_relief, hop)

        while frontier:
            node, incoming_relief, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            for neighbor in self.gi.G.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                edge_w = self.gi.edge_weight(node, neighbor)
                if edge_w <= 0:
                    continue
                neighbor_relief = incoming_relief * edge_w * self.hop_decay
                if neighbor_relief <= 1e-4:
                    continue
                relief[neighbor] = relief.get(neighbor, 0.0) + neighbor_relief
                frontier.append((neighbor, neighbor_relief, hop + 1))

        return relief

    def propagate_multi(
        self,
        zone_reductions: dict[str, float],
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> dict[str, float]:
        """
        Propagate relief from multiple simultaneously patrolled zones and
        sum overlapping spillover at shared neighbors.

        Parameters
        ----------
        zone_reductions : {zone_id: direct_reduction_fraction} for every
                           zone that received officers this shift.

        Returns
        -------
        dict[zone_id, total_spillover_relief] — aggregated across all
        patrolled zones. Zones that are themselves directly patrolled are
        excluded (their effect is the direct reduction, not spillover).
        """
        total_relief: dict[str, float] = {}
        for zone_id, reduction in zone_reductions.items():
            spillover = self.propagate(zone_id, reduction, max_hops=max_hops)
            for neighbor, relief in spillover.items():
                if neighbor in zone_reductions:
                    continue  # directly patrolled — don't double count as spillover
                total_relief[neighbor] = total_relief.get(neighbor, 0.0) + relief

        # Cap aggregated relief at 1.0 (100% reduction) per zone — physically
        # sensible upper bound even with overlapping spillover from many zones.
        return {z: min(r, 1.0) for z, r in total_relief.items()}


# ── CLI entry-point / smoke test ────────────────────────────────────────────

if __name__ == "__main__":
    import networkx as nx

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Build a tiny synthetic graph for a smoke test (no need for the real
    # 80K-node Bengaluru OSM graph here).
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
    spillover = gd.propagate("J1", direct_reduction=0.6, max_hops=2)

    print("\n── Spillover from patrolling J1 (direct_reduction=0.6) ──")
    for zone, relief in spillover.items():
        print(f"  {zone}: {relief * 100:.2f}% relief")

    assert "J2" in spillover and "J4" in spillover
    assert spillover["J2"] > spillover.get("J3", 0), "1-hop relief should exceed 2-hop relief"
    print("\nOK — graph diffusion propagates and decays correctly by hop.")