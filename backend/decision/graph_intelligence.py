"""
graph_intelligence.py
---------------------
Loads the Bengaluru OSM road network graph and computes structural centrality
metrics for every junction node.

Centrality metrics computed:
  - betweenness_centrality  (k=50 approximation, weight='length')
  - closeness_centrality    (used as secondary diffusion signal)

Results cached to:  data/processed/junction_centrality.parquet

Usage
-----
  from backend.decision.graph_intelligence import GraphIntelligence
  gi = GraphIntelligence()
  gi.build()                          # compute + cache (skip if cache exists)
  df = gi.get_centrality_df()         # polars DataFrame
  score = gi.get_betweenness("node_123")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
# backend/decision/graph_intelligence.py -> parents[0]=decision, [1]=backend, [2]=repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "data" / "external" / "bengaluru_osm_graph.graphml"
CACHE_PATH = REPO_ROOT / "data" / "processed" / "junction_centrality.parquet"


class GraphIntelligence:
    """
    Wrapper around the Bengaluru OSM road network graph.

    Attributes
    ----------
    G : nx.Graph  —  undirected graph (edges carry 'length' in metres)
    _centrality_df : pl.DataFrame | None  —  cached result
    """

    def __init__(self, graph_path: Path = GRAPH_PATH, cache_path: Path = CACHE_PATH):
        self.graph_path = graph_path
        self.cache_path = cache_path
        self.G: Optional[nx.Graph] = None
        self._centrality_df: Optional[pl.DataFrame] = None

    # ── Graph loading ──────────────────────────────────────────────────────────

    def load_graph(self) -> nx.Graph:
        """Load GraphML from disk. Converts to undirected for centrality."""
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"OSM graph not found at {self.graph_path}. "
                "Run Layer 2 build_feature_store.py first to download it."
            )
        logger.info("Loading OSM graph from %s …", self.graph_path)
        G_directed = nx.read_graphml(str(self.graph_path))
        self.G = G_directed.to_undirected()

        # GraphML stores all attributes as strings — cast 'length' to float so
        # weighted algorithms (betweenness, closeness, Dijkstra) don't choke
        # on "int + str" type errors.
        bad_edges = 0
        for u, v, data in self.G.edges(data=True):
            raw_length = data.get("length", 1.0)
            try:
                data["length"] = float(raw_length)
            except (TypeError, ValueError):
                data["length"] = 1.0
                bad_edges += 1
        if bad_edges:
            logger.warning("Coerced %d edges with invalid 'length' to 1.0", bad_edges)

        logger.info(
            "Graph loaded: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )
        return self.G

    # ── Centrality computation ─────────────────────────────────────────────────

    def compute_betweenness(self, k: int = 500) -> dict[str, float]:
        """
        Approximate betweenness centrality.

        Uses k=500 random pivot nodes (Brandes approximation) which gives a
        good trade-off between accuracy and runtime on the Bengaluru graph
        (~150K nodes). Full exact computation would take 30+ minutes.

        Parameters
        ----------
        k : int
            Number of pivot nodes for approximation. Increase for accuracy,
            decrease for speed.
        """
        assert self.G is not None, "Call load_graph() first."
        logger.info("Computing betweenness centrality (k=%d) …", k)

        bc = nx.betweenness_centrality(
            self.G,
            k=min(k, self.G.number_of_nodes()),
            normalized=True,
            weight="length",
            seed=42,
        )
        logger.info("Betweenness centrality computed for %d nodes.", len(bc))
        return bc

    def compute_closeness(self) -> dict[str, float]:
        """
        Closeness centrality — average reciprocal distance to all other nodes.
        Exact computation on largest connected component only (fast enough on
        small/medium graphs). On large graphs (100K+ nodes) this is O(n*m) and
        can take tens of hours — use compute_closeness_approx() instead.
        """
        assert self.G is not None, "Call load_graph() first."
        logger.info("Computing closeness centrality …")

        # Work on the largest connected component to avoid infinite distances
        lcc = max(nx.connected_components(self.G), key=len)
        G_lcc = self.G.subgraph(lcc).copy()

        cc = nx.closeness_centrality(G_lcc, distance="length")

        # Nodes NOT in LCC get score 0.0
        full_cc: dict[str, float] = {n: 0.0 for n in self.G.nodes()}
        full_cc.update(cc)

        logger.info("Closeness centrality computed for %d nodes.", len(full_cc))
        return full_cc

    def compute_closeness_approx(self, n_landmarks: int = 200, seed: int = 42) -> dict[str, float]:
        """
        Landmark-sampling approximation of closeness centrality.

        Exact closeness requires a full Dijkstra run FROM every node (O(n*m)
        total), which is infeasible on graphs with 100K+ nodes. Instead, this
        picks `n_landmarks` random nodes and runs Dijkstra FROM each of them —
        which is exactly the same primitive, just reused n_landmarks times
        instead of n times. Since distances are symmetric in an undirected
        graph, the distance from landmark L to node V equals the distance from
        V to L, so each landmark run gives us one "sample" of every node's
        distance profile.

        For each node, closeness ≈ 1 / (mean distance to the sampled
        landmarks) — an unbiased estimator of true closeness (which is
        1 / mean distance to ALL other nodes) whose variance shrinks as
        n_landmarks grows. This is the standard approach for approximating
        closeness centrality on large networks.

        Parameters
        ----------
        n_landmarks : int
            Number of random landmark nodes to sample. More landmarks = more
            accurate but slower (cost scales linearly: ~1 Dijkstra run per
            landmark, same cost as one node's worth of exact closeness work).
        seed : int
            RNG seed for reproducible landmark selection.
        """
        assert self.G is not None, "Call load_graph() first."
        logger.info("Computing approximate closeness centrality (n_landmarks=%d) …", n_landmarks)

        # Restrict to largest connected component, same as the exact method,
        # so distances are always finite.
        lcc = max(nx.connected_components(self.G), key=len)
        G_lcc = self.G.subgraph(lcc).copy()
        lcc_nodes = list(G_lcc.nodes())

        rng = np.random.default_rng(seed)
        n_landmarks = min(n_landmarks, len(lcc_nodes))
        landmark_idx = rng.choice(len(lcc_nodes), size=n_landmarks, replace=False)
        landmarks = [lcc_nodes[i] for i in landmark_idx]

        # Running sum + count of distances per node, across all landmark runs.
        dist_sum: dict[str, float] = {n: 0.0 for n in lcc_nodes}
        dist_count: dict[str, int] = {n: 0 for n in lcc_nodes}

        for i, landmark in enumerate(landmarks, 1):
            lengths = nx.single_source_dijkstra_path_length(G_lcc, landmark, weight="length")
            for node, d in lengths.items():
                if node == landmark:
                    continue  # exclude self-distance (0.0) — not meaningful for closeness
                dist_sum[node] += d
                dist_count[node] += 1
            if i % 50 == 0 or i == n_landmarks:
                logger.info("  landmark %d/%d done", i, n_landmarks)

        approx_cc: dict[str, float] = {}
        for node in lcc_nodes:
            cnt = dist_count[node]
            if cnt == 0:
                approx_cc[node] = 0.0
            else:
                mean_dist = dist_sum[node] / cnt
                approx_cc[node] = 1.0 / mean_dist if mean_dist > 0 else 0.0

        # Nodes NOT in LCC get score 0.0
        full_cc: dict[str, float] = {n: 0.0 for n in self.G.nodes()}
        full_cc.update(approx_cc)

        logger.info("Approximate closeness centrality computed for %d nodes (%d landmarks).",
                    len(full_cc), n_landmarks)
        return full_cc

    # ── Build / cache ──────────────────────────────────────────────────────────

    def build(
        self,
        k_betweenness: int = 500,
        force_recompute: bool = False,
        use_approx_closeness: bool = True,
        closeness_landmarks: int = 200,
    ) -> pl.DataFrame:
        """
        Main entry point. If cache exists and force_recompute=False, loads from
        parquet. Otherwise computes centrality and writes cache.

        Parameters
        ----------
        k_betweenness : int
            Pivot count for approximate betweenness centrality.
        force_recompute : bool
            Recompute even if a cache file already exists.
        use_approx_closeness : bool
            If True (default), use landmark-sampling approximate closeness
            (compute_closeness_approx) — required for large graphs (100K+
            nodes) where exact closeness would take tens of hours. Set False
            only for small/medium graphs where exact closeness is fast.
        closeness_landmarks : int
            Number of landmark nodes for approximate closeness. Only used
            when use_approx_closeness=True.

        Returns
        -------
        pl.DataFrame with columns:
            node_id, betweenness_centrality, closeness_centrality,
            betweenness_norm, closeness_norm
        """
        if self.cache_path.exists() and not force_recompute:
            logger.info("Loading centrality from cache: %s", self.cache_path)
            self._centrality_df = pl.read_parquet(str(self.cache_path))
            return self._centrality_df

        # Compute fresh
        if self.G is None:
            self.load_graph()

        bc = self.compute_betweenness(k=k_betweenness)
        if use_approx_closeness:
            cc = self.compute_closeness_approx(n_landmarks=closeness_landmarks)
        else:
            cc = self.compute_closeness()

        node_ids = list(self.G.nodes())
        bc_vals = np.array([bc.get(n, 0.0) for n in node_ids], dtype=np.float32)
        cc_vals = np.array([cc.get(n, 0.0) for n in node_ids], dtype=np.float32)

        # Min-max normalise to [0, 1]
        def _minmax(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

        df = pl.DataFrame(
            {
                "node_id": node_ids,
                "betweenness_centrality": bc_vals.tolist(),
                "closeness_centrality": cc_vals.tolist(),
                "betweenness_norm": _minmax(bc_vals).tolist(),
                "closeness_norm": _minmax(cc_vals).tolist(),
            }
        )

        # Cache to parquet
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(self.cache_path))
        logger.info(
            "Centrality cached → %s  (%d rows)", self.cache_path, len(df)
        )

        self._centrality_df = df
        return df

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_centrality_df(self) -> pl.DataFrame:
        """Return cached DataFrame (call build() first)."""
        if self._centrality_df is None:
            return self.build()
        return self._centrality_df

    def get_betweenness(self, node_id: str) -> float:
        """Normalised betweenness for a single node. Returns 0.0 if not found."""
        df = self.get_centrality_df()
        row = df.filter(pl.col("node_id") == node_id)
        if len(row) == 0:
            return 0.0
        return float(row["betweenness_norm"][0])

    def get_closeness(self, node_id: str) -> float:
        """Normalised closeness for a single node. Returns 0.0 if not found."""
        df = self.get_centrality_df()
        row = df.filter(pl.col("node_id") == node_id)
        if len(row) == 0:
            return 0.0
        return float(row["closeness_norm"][0])

    def top_n_by_betweenness(self, n: int = 20) -> pl.DataFrame:
        """Return top-n nodes ranked by betweenness centrality."""
        return (
            self.get_centrality_df()
            .sort("betweenness_norm", descending=True)
            .head(n)
        )

    def get_neighbors(self, node_id: str, hops: int = 2) -> list[str]:
        """
        Return all nodes within `hops` of node_id.
        Used by graph_diffusion.py for congestion propagation.
        """
        if self.G is None:
            self.load_graph()
        if node_id not in self.G:
            return []
        neighbors: set[str] = set()
        frontier = {node_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                for nb in self.G.neighbors(n):
                    if nb not in neighbors and nb != node_id:
                        next_frontier.add(nb)
            neighbors.update(next_frontier)
            frontier = next_frontier
        return list(neighbors)

    def edge_weight(self, u: str, v: str) -> float:
        """
        Return normalised edge weight between two adjacent nodes.
        Weight = 1 / length_metres (closer = stronger connection).
        Falls back to 1.0 if edge missing.
        """
        if self.G is None:
            self.load_graph()
        if not self.G.has_edge(u, v):
            return 0.0
        length = self.G[u][v].get("length", 100.0)
        # Normalise: assume max meaningful road segment = 2km
        return float(np.clip(1.0 - (length / 2000.0), 0.0, 1.0))


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    gi = GraphIntelligence()
    gi.load_graph()
    df = gi.build(k_betweenness=50, force_recompute=False)

    print("\n── Centrality summary ──────────────────────────────────")
    print(df.describe())
    print("\n── Top-10 structurally critical nodes ─────────────────")
    print(gi.top_n_by_betweenness(10))