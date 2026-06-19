"""
graph_intelligence.py
---------------------
Loads the Bengaluru OSM road network graph and computes structural centrality
metrics for every junction node.

Centrality metrics computed:
  - betweenness_centrality  (k=500 approximation, weight='length')
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
        Exact computation on largest connected component only (fast enough).
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

    # ── Build / cache ──────────────────────────────────────────────────────────

    def build(self, k_betweenness: int = 500, force_recompute: bool = False) -> pl.DataFrame:
        """
        Main entry point. If cache exists and force_recompute=False, loads from
        parquet. Otherwise computes centrality and writes cache.

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
    df = gi.build(k_betweenness=500, force_recompute=False)

    print("\n── Centrality summary ──────────────────────────────────")
    print(df.describe())
    print("\n── Top-10 structurally critical nodes ─────────────────")
    print(gi.top_n_by_betweenness(10))