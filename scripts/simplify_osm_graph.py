"""
scripts/simplify_osm_graph.py
------------------------------
One-time utility: simplifies the cached Bengaluru OSM graph by collapsing
interstitial nodes (pure geometry points with no real intersection) so that
betweenness/closeness centrality runs faster in Layer 6.

Run ONCE after Layer 2's OSM download, before Layer 6's
graph_intelligence.py build(). Overwrites the graphml in place (backs up
the original first).

Usage:
    python -m scripts.simplify_osm_graph
"""

import shutil
from pathlib import Path

import osmnx as ox

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO_ROOT / "data" / "external" / "bengaluru_osm_graph.graphml"
BACKUP_PATH = REPO_ROOT / "data" / "external" / "bengaluru_osm_graph_raw.graphml"


def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Graph not found at {GRAPH_PATH}. Run Layer 2 first.")

    # Back up the original — don't lose the raw download
    if not BACKUP_PATH.exists():
        shutil.copy(GRAPH_PATH, BACKUP_PATH)
        print(f"Backed up raw graph to {BACKUP_PATH}")

    print("Loading graph…")
    G = ox.load_graphml(GRAPH_PATH)
    print(f"Before simplify: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # osmnx graphs downloaded via graph_from_place() are usually already
    # simplified by default — if so, this is a no-op. Safe either way.
    if not G.graph.get("simplified", False):
        G = ox.simplify_graph(G)
        print(f"After simplify_graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    else:
        print("Graph already simplified — skipping simplify_graph().")

    # Optional, more aggressive: merge intersections within `tolerance` metres
    # of each other into a single node. Useful for messy/dense OSM data but
    # can merge genuinely distinct junctions if tolerance is too high.
    G_proj = ox.project_graph(G)
    G_consolidated = ox.consolidate_intersections(G_proj, tolerance=10, rebuild_graph=True)
    G_consolidated = ox.project_graph(G_consolidated, to_latlong=True)
    print(f"After consolidate_intersections: {len(G_consolidated.nodes)} nodes, "
          f"{len(G_consolidated.edges)} edges")

    ox.save_graphml(G_consolidated, GRAPH_PATH)
    print(f"Saved simplified graph → {GRAPH_PATH}")


if __name__ == "__main__":
    main()