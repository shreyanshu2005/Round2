"""
Drop this in E:\btip-gridlock2\ and run:
  python diagnose.py
It prints sample keys from both sides so we can see the exact mismatch.
"""
import ast
import networkx as nx
import polars as pl
from pathlib import Path

REPO_ROOT = Path(r"E:\btip-gridlock2")
CENTRALITY_PATH = REPO_ROOT / "data" / "processed" / "junction_centrality.parquet"
GRAPH_PATH      = REPO_ROOT / "data" / "external"  / "bengaluru_osm_graph.graphml"

print("=== CENTRALITY node_id samples ===")
centrality = pl.read_parquet(str(CENTRALITY_PATH))
print(centrality.head(5))
print("node_id dtype:", centrality["node_id"].dtype)
print("Sample node_ids:", centrality["node_id"].head(10).to_list())

print("\n=== GRAPHML node attribute samples ===")
G = nx.read_graphml(str(GRAPH_PATH))
nodes = list(G.nodes(data=True))[:10]
for node_id, attrs in nodes:
    print(f"  node_id={repr(node_id)}  type={type(node_id).__name__}  attrs_keys={list(attrs.keys())}  osmid={repr(attrs.get('osmid', 'MISSING'))}")

print("\n=== Do centrality node_ids appear in graph? ===")
graph_node_ids = set(str(n) for n in G.nodes())
cent_node_ids  = set(str(x) for x in centrality["node_id"].to_list())
overlap = graph_node_ids & cent_node_ids
print(f"  Graph nodes:      {len(graph_node_ids)}")
print(f"  Centrality nodes: {len(cent_node_ids)}")
print(f"  Overlap:          {len(overlap)}")
print(f"  Sample graph IDs: {list(graph_node_ids)[:5]}")
print(f"  Sample cent  IDs: {list(cent_node_ids)[:5]}")