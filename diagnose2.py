"""
Run from E:\btip-gridlock2\:
  python diagnose2.py
"""
import polars as pl
from pathlib import Path

REPO_ROOT = Path(r"E:\btip-gridlock2")
FEATURE_STORE_PATH = REPO_ROOT / "data" / "processed" / "clustered_feature_store.parquet"
CENTRALITY_PATH    = REPO_ROOT / "data" / "processed" / "junction_centrality.parquet"

fs = pl.read_parquet(str(FEATURE_STORE_PATH))
centrality = pl.read_parquet(str(CENTRALITY_PATH))

print("=== FEATURE STORE columns ===")
print(fs.columns)

junction_col = "junction_id_snapped" if "junction_id_snapped" in fs.columns else "junction_name"
print(f"\n=== junction column used: '{junction_col}' ===")
print(f"dtype: {fs[junction_col].dtype}")
print("Sample values:", fs[junction_col].drop_nulls().head(10).to_list())

print("\n=== CENTRALITY node_id samples ===")
print("dtype:", centrality["node_id"].dtype)
print("Sample:", centrality["node_id"].head(10).to_list())

print("\n=== Overlap check ===")
junc_ids  = set(fs[junction_col].drop_nulls().cast(pl.Utf8).to_list())
cent_ids  = set(centrality["node_id"].cast(pl.Utf8).to_list())
overlap   = junc_ids & cent_ids
print(f"  Unique junction IDs:  {len(junc_ids)}")
print(f"  Centrality node IDs:  {len(cent_ids)}")
print(f"  Overlap:              {len(overlap)}")
print(f"  Sample junction IDs:  {list(junc_ids)[:5]}")
print(f"  Sample centrality IDs:{list(cent_ids)[:5]}")