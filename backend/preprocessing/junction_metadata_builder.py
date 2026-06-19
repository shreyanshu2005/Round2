"""
Layer 2 — Junction Metadata Builder (Option C: OSM-enriched)

Two-source junction master list:
  SOURCE 1 — BTP-coded violations (~150K rows, ~150 unique junctions)
    Violations whose junction_name matches "BTP### - Name" get their
    centroid computed by averaging their UTM-projected GPS points.
    These are authoritative ground-truth junctions with real IDs.

  SOURCE 2 — OSM road-network intersections (Option C fix)
    osmnx pulls all driveable road intersections in Bengaluru.
    We keep only intersection nodes (degree >= 3 in the undirected graph)
    to exclude mid-block nodes that aren't real junctions.
    Each gets junction_id = "OSM_<node_id>", n_source_rows = 0.

Deduplication: any OSM node within 50 m of an existing BTP centroid
is dropped — the BTP entry wins (it has a real ID and source rows).

Output schema (unchanged — geo_snap.py needs zero modifications):
  junction_id, junction_name, centroid_lat, centroid_lng,
  centroid_utm_x, centroid_utm_y, n_source_rows

Run standalone:
    python backend/preprocessing/junction_metadata_builder.py

Or call build_junction_metadata(df) from build_feature_store.py.
"""

import os
import polars as pl
import numpy as np
from pyproj import Transformer
from scipy.spatial import KDTree

CLEAN_PATH = "data/processed/violations_clean.parquet"
OUT_PATH   = "data/processed/junction_metadata.parquet"
OSM_CACHE  = "data/external/bengaluru_osm_graph.graphml"

JUNCTION_RE    = r"^(BTP\d{3}) - (.+)$"
BTP_DEDUP_M    = 50.0   # OSM node within 50 m of a BTP centroid → drop
OSM_MIN_DEGREE = 3      # only keep real intersections (not mid-block nodes)

_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
_to_wgs = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# SOURCE 1: BTP-coded junctions (unchanged logic from original builder)
# ---------------------------------------------------------------------------

def _build_btp_junctions(df: pl.DataFrame) -> pl.DataFrame:
    named   = df.filter(
        pl.col("junction_name").is_not_null()
        & (pl.col("junction_name") != "No Junction")
    )
    matched = named.filter(pl.col("junction_name").str.contains(JUNCTION_RE))

    unmatched_n = named.height - matched.height
    if unmatched_n:
        print(
            f"⚠️  {unmatched_n} named rows didn't match 'BTP### - Name' "
            f"and were excluded from BTP junction list."
        )

    extracted = matched.with_columns(
        pl.col("junction_name").str.extract(JUNCTION_RE, 1).alias("junction_id"),
        pl.col("junction_name").str.extract(JUNCTION_RE, 2).alias("junction_label"),
    )

    lats  = extracted["latitude"].to_list()
    lngs  = extracted["longitude"].to_list()
    utm_x, utm_y = _to_utm.transform(lngs, lats)
    extracted = extracted.with_columns(
        pl.Series("utm_x", utm_x),
        pl.Series("utm_y", utm_y),
    )

    centroids = (
        extracted.group_by("junction_id")
        .agg(
            pl.col("junction_label").first().alias("junction_name"),
            pl.col("utm_x").mean().alias("centroid_utm_x"),
            pl.col("utm_y").mean().alias("centroid_utm_y"),
            pl.len().alias("n_source_rows"),
        )
        .sort("junction_id")
    )

    cx = centroids["centroid_utm_x"].to_list()
    cy = centroids["centroid_utm_y"].to_list()
    lng_back, lat_back = _to_wgs.transform(cx, cy)

    return centroids.with_columns(
        pl.Series("centroid_lng", lng_back),
        pl.Series("centroid_lat", lat_back),
    ).select(
        "junction_id", "junction_name",
        "centroid_lat", "centroid_lng",
        "centroid_utm_x", "centroid_utm_y",
        "n_source_rows",
    )


# ---------------------------------------------------------------------------
# SOURCE 2: OSM intersections
# ---------------------------------------------------------------------------

def _load_or_download_osm_graph():
    """Load cached graphml or download from OSM. Returns networkx graph."""
    import osmnx as ox

    os.makedirs(os.path.dirname(OSM_CACHE), exist_ok=True)

    if os.path.exists(OSM_CACHE):
        print(f"  Loading OSM graph from cache: {OSM_CACHE}")
        import networkx as nx
        G = nx.read_graphml(OSM_CACHE)
        # nx.read_graphml loses the CRS attribute — re-project aware attrs not
        # needed here since we only need node lat/lng which are stored as attrs.
        return G
    else:
        print("  Downloading Bengaluru OSM road network (drive) — this takes 10-20 min …")
        G = ox.graph_from_place(
            "Bengaluru, Karnataka, India",
            network_type="drive",
            simplify=True,
        )
        ox.save_graphml(G, OSM_CACHE)
        print(f"  ✅ Saved OSM graph to {OSM_CACHE}")
        return G


def _build_osm_junctions(btp_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract real intersection nodes from OSM graph, project to UTM,
    deduplicate against BTP centroids, return same schema as BTP table.
    """
    import networkx as nx

    G = _load_or_download_osm_graph()

    # osmnx stores node attributes as strings after graphml round-trip;
    # handle both the live graph (numeric) and loaded graphml (string) cases.
    def _safe_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Build undirected view to compute degree correctly
    G_und = G.to_undirected() if hasattr(G, "to_undirected") else G

    rows = []
    for node_id, data in G_und.nodes(data=True):
        if G_und.degree(node_id) < OSM_MIN_DEGREE:
            continue
        lat = _safe_float(data.get("y") or data.get("lat"))
        lng = _safe_float(data.get("x") or data.get("lon"))
        if lat is None or lng is None:
            continue
        # Bengaluru bounding box sanity check (same as batch_loader)
        if not (12.8 <= lat <= 13.2 and 77.4 <= lng <= 77.8):
            continue
        name = data.get("name") or data.get("street_count") or ""
        rows.append((str(node_id), str(name), lat, lng))

    if not rows:
        print("  ⚠️  No OSM intersection nodes found — check graph download.")
        return pl.DataFrame(schema={
            "junction_id": pl.Utf8, "junction_name": pl.Utf8,
            "centroid_lat": pl.Float64, "centroid_lng": pl.Float64,
            "centroid_utm_x": pl.Float64, "centroid_utm_y": pl.Float64,
            "n_source_rows": pl.Int64,
        })

    node_ids, names, lats, lngs = zip(*rows)
    utm_x_arr, utm_y_arr = _to_utm.transform(list(lngs), list(lats))

    osm_df = pl.DataFrame({
        "node_id":       list(node_ids),
        "osm_name":      list(names),
        "centroid_lat":  list(lats),
        "centroid_lng":  list(lngs),
        "centroid_utm_x": list(utm_x_arr),
        "centroid_utm_y": list(utm_y_arr),
    })

    print(f"  OSM intersection nodes (degree ≥ {OSM_MIN_DEGREE}, in bbox): {osm_df.height:,}")

    # --- Deduplicate against BTP centroids ---
    btp_utm = np.column_stack([
        btp_df["centroid_utm_x"].to_list(),
        btp_df["centroid_utm_y"].to_list(),
    ])
    btp_tree = KDTree(btp_utm)

    osm_utm = np.column_stack([
        osm_df["centroid_utm_x"].to_list(),
        osm_df["centroid_utm_y"].to_list(),
    ])
    dist, _ = btp_tree.query(osm_utm)

    keep_mask = dist > BTP_DEDUP_M
    dropped   = (~keep_mask).sum()
    print(f"  Dropping {dropped:,} OSM nodes within {BTP_DEDUP_M} m of a BTP centroid")

    osm_filtered = osm_df.filter(pl.Series("keep", keep_mask))

    return osm_filtered.with_columns(
        ("OSM_" + pl.col("node_id")).alias("junction_id"),
        pl.col("osm_name").alias("junction_name"),
        pl.lit(0).cast(pl.Int64).alias("n_source_rows"),
    ).select(
        "junction_id", "junction_name",
        "centroid_lat", "centroid_lng",
        "centroid_utm_x", "centroid_utm_y",
        "n_source_rows",
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def build_junction_metadata(df: pl.DataFrame) -> pl.DataFrame:
    print("Building BTP junctions from named violation rows …")
    btp = _build_btp_junctions(df)
    print(f"  BTP junctions: {btp.height}")

    print("Building OSM intersection junctions …")
    osm = _build_osm_junctions(btp)
    print(f"  OSM junctions after dedup: {osm.height:,}")

    combined = pl.concat([btp, osm], how="vertical_relaxed").sort("junction_id")
    print(f"  Total junctions (BTP + OSM): {combined.height:,}")

    return combined


if __name__ == "__main__":
    print(f"Loading {CLEAN_PATH} …")
    df  = pl.read_parquet(CLEAN_PATH)
    meta = build_junction_metadata(df)
    meta.write_parquet(OUT_PATH)
    print(f"\n✅ Wrote {OUT_PATH} ({meta.height:,} junctions)")

    btp_count = meta.filter(pl.col("junction_id").str.starts_with("BTP")).height
    osm_count = meta.filter(pl.col("junction_id").str.starts_with("OSM_")).height
    print(f"   BTP: {btp_count} | OSM: {osm_count:,}")
    print(meta.sort("n_source_rows", descending=True).head(10))