"""
Layer 2 — Feature Engineering

Adds, on top of geo-snapped + temporal-featured + dedup-flagged data:
  - rolling_7d_count / rolling_30d_count: per-junction trailing violation
    counts, computed on a gap-filled daily calendar per junction (so a
    junction with zero violations on some days doesn't silently shrink
    the window — confirmed necessary since violation density varies a
    lot by junction).
  - primary_violation_type / n_offences / primary_offence_code: parsed
    from the JSON-array-as-string violation_type / offence_code columns.
    Multi-offence rows are common (confirmed on the sample — e.g.
    ["WRONG PARKING","PARKING NEAR ..."] / [112,107]) so this takes the
    first listed offence as "primary" and keeps a count of how many
    offences were logged on the same stop, rather than silently
    collapsing multi-offence rows into a single category.
  - offence_time_cross: primary_violation_type + shift.
  - road_type / lanes: nearest-OSM-node lookup against the cached
    bengaluru_osm_graph.graphml. If that file doesn't exist yet (download
    can take 10-20 min per the build guide's pitfalls section), this
    fills both columns with null rather than blocking the pipeline —
    re-run this step alone once the graph is cached to backfill them.

Run standalone:
    python backend/preprocessing/feature_engineering.py
Or call build_features(df, meta) directly from build_feature_store.py.
"""

import os

import polars as pl

GRAPHML_PATH = "data/external/bengaluru_osm_graph.graphml"


def explode_offences(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("violation_type").str.json_decode(pl.List(pl.Utf8)).alias("_violation_list"),
        pl.col("offence_code").str.json_decode(pl.List(pl.Int64)).alias("_offence_list"),
    )
    df = df.with_columns(
        pl.col("_violation_list").list.first().alias("primary_violation_type"),
        pl.col("_violation_list").list.len().alias("n_offences"),
        pl.col("_offence_list").list.first().alias("primary_offence_code"),
    ).drop(["_violation_list", "_offence_list"])
    return df


def add_rolling_counts(df: pl.DataFrame) -> pl.DataFrame:
    scoped = df.filter(pl.col("junction_id_snapped").is_not_null())

    daily = (
        scoped.with_columns(pl.col("created_datetime_ist").dt.date().alias("date"))
        .group_by(["junction_id_snapped", "date"])
        .agg(pl.len().alias("daily_count"))
    )

    min_date, max_date = daily["date"].min(), daily["date"].max()
    full_dates = pl.date_range(min_date, max_date, interval="1d", eager=True)
    junction_ids = daily["junction_id_snapped"].unique()
    calendar = junction_ids.to_frame().join(pl.DataFrame({"date": full_dates}), how="cross")

    daily_full = (
        calendar.join(daily, on=["junction_id_snapped", "date"], how="left")
        .with_columns(pl.col("daily_count").fill_null(0))
        .sort(["junction_id_snapped", "date"])
        .with_columns(
            pl.col("daily_count")
            .rolling_sum(window_size=7, min_periods=1)
            .over("junction_id_snapped")
            .alias("rolling_7d_count"),
            pl.col("daily_count")
            .rolling_sum(window_size=30, min_periods=1)
            .over("junction_id_snapped")
            .alias("rolling_30d_count"),
        )
    )

    df = df.with_columns(pl.col("created_datetime_ist").dt.date().alias("date"))
    df = df.join(
        daily_full.select("junction_id_snapped", "date", "rolling_7d_count", "rolling_30d_count"),
        on=["junction_id_snapped", "date"],
        how="left",
    ).drop("date")
    return df


def add_offence_time_cross(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("primary_violation_type").fill_null("UNKNOWN") + "_" + pl.col("shift")).alias(
            "offence_time_cross"
        )
    )


def enrich_with_osm(df: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    if not os.path.exists(GRAPHML_PATH):
        print(
            f"⚠️  {GRAPHML_PATH} not found — road_type/lanes will be null "
            f"for this run. Download the OSM graph separately, then re-run "
            f"this step (or the full orchestrator) to backfill."
        )
        junction_osm = meta.select("junction_id").with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("road_type"),
            pl.lit(None, dtype=pl.Int64).alias("lanes"),
        )
    else:
        import networkx as nx
        from pyproj import Transformer
        from scipy.spatial import KDTree

        G = nx.read_graphml(GRAPHML_PATH)
        node_ids, node_x, node_y = [], [], []
        for n, d in G.nodes(data=True):
            if "x" in d and "y" in d:
                node_ids.append(n)
                node_x.append(float(d["x"]))
                node_y.append(float(d["y"]))

        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        nx_utm, ny_utm = to_utm.transform(node_x, node_y)
        tree = KDTree(list(zip(nx_utm, ny_utm)))

        _, idx = tree.query(
            list(zip(meta["centroid_utm_x"].to_list(), meta["centroid_utm_y"].to_list()))
        )

        road_types, lanes_list = [], []
        for i in idx:
            edges = list(G.edges(node_ids[i], data=True))
            highway = edges[0][2].get("highway") if edges else None
            lanes = edges[0][2].get("lanes") if edges else None
            road_types.append(str(highway) if highway else None)
            try:
                lanes_list.append(int(lanes) if lanes else None)
            except (ValueError, TypeError):
                lanes_list.append(None)

        junction_osm = meta.select("junction_id").with_columns(
            pl.Series("road_type", road_types),
            pl.Series("lanes", lanes_list),
        )

    return df.join(junction_osm, left_on="junction_id_snapped", right_on="junction_id", how="left")


def build_features(df: pl.DataFrame, meta: pl.DataFrame) -> pl.DataFrame:
    df = explode_offences(df)
    df = add_rolling_counts(df)
    df = add_offence_time_cross(df)
    df = enrich_with_osm(df, meta)
    return df


if __name__ == "__main__":
    IN_PATH = "data/processed/violations_deduped.parquet"
    META_PATH = "data/processed/junction_metadata.parquet"
    OUT_PATH = "data/processed/feature_store.parquet"

    print(f"Loading {IN_PATH} and {META_PATH} ...")
    df = pl.read_parquet(IN_PATH)
    meta = pl.read_parquet(META_PATH)
    df = build_features(df, meta)
    df.write_parquet(OUT_PATH)
    print(f"✅ Wrote {OUT_PATH} ({df.height} rows, {len(df.columns)} cols)")