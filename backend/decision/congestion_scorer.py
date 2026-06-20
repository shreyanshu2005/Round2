"""
congestion_scorer.py
--------------------
Computes a Congestion Impact Score (0–100) for every junction.

Formula
-------
congestion_score =
    0.35 × betweenness_norm
  + 0.30 × violation_density_norm
  + 0.20 × rolling_7d_norm
  + 0.15 × rush_hour_factor

Each component is independently min-max normalised to [0, 1] before weighting.
The final score is scaled to [0, 100] and stored in the `junction_stats`
PostgreSQL table.

Key outputs
-----------
  data/processed/junction_congestion_scores.parquet
  PostgreSQL table: junction_stats  (upserted)

Usage
-----
  from backend.decision.congestion_scorer import CongestionScorer
  cs = CongestionScorer()
  df = cs.compute()            # returns polars DataFrame
  cs.upsert_to_db(df)          # writes to PostgreSQL
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STORE_PATH = REPO_ROOT / "data" / "processed" / "clustered_feature_store.parquet"
CENTRALITY_PATH = REPO_ROOT / "data" / "processed" / "junction_centrality.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "junction_congestion_scores.parquet"

# ── Weights (must sum to 1.0) ──────────────────────────────────────────────────
WEIGHTS = {
    "betweenness": 0.20,       # reduced: prevents low-violation high-centrality nodes dominating
    "violation_density": 0.45,  # increased: violation count is the primary signal
    "rolling_7d": 0.25,         # increased: recent trend matters more than structure
    "rush_hour": 0.10,          # reduced slightly
}

# ── Rush-hour hours ────────────────────────────────────────────────────────────
RUSH_HOURS = {7, 8, 9, 17, 18, 19, 20}


def _minmax_norm(series: pl.Series) -> pl.Series:
    """Min-max normalise a Polars Series to [0, 1]. Returns zeros if flat."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pl.Series(series.name, [0.0] * len(series))
    return (series - mn) / (mx - mn)


class CongestionScorer:
    """
    Computes per-junction congestion impact scores by combining:
      1. Graph betweenness centrality (structural importance)
      2. Violation density (violations per km² estimated)
      3. Rolling 7-day violation count (recent trend)
      4. Rush-hour incident fraction (time-of-day sensitivity)

    Parameters
    ----------
    feature_store_path : Path to clustered_feature_store.parquet
    centrality_path    : Path to junction_centrality.parquet (from GraphIntelligence)
    output_path        : Where to write junction_congestion_scores.parquet
    """

    def __init__(
        self,
        feature_store_path: Path = FEATURE_STORE_PATH,
        centrality_path: Path = CENTRALITY_PATH,
        output_path: Path = OUTPUT_PATH,
    ):
        self.feature_store_path = feature_store_path
        self.centrality_path = centrality_path
        self.output_path = output_path

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_feature_store(self) -> pl.DataFrame:
        logger.info("Loading feature store from %s …", self.feature_store_path)
        df = pl.read_parquet(str(self.feature_store_path))
        logger.info("Feature store: %d rows, %d cols", df.height, df.width)
        return df

    def _load_centrality(self) -> pl.DataFrame:
        if not self.centrality_path.exists():
            raise FileNotFoundError(
                f"Centrality cache not found at {self.centrality_path}. "
                "Run graph_intelligence.py build() first."
            )
        logger.info("Loading centrality from %s …", self.centrality_path)
        return pl.read_parquet(str(self.centrality_path))

    # ── Per-junction aggregation ───────────────────────────────────────────────

    def _aggregate_junctions(self, fs: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregate feature store to one row per junction.

        Returns columns:
            junction_id, lat, lng, total_violations, violation_density_raw,
            rolling_7d_mean, rush_hour_fraction
        """
        # Determine which column holds junction identity
        junction_col = "junction_id_snapped" if "junction_id_snapped" in fs.columns else "junction_name"
        lat_col = "latitude"
        lng_col = "longitude"

        # Filter out noise (un-snapped rows)
        if "junction_id_snapped" in fs.columns:
            fs = fs.filter(pl.col("junction_id_snapped").is_not_null())

        # Rush-hour flag — recalculate if missing
        if "is_rush_hour" not in fs.columns:
            fs = fs.with_columns(
                pl.col("hour").is_in(list(RUSH_HOURS)).alias("is_rush_hour")
            )

        agg = (
            fs.group_by(junction_col)
            .agg(
                pl.col(lat_col).mean().alias("lat"),
                pl.col(lng_col).mean().alias("lng"),
                pl.len().alias("total_violations"),
                pl.col("rolling_7d_count").mean().alias("rolling_7d_mean"),
                pl.col("is_rush_hour").cast(pl.Float32).mean().alias("rush_hour_fraction"),
            )
            .rename({junction_col: "junction_id"})
        )

        # Violation density proxy: violations per degree² area (approx; no
        # real area data available without PostGIS). We use a 0.01° grid cell
        # area ≈ 1.2 km² around Bengaluru latitude — constant, so ranking is
        # preserved. Pure normalisation signal.
        agg = agg.with_columns(
            (pl.col("total_violations") / 1.2).alias("violation_density_raw")
        )

        logger.info("Aggregated to %d junctions.", agg.height)
        return agg

    # ── Score computation ──────────────────────────────────────────────────────

    def compute(self) -> pl.DataFrame:
        """
        Full computation pipeline.

        Returns
        -------
        pl.DataFrame with one row per junction, columns:
            junction_id, lat, lng, total_violations, rolling_7d_mean,
            rush_hour_fraction, betweenness_norm, closeness_norm,
            violation_density_norm, rolling_7d_norm, rush_hour_norm,
            congestion_score  (0–100)
        """
        fs = self._load_feature_store()
        centrality = self._load_centrality()
        agg = self._aggregate_junctions(fs)

        # ── Join centrality ────────────────────────────────────────────────────
        # junction_id_snapped has an "OSM_" prefix (e.g. "OSM_11896408351").
        # centrality.node_id is the raw OSM integer string (e.g. "11896408351").
        # Strip the prefix so they match. Non-OSM IDs (e.g. "BTP044") won't
        # match and correctly get betweenness = 0 (not in the road graph).
        agg = agg.with_columns(
            pl.col("junction_id")
            .str.strip_prefix("OSM_")
            .alias("junction_osm_id")
        )
        merged = agg.join(
            centrality.select(["node_id", "betweenness_norm", "closeness_norm"]),
            left_on="junction_osm_id",
            right_on="node_id",
            how="left",
        ).with_columns(
            pl.col("betweenness_norm").fill_null(0.0),
            pl.col("closeness_norm").fill_null(0.0),
        ).drop("junction_osm_id")

        # ── Normalise each component independently ─────────────────────────────
        merged = merged.with_columns(
            _minmax_norm(merged["violation_density_raw"]).alias("violation_density_norm"),
            _minmax_norm(merged["rolling_7d_mean"]).alias("rolling_7d_norm"),
            _minmax_norm(merged["rush_hour_fraction"]).alias("rush_hour_norm"),
            # betweenness_norm already in [0,1] from GraphIntelligence
        )

        # ── Weighted combination ───────────────────────────────────────────────
        merged = merged.with_columns(
            (
                WEIGHTS["betweenness"] * pl.col("betweenness_norm")
                + WEIGHTS["violation_density"] * pl.col("violation_density_norm")
                + WEIGHTS["rolling_7d"] * pl.col("rolling_7d_norm")
                + WEIGHTS["rush_hour"] * pl.col("rush_hour_norm")
            )
            .alias("congestion_score_raw")
        )

        # Scale to 0–100 and clip
        merged = merged.with_columns(
            (_minmax_norm(merged["congestion_score_raw"]) * 100.0)
            .round(2)
            .clip(0.0, 100.0)
            .alias("congestion_score")
        )

        # Drop intermediate column
        merged = merged.drop("congestion_score_raw")

        # ── Cache ──────────────────────────────────────────────────────────────
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(str(self.output_path))
        logger.info(
            "Congestion scores written → %s  (%d junctions)", self.output_path, len(merged)
        )

        return merged

    # ── DB upsert ─────────────────────────────────────────────────────────────

    def upsert_to_db(self, df: pl.DataFrame) -> None:
        """
        Upsert congestion scores into PostgreSQL junction_stats table.
        Creates the table if it does not exist.

        Requires DATABASE_URL in environment (set via .env / docker-compose).
        """
        import psycopg2
        from psycopg2.extras import execute_values

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not set — skipping DB upsert.")
            return

        create_sql = """
        CREATE TABLE IF NOT EXISTS junction_stats (
            junction_id             TEXT PRIMARY KEY,
            lat                     DOUBLE PRECISION,
            lng                     DOUBLE PRECISION,
            total_violations        INTEGER,
            rolling_7d_mean         DOUBLE PRECISION,
            rush_hour_fraction      DOUBLE PRECISION,
            betweenness_norm        DOUBLE PRECISION,
            closeness_norm          DOUBLE PRECISION,
            violation_density_norm  DOUBLE PRECISION,
            rolling_7d_norm         DOUBLE PRECISION,
            rush_hour_norm          DOUBLE PRECISION,
            congestion_score        DOUBLE PRECISION,
            updated_at              TIMESTAMPTZ DEFAULT NOW()
        );
        """

        upsert_sql = """
        INSERT INTO junction_stats (
            junction_id, lat, lng, total_violations, rolling_7d_mean,
            rush_hour_fraction, betweenness_norm, closeness_norm,
            violation_density_norm, rolling_7d_norm, rush_hour_norm,
            congestion_score
        ) VALUES %s
        ON CONFLICT (junction_id) DO UPDATE SET
            lat                    = EXCLUDED.lat,
            lng                    = EXCLUDED.lng,
            total_violations       = EXCLUDED.total_violations,
            rolling_7d_mean        = EXCLUDED.rolling_7d_mean,
            rush_hour_fraction     = EXCLUDED.rush_hour_fraction,
            betweenness_norm       = EXCLUDED.betweenness_norm,
            closeness_norm         = EXCLUDED.closeness_norm,
            violation_density_norm = EXCLUDED.violation_density_norm,
            rolling_7d_norm        = EXCLUDED.rolling_7d_norm,
            rush_hour_norm         = EXCLUDED.rush_hour_norm,
            congestion_score       = EXCLUDED.congestion_score,
            updated_at             = NOW();
        """

        # Build rows — handle possibly missing columns gracefully
        def _get(row: dict, col: str, default=0.0):
            return row.get(col, default)

        records = []
        rows = df.to_dicts()
        for row in rows:
            records.append((
                str(row["junction_id"]),
                _get(row, "lat"),
                _get(row, "lng"),
                int(_get(row, "total_violations", 0)),
                _get(row, "rolling_7d_mean"),
                _get(row, "rush_hour_fraction"),
                _get(row, "betweenness_norm"),
                _get(row, "closeness_norm"),
                _get(row, "violation_density_norm"),
                _get(row, "rolling_7d_norm"),
                _get(row, "rush_hour_norm"),
                _get(row, "congestion_score"),
            ))

        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                execute_values(cur, upsert_sql, records, page_size=1000)
            conn.commit()

        logger.info("Upserted %d rows into junction_stats.", len(records))

    # ── Convenience accessors ──────────────────────────────────────────────────

    def get_top_n(self, n: int = 20, loaded_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """Return top-n junctions by congestion_score."""
        df = loaded_df if loaded_df is not None else pl.read_parquet(str(self.output_path))
        return df.sort("congestion_score", descending=True).head(n)

    def get_score(
        self, junction_id: str, loaded_df: Optional[pl.DataFrame] = None
    ) -> float:
        """Return congestion score for a single junction_id."""
        df = loaded_df if loaded_df is not None else pl.read_parquet(str(self.output_path))
        row = df.filter(pl.col("junction_id") == junction_id)
        return float(row["congestion_score"][0]) if len(row) > 0 else 0.0


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    scorer = CongestionScorer()
    result = scorer.compute()

    print("\n── Congestion score summary ────────────────────────────")
    print(result.select(["junction_id", "congestion_score", "betweenness_norm",
                          "violation_density_norm"]).describe())

    print("\n── Top-10 by congestion score ──────────────────────────")
    top10 = scorer.get_top_n(10, loaded_df=result)
    print(top10.select(["junction_id", "congestion_score", "total_violations",
                         "betweenness_norm"]))

    print("\n── Top-10 by raw violation count ───────────────────────")
    top10_raw = result.sort("total_violations", descending=True).head(10)
    print(top10_raw.select(["junction_id", "total_violations", "congestion_score"]))

    # Attempt DB upsert (silent if no DATABASE_URL)
    scorer.upsert_to_db(result)