"""
Layer 2 — Deduplication

Flags likely-duplicate violations: same vehicle_number + junction_id_snapped
within a 5-minute window. Sorted by (vehicle, junction, time), then flags
a row as a duplicate if the previous row in the same vehicle+junction
group is within DEDUP_WINDOW_SECONDS.

Deliberately FLAGS rather than DROPS rows (is_duplicate: bool). The
original build guide says "remove duplicates", but also lists
is_duplicate as a final feature_store column — which only makes sense if
rows are kept and flagged. Flagging is also safer for a hackathon
timeline: if the 5-minute window logic needs tuning later, dropped rows
can't be recovered without re-running the whole pipeline from clean
parquet, but a flag can just be ignored by anything that doesn't want it.

Rows with a null junction_id_snapped are never flagged as duplicates —
without a reliable location, "same junction" can't be established.

Run standalone:
    python backend/preprocessing/deduplication.py
Or call flag_duplicates(df) directly from build_feature_store.py.
"""

import polars as pl

DEDUP_WINDOW_SECONDS = 5 * 60


def flag_duplicates(df: pl.DataFrame, ts_col: str = "created_datetime") -> pl.DataFrame:
    df = df.sort(["vehicle_number", "junction_id_snapped", ts_col])

    df = df.with_columns(
        pl.col(ts_col)
        .diff()
        .over(["vehicle_number", "junction_id_snapped"])
        .alias("_time_diff")
    )

    df = df.with_columns(
        (
            pl.col("_time_diff").is_not_null()
            & (pl.col("_time_diff").dt.total_seconds() <= DEDUP_WINDOW_SECONDS)
            & pl.col("junction_id_snapped").is_not_null()
        ).alias("is_duplicate")
    ).drop("_time_diff")

    n_dupes = df["is_duplicate"].sum()
    print(
        f"Flagged {n_dupes} / {df.height} rows as duplicates "
        f"(same vehicle+junction within {DEDUP_WINDOW_SECONDS}s)"
    )

    return df


if __name__ == "__main__":
    IN_PATH = "data/processed/violations_temporal.parquet"
    OUT_PATH = "data/processed/violations_deduped.parquet"

    print(f"Loading {IN_PATH} ...")
    df = pl.read_parquet(IN_PATH)
    df = flag_duplicates(df)
    df.write_parquet(OUT_PATH)
    print(f"✅ Wrote {OUT_PATH} ({df.height} rows)")