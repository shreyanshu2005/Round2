import polars as pl

meta = pl.read_parquet("data/processed/junction_metadata.parquet")
print(f"Junction count: {meta.height}")
print(meta.head(5))

df = pl.read_parquet("data/processed/violations_clean.parquet")
named = df["junction_name"].is_not_null() & (df["junction_name"] != "No Junction")
print(f"\nNamed rows: {named.sum()} / {df.height}")
print(f"Expected coverage ceiling (named only): {named.sum() / df.height:.1%}")