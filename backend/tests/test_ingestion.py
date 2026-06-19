"""
backend/tests/test_ingestion.py

Tests for batch_loader and validator using the project's real-schema sample
data (violations_sample.csv, 500 rows) rather than synthetic fixtures, so the
tests catch real CSV quirks (UTC-offset timestamps, JSON-array-as-string
columns, mostly-null description/closed_datetime fields, etc).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from backend.ingestion import batch_loader, validator

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "violations_sample.csv"


@pytest.fixture(scope="module")
def clean_df():
    df = batch_loader.load_raw(str(FIXTURE_CSV))
    clean, stats = batch_loader.clean(df)
    return clean, stats


def test_batch_loader_drops_invalid_rows(clean_df):
    clean, stats = clean_df
    assert stats["rows_in"] > 0
    assert stats["rows_out"] <= stats["rows_in"]
    assert stats["rows_out"] + stats["rows_dropped"] == stats["rows_in"]


def test_batch_loader_output_in_bengaluru_bounds(clean_df):
    clean, _ = clean_df
    assert clean.filter(
        (pl.col("latitude") < 12.8) | (pl.col("latitude") > 13.2)
    ).height == 0
    assert clean.filter(
        (pl.col("longitude") < 77.4) | (pl.col("longitude") > 77.8)
    ).height == 0


def test_batch_loader_no_null_ids_or_timestamps(clean_df):
    clean, _ = clean_df
    assert clean["id"].null_count() == 0
    assert clean["created_datetime"].null_count() == 0


def test_validator_passes_on_clean_data(clean_df):
    clean, _ = clean_df
    validated = validator.validate(clean)
    assert validated.height == clean.height


def test_validator_fails_on_bad_latitude(clean_df):
    clean, _ = clean_df
    bad = clean.with_columns(pl.lit(99.0).alias("latitude"))
    with pytest.raises(Exception):
        validator.validate(bad)


def test_validator_fails_on_null_id(clean_df):
    clean, _ = clean_df
    bad = clean.with_columns(pl.lit(None).cast(pl.Utf8).alias("id"))
    with pytest.raises(Exception):
        validator.validate(bad)
