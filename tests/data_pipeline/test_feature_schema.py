"""Tests de feature_schema.py — RNF12: versionado del esquema de features."""

import json

import pandas as pd
import pytest

from DataPipeline.feature_schema import FEATURE_COLUMNS, enforce_schema, write_schema_manifest


def test_enforce_schema_reorders_columns():
    expected = FEATURE_COLUMNS["sp500"]
    shuffled = expected[::-1]
    df = pd.DataFrame({col: [1, 2] for col in shuffled})

    result = enforce_schema(df, "sp500")

    assert list(result.columns) == expected


def test_enforce_schema_raises_on_missing_column():
    incomplete = {col: [1] for col in FEATURE_COLUMNS["sp500"][:-1]}  # falta la última
    df = pd.DataFrame(incomplete)

    with pytest.raises(ValueError, match="faltan columnas"):
        enforce_schema(df, "sp500")


def test_write_schema_manifest_creates_expected_json(tmp_path):
    manifest_path = write_schema_manifest("sp500", tmp_path)

    assert manifest_path.exists()
    content = json.loads(manifest_path.read_text())
    assert content["dataset"] == "sp500"
    assert content["columns"] == FEATURE_COLUMNS["sp500"]
    assert "hash" in content and len(content["hash"]) == 12


def test_schema_hash_is_stable_for_same_columns(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    hash_a = json.loads(write_schema_manifest("sp500", dir_a).read_text())["hash"]
    hash_b = json.loads(write_schema_manifest("sp500", dir_b).read_text())["hash"]
    assert hash_a == hash_b
