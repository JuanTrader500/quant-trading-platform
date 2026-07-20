"""Tests de pipeline_manager.py — gate de reentrenamiento y orquestación
de extracción + preparación, sin tocar red ni disco real."""

from datetime import datetime, timedelta

import joblib
import pytest

from DataPipeline.pipeline_manager import PipelineManager


def _write_metadata(path, last_trained: datetime):
    joblib.dump({"last_trained": last_trained.strftime("%Y-%m-%d")}, path)


def test_check_retraining_needed_true_when_no_metadata(tmp_path):
    manager = PipelineManager(model_metadata_path=tmp_path / "missing.pkl")
    assert manager.check_retraining_needed() is True


def test_check_retraining_needed_true_when_model_is_stale(tmp_path):
    metadata_path = tmp_path / "model_metadata.pkl"
    _write_metadata(metadata_path, datetime.now() - timedelta(weeks=2))

    manager = PipelineManager(model_metadata_path=metadata_path)
    assert manager.check_retraining_needed() is True


def test_check_retraining_needed_false_when_model_is_fresh(tmp_path):
    metadata_path = tmp_path / "model_metadata.pkl"
    _write_metadata(metadata_path, datetime.now() - timedelta(days=1))

    manager = PipelineManager(model_metadata_path=metadata_path)
    assert manager.check_retraining_needed() is False


def test_check_retraining_needed_true_on_corrupted_metadata(tmp_path):
    metadata_path = tmp_path / "model_metadata.pkl"
    metadata_path.write_bytes(b"esto no es un pickle valido")

    manager = PipelineManager(model_metadata_path=metadata_path)
    assert manager.check_retraining_needed() is True


def test_execute_skips_pipeline_when_model_is_fresh(tmp_path, monkeypatch):
    metadata_path = tmp_path / "model_metadata.pkl"
    _write_metadata(metadata_path, datetime.now() - timedelta(days=1))
    manager = PipelineManager(model_metadata_path=metadata_path)

    def _should_not_run():
        raise AssertionError("run_full_pipeline no debería ejecutarse con modelo vigente")

    monkeypatch.setattr(manager, "run_full_pipeline", _should_not_run)
    manager.execute()  # no debe lanzar AssertionError


def test_execute_runs_pipeline_when_retraining_needed(tmp_path, monkeypatch):
    manager = PipelineManager(model_metadata_path=tmp_path / "missing.pkl")
    called = {"ran": False}

    def _fake_run():
        called["ran"] = True
        return True

    monkeypatch.setattr(manager, "run_full_pipeline", _fake_run)
    manager.execute()

    assert called["ran"] is True


def test_run_full_pipeline_aborts_when_extraction_fails(monkeypatch):
    manager = PipelineManager()
    monkeypatch.setattr(manager, "_run_extraction", lambda: False)

    def _should_not_run():
        raise AssertionError("la preparación no debe correr si la extracción falló")

    monkeypatch.setattr(manager, "_run_preparation", _should_not_run)

    assert manager.run_full_pipeline() is False


def test_run_full_pipeline_succeeds_when_both_steps_succeed(monkeypatch):
    manager = PipelineManager()
    monkeypatch.setattr(manager, "_run_extraction", lambda: True)
    monkeypatch.setattr(manager, "_run_preparation", lambda: True)

    assert manager.run_full_pipeline() is True
