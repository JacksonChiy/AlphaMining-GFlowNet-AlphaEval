from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.pipeline.daily_local import (
    DailyLocalPaths,
    build_stage_sequence,
    run_daily_local,
)
from src.utils import load_config, validate_research_date_split


def _write_local_config(tmp_path: Path) -> Path:
    result_dir = tmp_path / "daily_local"
    config = {
        "dataset": {
            "file": str(tmp_path / "price.csv"),
            "output": str(tmp_path / "daily.pkl"),
            "horizon": 5,
            "start_date": "2020-01-01",
            "end_date": "2026-12-31",
            "mining_start_date": "2020-01-01",
            "mining_end_date": "2023-12-31",
            "out_of_sample_start_date": "2024-01-01",
            "out_of_sample_end_date": "2026-12-31",
        },
        "lightgbm": {
            "prediction_start_date": "2024-01-01",
            "prediction_end_date": "2026-12-31",
        },
        "outputs": {
            "data_quality_report": str(result_dir / "quality.json"),
            "checkpoint": str(result_dir / "model.pt"),
            "metrics": str(result_dir / "metrics.csv"),
            "trajectory_metrics": str(result_dir / "trajectories.csv"),
            "alpha_pool": str(result_dir / "pool.csv"),
            "factor_matrix": str(result_dir / "factors.pkl"),
            "oos_factor_matrix": str(result_dir / "factors_oos.pkl"),
            "alpha_eval": str(result_dir / "alpha_eval.csv"),
            "lightgbm_dir": str(result_dir / "lightgbm"),
            "backtest_dir": str(result_dir / "backtest"),
            "pipeline_manifest": str(result_dir / "manifest.json"),
        },
    }
    path = tmp_path / "local.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_daily_local_config_is_cpu_only_and_has_isolated_outputs() -> None:
    config = load_config("configs/daily/local.yaml")

    assert validate_research_date_split(config) == {
        "training": "2020-01-01..2023-12-31",
        "out_of_sample": "2024-01-01..2026-12-31",
    }
    assert config["training"]["mixed_precision"] is False
    assert config["operators"]["time_series_backend"] == "pandas"
    assert config["outputs"]["factor_matrix"].startswith("results/daily_local/")
    assert config["outputs"]["alpha_eval"].startswith("results/daily_local/")
    assert config["outputs"]["lightgbm_dir"] == "results/daily_local/lightgbm"


def test_daily_local_stage_sequence_supports_resume() -> None:
    assert build_stage_sequence("prepare", "lightgbm") == (
        "prepare", "gflownet", "alpha_eval", "lightgbm"
    )
    assert build_stage_sequence("alpha_eval", "backtest") == (
        "alpha_eval", "lightgbm", "backtest"
    )
    with pytest.raises(ValueError, match="must not be later"):
        build_stage_sequence("lightgbm", "prepare")


def test_daily_local_paths_are_entirely_config_driven(tmp_path) -> None:
    config_path = _write_local_config(tmp_path)
    paths = DailyLocalPaths.from_config(load_config(config_path))

    assert paths.prepared_price == tmp_path / "daily.pkl"
    assert paths.alpha_pool == tmp_path / "daily_local" / "pool.csv"
    assert paths.lightgbm_dir == tmp_path / "daily_local" / "lightgbm"
    assert paths.manifest == tmp_path / "daily_local" / "manifest.json"


def test_prepare_only_run_records_completed_manifest(tmp_path, monkeypatch) -> None:
    config_path = _write_local_config(tmp_path)
    prepared = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
        "code": ["000001.SZ", "000001.SZ"],
    })

    def fake_prepare(*args, **kwargs):
        return prepared

    monkeypatch.setattr("src.pipeline.daily_local.prepare_price_csv", fake_prepare)
    manifest = run_daily_local(
        config_path, from_stage="prepare", to_stage="prepare"
    )

    manifest_path = tmp_path / "daily_local" / "manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert persisted["stages"]["prepare"]["status"] == "completed"
    assert persisted["stages"]["prepare"]["details"]["rows"] == 2


def test_resume_fails_early_when_required_artifact_is_missing(tmp_path) -> None:
    config_path = _write_local_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="requires missing artifacts"):
        run_daily_local(
            config_path,
            from_stage="alpha_eval",
            to_stage="alpha_eval",
        )

    manifest_path = tmp_path / "daily_local" / "manifest.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert "FileNotFoundError" in persisted["error"]


def test_backtest_stage_requires_explicit_local_bundle(tmp_path) -> None:
    config_path = _write_local_config(tmp_path)
    with pytest.raises(ValueError, match="requires --rqalpha-bundle"):
        run_daily_local(
            config_path, from_stage="backtest", to_stage="backtest"
        )
