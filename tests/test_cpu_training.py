from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from src.utils import load_config, validate_research_date_split


def _load_cpu_launcher():
    path = Path("scripts/train_cpu.py")
    spec = importlib.util.spec_from_file_location("train_cpu_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_launcher_freezes_cpu_environment(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "cpu.yaml"
    config_path.write_text(
        "cpu_runtime:\n  torch_threads: 3\n  interop_threads: 1\n  blas_threads: 2\n",
        encoding="utf-8",
    )
    for name in (
        "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    report = _load_cpu_launcher().configure_environment(config_path)

    assert report["torch_threads"] == 3
    assert report["interop_threads"] == 1
    assert report["blas_threads"] == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_cpu_configs_disable_amp_and_use_separate_outputs() -> None:
    daily = load_config("configs/training_cpu.yaml")
    minute = load_config("configs/minute_training_cpu.yaml")
    assert validate_research_date_split(daily)["training"] == "2020-01-01..2023-12-31"
    assert daily["training"]["mixed_precision"] is False
    assert minute["training"]["mixed_precision"] is False
    assert "cpu" in daily["outputs"]["checkpoint"]
    assert "cpu" in minute["outputs"]["checkpoint"]
    assert daily["operators"]["time_series_backend"] == "pandas"
