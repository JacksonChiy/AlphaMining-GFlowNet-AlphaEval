from __future__ import annotations

import ast
import json
from pathlib import Path


def test_ppu_training_notebook_is_valid_and_self_contained() -> None:
    path = Path("notebooks/deployment/minute_ppu_memmap.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"

    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")

    assert "ALPHAMINING_MEMMAP_DIR" in code
    assert "ALPHAMINING_BLOCK_CACHE_DIR" in code
    assert "scripts/train_cpu.py" in code
    assert "start_new_session=True" in code
    assert "training_state.json" in code
    assert "manifest.get('complete') is not True" in code
    assert "PRESERVE_PLATFORM_TORCH = True" in code


def test_ppu_ddb_ram_notebook_configures_eager_disk_cache() -> None:
    path = Path("notebooks/deployment/minute_ppu_ram.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if source.lstrip().startswith("%"):
                continue
            ast.parse(source, filename=f"ddb-ram-notebook-cell-{index}")
    assert "ALPHAMINING_RAM_CACHE_DIR" in code
    assert "ram_cache_manifest.json" in code
    assert "cache_ready" in code
    assert "RUN_GFLOWNET_TRAINING" in code
    assert "src.alpha_eval.run_evaluation" in code
    assert "src.model.run_lightgbm" in code
    assert "REBUILD_DAILY_PRICE_IF_MISSING" in code
    assert "scripts/export_ddb_daily.py" in code
    assert "outputs['daily_price']" in code
    assert "alpha_factor_matrix.csv.gz" not in code  # Paths must come from YAML.
    assert "postprocess_manifest" in code
    assert "zipfile.ZipFile" in code


def test_ppu_ddb_ram_config_contains_postprocessing_pipeline() -> None:
    import yaml

    config = yaml.safe_load(
        Path("configs/minute/ppu_ddb_ram.yaml").read_text(encoding="utf-8")
    )
    assert config["dataset"]["horizon"] == 5
    assert config["dataset"]["mining_end_date"] < config["dataset"]["out_of_sample_start_date"]
    assert config["alpha_eval"]["dpp_k"] > 0
    assert config["lightgbm"]["prediction_start_date"] == "2024-01-01"
    assert config["backtest"]["stock_commission_multiplier"] == 1.0
    assert config["outputs"]["factor_matrix"].endswith(".csv.gz")
    assert config["outputs"]["factor_matrix_pickle"].endswith(".pkl")
    assert config["outputs"]["daily_price"].endswith("daily_price.pkl")
    assert config["outputs"]["artifact_package"].endswith(".zip")
