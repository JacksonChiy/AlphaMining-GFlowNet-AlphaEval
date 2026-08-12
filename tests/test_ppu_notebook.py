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
