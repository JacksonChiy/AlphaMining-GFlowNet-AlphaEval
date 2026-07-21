from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_experiment(
    config_path: str | Path,
    root: str | Path = "experiments",
    experiment_id: str | None = None,
) -> Path:
    config_path = Path(config_path)
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")
    output = Path(root) / experiment_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "backtest_report").mkdir()
    config_text = config_path.read_text(encoding="utf-8")
    (output / "config.yaml").write_text(config_text, encoding="utf-8")
    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "created",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output

