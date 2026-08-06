from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dolphindb_minute import (
    DolphinDBMinuteLoader,
    MinuteDolphinDBConfig,
    create_dolphindb_session,
)
from src.data_loader.minute_memmap import (
    DolphinDBMinuteMemMapBuilder,
    MinuteMemMapConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build report-style local minute MemMaps from a remote DolphinDB source"
    )
    parser.add_argument("--config", default="configs/minute_training_cpu_ddb.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    dataset = root.get("dataset", {})
    ddb_values = dataset.get("dolphindb", {})
    memmap_values = dataset.get("memmap", {})
    ddb_config = MinuteDolphinDBConfig.from_mapping(dataset, ddb_values)
    memmap_config = MinuteMemMapConfig.from_mapping(memmap_values)
    session = create_dolphindb_session(ddb_values)
    try:
        loader = DolphinDBMinuteLoader(ddb_config, session)
        manifest = DolphinDBMinuteMemMapBuilder(loader, memmap_config).build()
        print(f"[MemMapBuild] manifest={manifest}", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()

