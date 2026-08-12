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
from src.data_loader.minute_quality_audit import (
    DolphinDBMinuteQualityAuditor,
    MinuteQualityAuditConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit DolphinDB minute grids, duplicates and invalid market values"
    )
    parser.add_argument("--config", default="configs/minute/cpu_ddb_memmap.yaml")
    parser.add_argument("--scope", choices=("grid", "full"), default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    dataset = root.get("dataset", {})
    ddb_values = dataset.get("dolphindb", {})
    memmap_values = dataset.get("memmap", {})
    ddb_config = MinuteDolphinDBConfig.from_mapping(dataset, ddb_values)
    audit_config = MinuteQualityAuditConfig.from_mapping(
        memmap_values,
        output_dir=args.output_dir,
        scope=args.scope,
    )
    session = create_dolphindb_session(ddb_values)
    try:
        loader = DolphinDBMinuteLoader(ddb_config, session)
        summary = DolphinDBMinuteQualityAuditor(loader, audit_config).run()
        print(f"[DDBQuality] summary={summary}", flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    main()
