from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and cache DolphinDB minute bars")
    parser.add_argument("--config", default="configs/minute/cpu_ddb_memmap.yaml")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument(
        "--audit-output", default="results/minute_cpu_ddb/field_audit.json"
    )
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    dataset = root.get("dataset", {})
    values = dataset.get("dolphindb", {})
    config = MinuteDolphinDBConfig.from_mapping(dataset, values)
    if args.force_refresh:
        config = replace(config, force_refresh=True)
    session = create_dolphindb_session(values)
    try:
        loader = DolphinDBMinuteLoader(config, session)
        if args.audit_only or config.load_mode == "stream":
            audit = loader.audit()
            output = Path(args.audit_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(audit.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[DDB] audit_complete passed={audit.passed} "
                f"rows={audit.source_rows:,} "
                f"source_range={audit.source_min_date}..{audit.source_max_date} "
                f"output={output}",
                flush=True,
            )
            if config.load_mode == "stream" and not args.audit_only:
                print(
                    "[DDB] stream_ready raw_minute_files=false; "
                    "training will query DolphinDB directly and cache only daily blocks in memory",
                    flush=True,
                )
        else:
            cache_dir, daily_file = loader.extract()
            print(
                f"[DDB] prepare_complete cache_dir={cache_dir} daily_file={daily_file}",
                flush=True,
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
