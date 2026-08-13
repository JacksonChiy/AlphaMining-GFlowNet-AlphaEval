from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.daily_artifact import save_daily_price_artifact
from src.data_loader.dolphindb_minute import (
    DolphinDBMinuteLoader,
    MinuteDolphinDBConfig,
    create_dolphindb_session,
)
from src.data_loader.minute_memmap import (
    MinuteMemMapConfig,
    build_ddb_minute_time_filter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the daily OHLCV panel required after PPU minute training"
    )
    parser.add_argument("--config", default="configs/minute/ppu_ddb_ram.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=None,
        help="Override DDB daily aggregation chunk size when partition limits are low",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    root = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = root.get("dataset", {})
    values = dataset.get("dolphindb", {})
    source = MinuteDolphinDBConfig.from_mapping(dataset, values)
    if args.chunk_days is not None:
        if args.chunk_days < 1:
            raise ValueError("--chunk-days must be positive")
        source = replace(source, daily_aggregate_chunk_days=args.chunk_days)
    memory = MinuteMemMapConfig.from_mapping(dataset.get("memory", {}))
    output = Path(
        args.output
        or root.get("outputs", {}).get("daily_price", "data/daily_price.pkl")
    )

    session = create_dolphindb_session(values)
    try:
        loader = DolphinDBMinuteLoader(source, session)
        time_filter_sql = build_ddb_minute_time_filter(
            memory.minute_sessions, memory.minute_extra_times
        )
        print(
            f"[DailyExport] start range={source.start_date}..{source.end_date} "
            f"chunk_days={source.daily_aggregate_chunk_days} output={output}",
            flush=True,
        )
        daily = loader.build_daily_in_memory(
            source.start_date,
            source.end_date,
            time_filter_sql=time_filter_sql,
        )
        save_daily_price_artifact(
            daily,
            output,
            source="dolphindb_minute_daily_aggregate",
            minute_grid=memory.minute_grid,
            extra_metadata={
                "database": source.database,
                "table": source.table,
                "prices_are_adjusted": source.prices_are_adjusted,
                "daily_aggregate_chunk_days": source.daily_aggregate_chunk_days,
                "minute_sessions": [list(item) for item in memory.minute_sessions],
                "minute_extra_times": list(memory.minute_extra_times),
            },
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
