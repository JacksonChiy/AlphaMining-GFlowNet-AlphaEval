"""Download point-in-time index weights once and persist them locally.

``rqdatac.index_weights`` returns monthly source weights expanded to every
trading day for a date-range query.  This module deliberately makes one range
request per index; all later research and backtests must use the local file.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .universe import INDEX_SPECS, normalize_order_book_id


def normalize_weight_history(
    index_key: str,
    index_code: str,
    index_name: str,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Convert an RQData range response into a validated long table.

    RQData weights are rounded and may sum to values such as ``1.0001``.
    We normalize each date to exactly one so the persisted values can be used
    directly as benchmark portfolio weights.
    """
    if history is None or history.empty:
        raise ValueError(f"RQData returned no index weights for {index_code}")
    if not isinstance(history, pd.DataFrame):
        raise TypeError(
            "A range index_weights query must return a pandas.DataFrame; "
            f"received {type(history).__name__}"
        )

    frame = history.reset_index()
    required = {"date", "order_book_id", "weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Index weight response missing columns: {sorted(missing)}")

    frame = frame.loc[:, ["date", "order_book_id", "weight"]].rename(
        columns={"order_book_id": "code", "weight": "benchmark_weight"}
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].map(normalize_order_book_id)
    frame["benchmark_weight"] = pd.to_numeric(
        frame["benchmark_weight"], errors="coerce"
    )
    finite = np.isfinite(frame["benchmark_weight"].to_numpy(dtype=float))
    if not finite.all() or (frame["benchmark_weight"] < 0).any():
        raise ValueError("Index weights must be finite and non-negative")
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Index weight response contains duplicate date/code rows")

    daily_sum = frame.groupby("date")["benchmark_weight"].transform("sum")
    if (daily_sum <= 0).any():
        raise ValueError("Index weight response contains a date with zero total weight")
    frame["benchmark_weight"] = frame["benchmark_weight"] / daily_sum
    frame.insert(1, "index_key", index_key)
    frame.insert(2, "index_code", index_code)
    frame.insert(3, "index_name", index_name)
    return frame.sort_values(["date", "index_code", "code"]).reset_index(drop=True)


def _rqdatac_provider(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import rqdatac
    except ImportError as exc:
        raise RuntimeError(
            "rqdatac is required only for the one-time index-weight download. "
            "Run this command in the locally authorized RQData environment."
        ) from exc
    if not rqdatac.initialized():
        rqdatac.init()
    return rqdatac.index_weights(
        index_code,
        date=None,
        start_date=start_date,
        end_date=end_date,
        market="cn",
    )


def fetch_and_save_weights(
    output_path: str | Path,
    start_date: str,
    end_date: str,
    index_specs: Mapping[str, Mapping[str, Any]] = INDEX_SPECS,
    provider: Callable[[str, str, str], pd.DataFrame] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch each index once and atomically persist a compressed long table."""
    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(
            f"Index weight file already exists: {output.resolve()}. "
            "Reuse it downstream; pass --force only for an intentional refresh."
        )
    provider = provider or _rqdatac_provider
    frames: list[pd.DataFrame] = []
    for index_key, spec in index_specs.items():
        index_code = str(spec["order_book_id"])
        index_name = str(spec.get("name", index_key))
        print(
            f"[IndexWeight] fetch index={index_key} code={index_code} "
            f"start={start_date} end={end_date}",
            flush=True,
        )
        frame = normalize_weight_history(
            index_key,
            index_code,
            index_name,
            provider(index_code, start_date, end_date),
        )
        frames.append(frame)
        print(
            f"[IndexWeight] done index={index_key} dates={frame['date'].nunique()} "
            f"rows={len(frame):,}",
            flush=True,
        )

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["date", "index_code", "code"]
    )
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    compression = "gzip" if output.name.endswith(".gz") else None
    result.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, output)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source": "rqdatac.index_weights",
        "source_frequency": "monthly, forward-filled by rqdatac to trading days",
        "daily_weights_normalized": True,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "output": str(output.resolve()),
        "rows": int(len(result)),
        "api_calls": int(len(index_specs)),
        "indexes": {
            key: {
                "order_book_id": spec["order_book_id"],
                "name": spec.get("name", key),
                "dates": int(result.loc[result["index_key"] == key, "date"].nunique()),
                "rows": int((result["index_key"] == key).sum()),
            }
            for key, spec in index_specs.items()
        },
    }
    metadata_path = output.with_name(output.name + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[IndexWeight] saved rows={len(result):,} file={output.resolve()}", flush=True)
    return result


def load_index_weights(path: str | Path) -> pd.DataFrame:
    """Load locally persisted weights without making an RQData request."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Index weight file not found: {source.resolve()}. "
            "Run python -m src.index_enhancement.weights once first."
        )
    frame = pd.read_csv(source)
    required = {
        "date",
        "index_key",
        "index_code",
        "index_name",
        "code",
        "benchmark_weight",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Index weight file missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].map(normalize_order_book_id)
    frame["benchmark_weight"] = pd.to_numeric(
        frame["benchmark_weight"], errors="coerce"
    )
    if frame.duplicated(["date", "index_code", "code"]).any():
        raise ValueError("Index weight file contains duplicate date/index/code rows")
    weights = frame["benchmark_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Index weight file contains invalid weights")
    daily_sums = frame.groupby(["date", "index_code"])["benchmark_weight"].sum()
    if not np.allclose(daily_sums.to_numpy(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Index weights do not sum to one for every date/index")
    return frame.sort_values(["date", "index_code", "code"]).reset_index(drop=True)


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="一次性下载并保存三类指数历史权重")
    parser.add_argument("--config", default="configs/index_enhancement/default.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root_config = _load_config(args.config)
    config = root_config.get("weight_data", {})
    fetch_and_save_weights(
        output_path=args.output or config.get("file", "data/index_weights.csv.gz"),
        start_date=args.start_date or config.get("start_date", "2020-01-01"),
        end_date=args.end_date or config.get("end_date", "2026-07-27"),
        index_specs=root_config.get("indexes", INDEX_SPECS),
        force=args.force,
    )


if __name__ == "__main__":
    main()
