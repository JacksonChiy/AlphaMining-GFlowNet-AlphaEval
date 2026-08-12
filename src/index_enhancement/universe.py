"""Fetch index constituents once and persist them for all downstream stages."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


INDEX_SPECS = {
    "csi300": {"order_book_id": "000300.XSHG", "name": "沪深300"},
    "csi500": {"order_book_id": "000905.XSHG", "name": "中证500"},
    "csi1000": {"order_book_id": "000852.XSHG", "name": "中证1000"},
}


def normalize_order_book_id(value: Any) -> str:
    value = str(value).strip().upper()
    replacements = {".SH": ".XSHG", ".SZ": ".XSHE", ".BJ": ".XBSE"}
    for source, target in replacements.items():
        if value.endswith(source):
            return value[: -len(source)] + target
    if re.fullmatch(r"\d{6}", value):
        if value.startswith(("5", "6", "9")):
            return f"{value}.XSHG"
        if value.startswith(("4", "8")):
            return f"{value}.XBSE"
        return f"{value}.XSHE"
    return value


def normalize_component_history(
    index_key: str,
    index_code: str,
    index_name: str,
    history: Mapping[Any, Sequence[str] | tuple[Sequence[str], Any]],
) -> pd.DataFrame:
    """Convert rqdatac's date-to-members mapping into a stable long table."""
    rows: list[dict[str, Any]] = []
    for raw_date, raw_members in sorted(history.items(), key=lambda item: pd.Timestamp(item[0])):
        members = raw_members[0] if isinstance(raw_members, tuple) else raw_members
        trade_date = pd.Timestamp(raw_date).normalize()
        for member in members:
            rows.append(
                {
                    "date": trade_date,
                    "index_key": index_key,
                    "index_code": index_code,
                    "index_name": index_name,
                    "code": normalize_order_book_id(member),
                }
            )
    if not rows:
        raise ValueError(f"RQData returned no components for {index_code}")
    result = pd.DataFrame(rows).drop_duplicates(["date", "index_code", "code"])
    return result.sort_values(["date", "index_code", "code"]).reset_index(drop=True)


def _rqdatac_provider(index_code: str, start_date: str, end_date: str) -> Mapping:
    try:
        import rqdatac
    except ImportError as exc:
        raise RuntimeError(
            "rqdatac is only required for the one-time constituent download. "
            "Run this command in the locally authorized RQData environment."
        ) from exc
    if not rqdatac.initialized():
        rqdatac.init()
    data = rqdatac.index_components(
        index_code,
        date=None,
        start_date=start_date,
        end_date=end_date,
        market="cn",
        return_create_tm=False,
    )
    return data or {}


def fetch_and_save_components(
    output_path: str | Path,
    start_date: str,
    end_date: str,
    index_specs: Mapping[str, Mapping[str, Any]] = INDEX_SPECS,
    provider: Callable[[str, str, str], Mapping] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Call RQData once per index, then atomically persist one compressed CSV."""
    output = Path(output_path)
    if output.exists() and not force:
        raise FileExistsError(
            f"Component file already exists: {output.resolve()}. "
            "Downstream stages should reuse it; pass --force only for an intentional refresh."
        )
    provider = provider or _rqdatac_provider
    frames = []
    for index_key, spec in index_specs.items():
        index_code = str(spec["order_book_id"])
        index_name = str(spec.get("name", index_key))
        print(
            f"[Universe] fetch index={index_key} code={index_code} "
            f"start={start_date} end={end_date}",
            flush=True,
        )
        history = provider(index_code, start_date, end_date)
        frame = normalize_component_history(index_key, index_code, index_name, history)
        frames.append(frame)
        print(
            f"[Universe] done index={index_key} dates={frame['date'].nunique()} "
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
        "source": "rqdatac.index_components",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "output": str(output.resolve()),
        "rows": int(len(result)),
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
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Universe] saved rows={len(result):,} file={output.resolve()}", flush=True)
    return result


def load_components(path: str | Path) -> pd.DataFrame:
    """Load and validate the persisted constituent table without any API call."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Index component file not found: {source.resolve()}. "
            "Run python -m src.index_enhancement.universe once first."
        )
    frame = pd.read_csv(source)
    required = {"date", "index_key", "index_code", "index_name", "code"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Component file missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].map(normalize_order_book_id)
    if frame.duplicated(["date", "index_code", "code"]).any():
        raise ValueError("Component file contains duplicate date/index/code rows")
    return frame.sort_values(["date", "index_code", "code"]).reset_index(drop=True)


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="一次性下载并保存三类指数历史成分股")
    parser.add_argument("--config", default="configs/index_enhancement/default.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config).get("component_data", {})
    specs = _load_config(args.config).get("indexes", INDEX_SPECS)
    fetch_and_save_components(
        output_path=args.output or config.get("file", "data/index_components.csv.gz"),
        start_date=args.start_date or config.get("start_date", "2020-01-01"),
        end_date=args.end_date or config.get("end_date", "2026-07-27"),
        index_specs=specs,
        force=args.force,
    )


if __name__ == "__main__":
    main()
