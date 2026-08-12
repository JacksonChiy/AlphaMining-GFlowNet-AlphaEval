from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dolphindb_minute import MinuteDolphinDBConfig


RAM_CACHE_FORMAT_VERSION = 1


def ram_cache_fingerprint(
    source: MinuteDolphinDBConfig,
    minute_grid: Sequence[str],
    channels: Sequence[str],
) -> str:
    """Return the identity of a reusable eager-RAM snapshot."""
    payload = {
        "cache_format_version": RAM_CACHE_FORMAT_VERSION,
        "database": source.database,
        "table": source.table,
        "start_date": str(pd.Timestamp(source.start_date).date()),
        "end_date": str(pd.Timestamp(source.end_date).date()),
        "prices_are_adjusted": source.prices_are_adjusted,
        "channels": list(channels),
        "minute_grid": list(minute_grid),
        "dtype": "float32",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_numpy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    temporary.replace(path)


def manifest_miss_reason(
    manifest: Mapping[str, Any], expected_fingerprint: str
) -> str | None:
    if not manifest.get("complete"):
        return "incomplete_snapshot"
    if int(manifest.get("cache_format_version", -1)) != RAM_CACHE_FORMAT_VERSION:
        return "format_version_mismatch"
    if manifest.get("fingerprint") != expected_fingerprint:
        return "fingerprint_mismatch"
    return None


def validate_eager_array(
    values: np.ndarray,
    expected_shape: Sequence[int],
    expected_dtype: np.dtype[Any] | type[np.generic],
    label: str,
) -> None:
    if isinstance(values, np.memmap):
        raise TypeError(f"RAM cache {label} must be eagerly loaded")
    if values.shape != tuple(expected_shape) or values.dtype != np.dtype(expected_dtype):
        raise ValueError(
            f"invalid cached {label}: shape={values.shape} dtype={values.dtype}"
        )
