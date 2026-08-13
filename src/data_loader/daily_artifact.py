from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_DAILY_COLUMNS = (
    "date", "code", "open", "high", "low", "close", "volume", "amount", "vwap"
)


def save_daily_price_artifact(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    source: str,
    minute_grid: Sequence[str] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Validate and atomically persist the daily panel used by Reward and models."""
    output = Path(output_path)
    missing = sorted(set(REQUIRED_DAILY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Daily price artifact is missing columns: {missing}")

    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["code"] = daily["code"].astype("string").str.strip()
    if daily["date"].isna().any() or daily["code"].isna().any():
        raise ValueError("Daily price artifact contains invalid date/code keys")
    if daily.duplicated(["date", "code"]).any():
        raise ValueError("Daily price artifact contains duplicate date/code keys")
    for column in REQUIRED_DAILY_COLUMNS[2:]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    daily = daily.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    if daily.empty:
        raise ValueError("Daily price artifact is empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    daily.to_pickle(temporary)
    temporary.replace(output)
    digest_builder = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    metadata = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "path": str(output),
        "sha256": digest,
        "rows": int(len(daily)),
        "dates": int(daily["date"].nunique()),
        "stocks": int(daily["code"].nunique()),
        "start_date": str(daily["date"].min().date()),
        "end_date": str(daily["date"].max().date()),
        "columns": list(daily.columns),
        "minute_grid": list(minute_grid or ()),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata_temporary.replace(metadata_path)
    print(
        f"[DailyArtifact] saved path={output} rows={len(daily):,} "
        f"dates={metadata['dates']:,} stocks={metadata['stocks']:,} "
        f"range={metadata['start_date']}..{metadata['end_date']}",
        flush=True,
    )
    return output
