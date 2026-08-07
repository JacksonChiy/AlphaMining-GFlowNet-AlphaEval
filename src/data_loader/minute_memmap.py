from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.operators.minute import EPS, validate_minute_data

from .dolphindb_minute import DolphinDBMinuteLoader
from .minute_quality_audit import (
    DEFAULT_MINUTE_EXTRA_TIMES,
    DEFAULT_MINUTE_SESSIONS,
    build_expected_minute_grid,
)


MEMMAP_CHANNELS = (
    "open", "high", "low", "close", "vol", "amount",
    "ret", "vwap", "hl_pct", "bar_pos", "amihud", "rv",
    "signed_vol", "signed_amt", "typical", "vwap_cum", "twap", "obv", "pvt",
)
MEMMAP_FORMAT_VERSION = 1
FAST_LOAD_SOURCE_COLUMNS = (
    "sym", "time", "open", "high", "low", "close", "volume", "amount",
)
LOAD_TIMING_KEYS = (
    "ddb_query_s",
    "decode_index_s",
    "base_matrix_s",
    "numpy_feature_s",
    "memory_write_s",
)


@dataclass(frozen=True)
class MinuteMemMapConfig:
    root: Path
    block_cache_dir: Path
    expected_minutes: int = 241
    minute_sessions: tuple[tuple[str, str], ...] = DEFAULT_MINUTE_SESSIONS
    minute_extra_times: tuple[str, ...] = DEFAULT_MINUTE_EXTRA_TIMES
    stock_tile_size: int = 256
    workers: int = 1
    build_workers: int = 1
    flush_every_days: int = 1
    reward_chunk_days: int = 20
    reward_blocks_per_task: int = 2
    reward_group_by_channels: bool = True
    reward_cache_commit_tasks: int = 10
    reward_backend: str = "numpy"
    reward_parallel_backend: str = "loky"
    numpy_fallback: bool = True
    force_rebuild: bool = False

    def __post_init__(self) -> None:
        if min(
            self.expected_minutes, self.stock_tile_size, self.workers,
            self.build_workers, self.flush_every_days, self.reward_chunk_days,
            self.reward_blocks_per_task,
            self.reward_cache_commit_tasks,
        ) < 1:
            raise ValueError("MemMap size settings must be positive")
        actual = len(build_expected_minute_grid(
            self.minute_sessions, self.minute_extra_times
        ))
        if actual != self.expected_minutes:
            raise ValueError(
                f"Configured minute grid contains {actual} points, "
                f"expected_minutes={self.expected_minutes}"
            )
        if self.reward_backend not in {"numpy", "pandas"}:
            raise ValueError("reward_backend must be 'numpy' or 'pandas'")
        if self.reward_parallel_backend not in {"loky", "threading"}:
            raise ValueError("reward_parallel_backend must be 'loky' or 'threading'")

    @property
    def minute_grid(self) -> tuple[str, ...]:
        return build_expected_minute_grid(
            self.minute_sessions, self.minute_extra_times
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MinuteMemMapConfig":
        root_env = str(values.get("root_env", "ALPHAMINING_MEMMAP_DIR"))
        cache_env = str(
            values.get("block_cache_env", "ALPHAMINING_BLOCK_CACHE_DIR")
        )
        root = values.get("root") or os.environ.get(root_env)
        if not root:
            raise ValueError(f"Missing minute MemMap directory: set {root_env}")
        cache = values.get("block_cache_dir") or os.environ.get(cache_env)
        cache_path = Path(str(cache)) if cache else Path(str(root)) / "block_cache"
        raw_sessions = values.get("minute_sessions", DEFAULT_MINUTE_SESSIONS)
        minute_sessions = tuple(
            (str(item[0]), str(item[1])) for item in raw_sessions
        )
        minute_extra_times = tuple(
            str(value) for value in values.get(
                "minute_extra_times", DEFAULT_MINUTE_EXTRA_TIMES
            )
        )
        return cls(
            root=Path(str(root)),
            block_cache_dir=cache_path,
            expected_minutes=int(values.get("expected_minutes", 241)),
            minute_sessions=minute_sessions,
            minute_extra_times=minute_extra_times,
            stock_tile_size=int(values.get("stock_tile_size", 256)),
            workers=int(values.get("workers", 1)),
            build_workers=int(values.get("build_workers", 1)),
            flush_every_days=int(values.get("flush_every_days", 1)),
            reward_chunk_days=int(values.get("reward_chunk_days", 20)),
            reward_blocks_per_task=int(values.get("reward_blocks_per_task", 2)),
            reward_group_by_channels=bool(values.get("reward_group_by_channels", True)),
            reward_cache_commit_tasks=int(values.get("reward_cache_commit_tasks", 10)),
            reward_backend=str(values.get("reward_backend", "numpy")).lower(),
            reward_parallel_backend=str(
                values.get("reward_parallel_backend", "loky")
            ).lower(),
            numpy_fallback=bool(values.get("numpy_fallback", True)),
            force_rebuild=bool(values.get("force_rebuild", False)),
        )


def _time_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.time()
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Cannot normalize minute time value: {value!r}")
    return parsed.strftime("%H:%M:%S")


def _configured_minute_numbers(minute_lookup: Mapping[str, int]) -> np.ndarray:
    lookup = np.full(24 * 60, -1, dtype=np.int16)
    for value, index in minute_lookup.items():
        parsed = pd.Timestamp(f"2000-01-01 {value}")
        lookup[parsed.hour * 60 + parsed.minute] = int(index)
    return lookup


def _minute_numbers(values: pd.Series) -> np.ndarray:
    """Decode common DolphinDB time representations without per-row pandas parsing."""
    if pd.api.types.is_timedelta64_dtype(values.dtype):
        raw = values.to_numpy(dtype="timedelta64[m]").astype(np.int64)
        raw[values.isna().to_numpy()] = -1
        return raw
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        parsed = values.dt
        output = (parsed.hour * 60 + parsed.minute).to_numpy(dtype=np.float64)
        return np.where(np.isfinite(output), output, -1).astype(np.int64)
    if pd.api.types.is_numeric_dtype(values.dtype):
        raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        maximum = float(np.max(np.abs(finite))) if finite.size else 0.0
        divisor = 1.0
        if maximum > 86_400_000:
            divisor = 60_000_000_000.0
        elif maximum > 86_400:
            divisor = 60_000.0
        elif maximum > 1_440:
            divisor = 60.0
        return np.where(np.isfinite(raw), np.floor(raw / divisor), -1).astype(np.int64)

    def decode(value: Any) -> int:
        if value is None or value is pd.NaT:
            return -1
        if hasattr(value, "hour") and hasattr(value, "minute"):
            return int(value.hour) * 60 + int(value.minute)
        text = str(value).strip()
        match = text.rsplit(" ", 1)[-1].split(":")
        if len(match) >= 2 and match[0][-2:].isdigit() and match[1][:2].isdigit():
            return int(match[0][-2:]) * 60 + int(match[1][:2])
        parsed = pd.to_datetime(text, errors="coerce")
        return -1 if pd.isna(parsed) else int(parsed.hour) * 60 + int(parsed.minute)

    return np.fromiter((decode(value) for value in values.array), dtype=np.int64, count=len(values))


def _numeric_values(values: pd.Series) -> np.ndarray:
    raw = values.to_numpy(copy=False)
    if np.issubdtype(raw.dtype, np.number):
        return raw.astype(np.float64, copy=False)
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _safe_divide_array(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.divide(
        left,
        right,
        out=np.full(left.shape, np.nan, dtype=np.float64),
        where=np.isfinite(left) & np.isfinite(right) & (np.abs(right) > EPS),
    )


def _previous_observed(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    minute_positions = np.arange(values.shape[0], dtype=np.int32)[:, None]
    last_seen = np.maximum.accumulate(
        np.where(mask, minute_positions, -1), axis=0
    )
    previous = np.empty_like(last_seen)
    previous[0, :] = -1
    previous[1:, :] = last_seen[:-1, :]
    output = np.take_along_axis(values, np.maximum(previous, 0), axis=0)
    output[previous < 0] = np.nan
    return output


def _skipna_cumsum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    finite = mask & np.isfinite(values)
    cumulative = np.cumsum(np.where(finite, values, 0.0), axis=0)
    return np.where(finite, cumulative, np.nan)


def _build_dense_minute_channels(
    frame: pd.DataFrame,
    stock_codes: Sequence[str],
    minute_lookup: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float | int]]:
    """Map one raw DDB day directly to dense arrays and derive all stored leaves."""
    missing = sorted(set(FAST_LOAD_SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"DolphinDB fast minute query is missing columns: {missing}")
    decode_started = time.perf_counter()
    minute_numbers = _minute_numbers(frame["time"])
    minute_number_lookup = _configured_minute_numbers(minute_lookup)
    minute_index = np.full(len(frame), -1, dtype=np.int64)
    valid_minute_number = (minute_numbers >= 0) & (minute_numbers < len(minute_number_lookup))
    minute_index[valid_minute_number] = minute_number_lookup[minute_numbers[valid_minute_number]]

    categories = pd.Index(np.asarray(stock_codes, dtype=str))
    stock_index = pd.Categorical(frame["sym"], categories=categories).codes.astype(
        np.int64, copy=False
    )
    if np.any(stock_index < 0):
        cleaned = frame["sym"].astype("string").str.strip()
        stock_index = pd.Categorical(cleaned, categories=categories).codes.astype(
            np.int64, copy=False
        )
    numeric = {
        name: _numeric_values(frame[source])
        for name, source in (
            ("open", "open"), ("high", "high"), ("low", "low"),
            ("close", "close"), ("vol", "volume"), ("amount", "amount"),
        )
    }
    valid = (minute_index >= 0) & (stock_index >= 0)
    valid &= np.isfinite(numeric["open"]) & (numeric["open"] > 0)
    valid &= np.isfinite(numeric["high"]) & (numeric["high"] > 0)
    valid &= np.isfinite(numeric["low"]) & (numeric["low"] > 0)
    valid &= np.isfinite(numeric["close"]) & (numeric["close"] > 0)
    valid &= numeric["low"] <= numeric["high"]
    if not valid.any():
        raise ValueError("DolphinDB fast minute query produced no valid configured rows")
    numeric["vol"] = np.clip(numeric["vol"], 0.0, None)
    numeric["amount"] = np.clip(numeric["amount"], 0.0, None)
    decode_index_s = time.perf_counter() - decode_started

    base_started = time.perf_counter()
    shape = (len(minute_lookup), len(categories))
    mask = np.zeros(shape, dtype=bool)
    flat_index = minute_index[valid] * shape[1] + stock_index[valid]
    mask.ravel()[flat_index] = True
    channels: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close", "vol", "amount"):
        output = np.full(shape, np.nan, dtype=np.float64)
        output.ravel()[flat_index] = numeric[name][valid]
        channels[name] = output
    base_matrix_s = time.perf_counter() - base_started

    feature_started = time.perf_counter()
    high, low = channels["high"], channels["low"]
    close, vol, amount = channels["close"], channels["vol"], channels["amount"]
    previous_close = _previous_observed(close, mask)
    ret = _safe_divide_array(close, previous_close) - 1.0
    vwap = _safe_divide_array(amount, vol)
    hl_pct = _safe_divide_array(high - low, np.abs(close))
    bar_pos = _safe_divide_array(close - low, high - low)
    amihud = _safe_divide_array(np.abs(ret), np.abs(amount))
    direction = np.where(np.isfinite(ret), np.sign(ret), 0.0)
    cumulative_amount = _skipna_cumsum(amount, mask)
    cumulative_vol = _skipna_cumsum(vol, mask)
    finite_close = mask & np.isfinite(close)
    close_sum = np.cumsum(np.where(finite_close, close, 0.0), axis=0)
    close_count = np.cumsum(finite_close, axis=0)
    twap = np.divide(
        close_sum,
        close_count,
        out=np.full(shape, np.nan, dtype=np.float64),
        where=finite_close & (close_count > 0),
    )
    channels.update({
        "ret": ret,
        "vwap": vwap,
        "hl_pct": hl_pct,
        "bar_pos": bar_pos,
        "amihud": amihud,
        "rv": np.square(ret),
        "signed_vol": direction * vol,
        "signed_amt": direction * amount,
        "typical": (high + low + close) / 3.0,
        "vwap_cum": _safe_divide_array(cumulative_amount, cumulative_vol),
        "twap": twap,
        "obv": _skipna_cumsum(direction * vol, mask),
        "pvt": _skipna_cumsum(ret * vol, mask),
    })
    for name in MEMMAP_CHANNELS:
        channels[name] = np.where(mask, channels[name], np.nan)
    numpy_feature_s = time.perf_counter() - feature_started
    return channels, mask, {
        "decode_index_s": decode_index_s,
        "base_matrix_s": base_matrix_s,
        "numpy_feature_s": numpy_feature_s,
        "valid_rows": int(valid.sum()),
        "excluded_rows": int(len(frame) - valid.sum()),
    }


def _source_fingerprint(
    loader: DolphinDBMinuteLoader,
    dates: Sequence[pd.Timestamp],
    minute_grid: Sequence[str],
) -> str:
    payload = {
        "format_version": MEMMAP_FORMAT_VERSION,
        "database": loader.config.database,
        "table": loader.config.table,
        "start_date": str(pd.Timestamp(dates[0]).date()),
        "end_date": str(pd.Timestamp(dates[-1]).date()),
        "prices_are_adjusted": loader.config.prices_are_adjusted,
        "channels": MEMMAP_CHANNELS,
        "minute_grid": list(minute_grid),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DolphinDBMinuteMemMapBuilder:
    """One-time DDB extraction into local yearly/channel float32 MemMaps."""

    def __init__(
        self,
        loader: DolphinDBMinuteLoader,
        config: MinuteMemMapConfig,
        loader_factory: Callable[[], DolphinDBMinuteLoader] | None = None,
    ) -> None:
        self.loader = loader
        self.config = config
        self.loader_factory = loader_factory
        if config.build_workers > 1 and loader_factory is None:
            raise ValueError(
                "MemMap build_workers > 1 requires a loader_factory so each thread "
                "uses an independent DolphinDB session"
            )

    def build(self) -> Path:
        audit = self.loader.audit()
        if not audit.passed:
            raise ValueError(
                "MemMap build requires adjusted OHLC; set prices_are_adjusted=true only "
                "after confirming the source convention"
            )
        dates = self.loader.load_trade_dates(
            self.loader.config.start_date, self.loader.config.end_date
        )
        root = self.config.root
        root.mkdir(parents=True, exist_ok=True)
        audit_path = root / "field_audit.json"
        self._write_json(audit_path, audit.to_dict())
        manifest_path = root / "manifest.json"
        configured_grid = self.config.minute_grid
        fingerprint = _source_fingerprint(self.loader, dates, configured_grid)
        self.ensure_duplicate_audit(dates, fingerprint)
        existing = self._read_json(manifest_path)
        if (
            existing
            and existing.get("complete")
            and existing.get("fingerprint") == fingerprint
            and not self.config.force_rebuild
        ):
            print(f"[MemMapBuild] cache_reused root={root}", flush=True)
            return manifest_path
        if existing and existing.get("fingerprint") != fingerprint and not self.config.force_rebuild:
            raise ValueError(
                "Existing MemMap source fingerprint differs. Use a new directory or set "
                "memmap.force_rebuild=true after preserving any needed cache."
            )

        daily_path = root / "daily_price.csv.gz"
        reuse_metadata = bool(
            existing
            and existing.get("fingerprint") == fingerprint
            and not self.config.force_rebuild
            and daily_path.exists()
            and (root / "stocks.npy").exists()
            and (root / "minute_grid.npy").exists()
        )
        if reuse_metadata:
            stocks = np.load(root / "stocks.npy", allow_pickle=False)
            minute_grid = tuple(
                np.load(root / "minute_grid.npy", allow_pickle=False).astype(str)
            )
            manifest = existing
            print(
                f"[MemMapBuild] metadata_reused resume=true root={root}", flush=True
            )
        else:
            minute_grid = self._load_minute_grid(dates)
            daily = self.loader.build_daily_in_memory(
                self.loader.config.start_date,
                self.loader.config.end_date,
                time_filter_sql=self._ddb_time_filter_sql(),
            )
            daily.to_csv(daily_path, index=False, compression="gzip")
            stocks = np.array(sorted(daily["code"].astype(str).unique()), dtype=str)
            np.save(root / "stocks.npy", stocks, allow_pickle=False)
            np.save(
                root / "minute_grid.npy", np.array(minute_grid, dtype=str), allow_pickle=False
            )
            manifest = {
                "format_version": MEMMAP_FORMAT_VERSION,
                "complete": False,
                "fingerprint": fingerprint,
                "source": {
                    "database": self.loader.config.database,
                    "table": self.loader.config.table,
                    "start_date": str(dates[0].date()),
                    "end_date": str(dates[-1].date()),
                    "prices_are_adjusted": self.loader.config.prices_are_adjusted,
                },
                "field_audit_file": audit_path.name,
                "minute_grid_audit_file": "minute_grid_audit.json",
                "minute_sessions": [list(item) for item in self.config.minute_sessions],
                "minute_extra_times": list(self.config.minute_extra_times),
                "layout": "year/channel -> (n_days, n_minutes, n_stocks)",
                "dtype": "float32",
                "channels": list(MEMMAP_CHANNELS),
                "n_minutes": len(minute_grid),
                "n_stocks": len(stocks),
                "stocks_file": "stocks.npy",
                "minute_grid_file": "minute_grid.npy",
                "daily_file": daily_path.name,
                "years": {},
            }
        manifest["field_audit_file"] = audit_path.name
        self._write_json(manifest_path, manifest)
        stock_lookup = {code: index for index, code in enumerate(stocks.tolist())}
        minute_lookup = {value: index for index, value in enumerate(minute_grid)}

        for year in sorted({date.year for date in dates}):
            year_dates = tuple(date for date in dates if date.year == year)
            self._build_year(
                year, year_dates, stocks, stock_lookup, minute_lookup, fingerprint
            )
            manifest["years"][str(year)] = {
                "dates_file": f"{year}/dates.npy",
                "n_days": len(year_dates),
                "shape": [len(year_dates), len(minute_grid), len(stocks)],
                "channels": {
                    channel: f"{year}/{channel}.npy" for channel in MEMMAP_CHANNELS
                },
                "valid_mask": f"{year}/valid_mask.npy",
            }
            self._write_json(manifest_path, manifest)
        manifest["complete"] = True
        self._write_json(manifest_path, manifest)
        print(
            f"[MemMapBuild] complete root={root} dates={len(dates):,} "
            f"stocks={len(stocks):,} channels={len(MEMMAP_CHANNELS)}",
            flush=True,
        )
        return manifest_path

    def ensure_duplicate_audit(
        self,
        dates: Sequence[pd.Timestamp],
        fingerprint: str,
    ) -> None:
        audit_path = self.config.root / "duplicate_key_audit.json"
        existing = self._read_json(audit_path)
        if (
            existing
            and existing.get("complete") is True
            and existing.get("fingerprint") == fingerprint
        ):
            print(
                f"[MemMapBuild] duplicate_audit_reused path={audit_path}",
                flush=True,
            )
            return
        self.loader.assert_no_duplicate_minute_keys(
            dates[0], dates[-1], time_filter_sql=self._ddb_time_filter_sql()
        )
        self._write_json(audit_path, {
            "complete": True,
            "fingerprint": fingerprint,
            "start_date": str(pd.Timestamp(dates[0]).date()),
            "end_date": str(pd.Timestamp(dates[-1]).date()),
            "duplicate_keys": 0,
            "scope": "configured_minute_grid",
        })

    def _load_minute_grid(self, dates: Sequence[pd.Timestamp]) -> tuple[str, ...]:
        sample_dates = []
        for year in sorted({date.year for date in dates}):
            sample_dates.append(next(date for date in dates if date.year == year))
        configured = self.config.minute_grid
        expected = set(configured)
        sample_audit: list[dict[str, Any]] = []
        for date in sample_dates:
            script = (
                "select distinct time as minuteTime from "
                f"{self.loader.table_expression} where date = {date:%Y.%m.%d} "
                "order by minuteTime"
            )
            result = pd.DataFrame(self.loader.session.run(script))
            if "minuteTime" not in result:
                raise ValueError("DolphinDB minute-grid query returned no minuteTime column")
            observed = {
                _time_key(value) for value in result["minuteTime"].dropna()
            }
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            sample_audit.append({
                "date": str(date.date()),
                "observed_minutes": len(observed),
                "missing_configured_times": missing,
                "excluded_source_times": extra,
            })
        audit = {
            "policy": "explicit_configured_grid",
            "expected_minutes": self.config.expected_minutes,
            "sessions": [list(item) for item in self.config.minute_sessions],
            "extra_times": list(self.config.minute_extra_times),
            "selected_grid": list(configured),
            "sample_dates": sample_audit,
        }
        self._write_json(self.config.root / "minute_grid_audit.json", audit)
        missing_samples = [item for item in sample_audit if item["missing_configured_times"]]
        if missing_samples:
            raise ValueError(
                "Configured 241-point minute grid is absent on sampled dates. "
                f"See {self.config.root / 'minute_grid_audit.json'} for exact missing times."
            )
        excluded = sorted({
            value for item in sample_audit for value in item["excluded_source_times"]
        })
        print(
            f"[MemMapBuild] minute_grid_ready count={len(configured)} "
            f"first={configured[0]} last={configured[-1]} "
            f"excluded_source_times={len(excluded)}",
            flush=True,
        )
        return configured

    def _ddb_time_filter_sql(self) -> str:
        clauses = [
            f"minute(time) = {self._ddb_minute_literal(value)}"
            for value in self.config.minute_extra_times
        ]
        clauses.extend(
            "(minute(time) >= "
            f"{self._ddb_minute_literal(start)} and minute(time) <= "
            f"{self._ddb_minute_literal(end)})"
            for start, end in self.config.minute_sessions
        )
        return " or ".join(clauses)

    @staticmethod
    def _ddb_minute_literal(value: str) -> str:
        parsed = pd.Timestamp(f"2000-01-01 {value}")
        if parsed.second or parsed.microsecond:
            raise ValueError(f"Minute-grid values must align to whole minutes: {value}")
        return parsed.strftime("%H:%M") + "m"

    def _build_year(
        self,
        year: int,
        dates: Sequence[pd.Timestamp],
        stocks: np.ndarray,
        stock_lookup: dict[str, int],
        minute_lookup: dict[str, int],
        fingerprint: str,
    ) -> None:
        year_dir = self.config.root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        dates_array = np.array([str(date.date()) for date in dates], dtype=str)
        np.save(year_dir / "dates.npy", dates_array, allow_pickle=False)
        shape = (len(dates), len(minute_lookup), len(stocks))
        progress_path = year_dir / "progress.json"
        progress = self._read_json(progress_path) or {}
        resume = (
            not self.config.force_rebuild
            and
            progress.get("fingerprint") == fingerprint
            and progress.get("shape") == list(shape)
        )
        completed_indices: set[int] = set()
        if resume:
            saved_indices = progress.get("completed_day_indices")
            if isinstance(saved_indices, list):
                completed_indices = {int(value) for value in saved_indices}
            else:
                completed_indices = set(range(int(progress.get("completed_days", 0))))
        mode = "r+" if resume and all(
            (year_dir / f"{channel}.npy").exists() for channel in MEMMAP_CHANNELS
        ) and (year_dir / "valid_mask.npy").exists() else "w+"
        arrays = {
            channel: np.lib.format.open_memmap(
                year_dir / f"{channel}.npy", mode=mode, dtype=np.float32, shape=shape
            )
            for channel in MEMMAP_CHANNELS
        }
        mask = np.lib.format.open_memmap(
            year_dir / "valid_mask.npy", mode=mode, dtype=np.uint8, shape=shape
        )
        if mode == "w+":
            completed_indices.clear()
        pending_indices = [
            index for index in range(len(dates)) if index not in completed_indices
        ]
        print(
            f"[MemMapBuild] year_start year={year} days={len(dates)} "
            f"completed={len(completed_indices)} pending={len(pending_indices)} "
            f"build_workers={self.config.build_workers} shape={shape}",
            flush=True,
        )
        if not pending_indices:
            return
        batch_size = max(self.config.build_workers, self.config.flush_every_days)
        if self.config.build_workers == 1:
            for offset in range(0, len(pending_indices), batch_size):
                batch = pending_indices[offset:offset + batch_size]
                results = [
                    self._write_day(
                        self.loader, index, pd.Timestamp(dates[index]), arrays, mask,
                        stock_lookup, minute_lookup,
                    )
                    for index in batch
                ]
                self._commit_batch(
                    year, dates, results, arrays, mask, completed_indices,
                    progress_path, fingerprint, shape,
                )
            return

        worker_local = threading.local()
        worker_loaders: list[DolphinDBMinuteLoader] = []
        worker_lock = threading.Lock()

        def initialize_worker() -> None:
            assert self.loader_factory is not None
            worker_local.loader = self.loader_factory()
            with worker_lock:
                worker_loaders.append(worker_local.loader)

        def process_day(day_index: int) -> dict[str, Any]:
            return self._write_day(
                worker_local.loader,
                day_index,
                pd.Timestamp(dates[day_index]),
                arrays,
                mask,
                stock_lookup,
                minute_lookup,
            )

        try:
            with ThreadPoolExecutor(
                max_workers=self.config.build_workers,
                thread_name_prefix="ddb-memmap",
                initializer=initialize_worker,
            ) as executor:
                for offset in range(0, len(pending_indices), batch_size):
                    batch = pending_indices[offset:offset + batch_size]
                    futures = {executor.submit(process_day, index): index for index in batch}
                    results = [future.result() for future in as_completed(futures)]
                    self._commit_batch(
                        year, dates, results, arrays, mask, completed_indices,
                        progress_path, fingerprint, shape,
                    )
        finally:
            for worker_loader in worker_loaders:
                worker_loader.session.close()

    def _write_day(
        self,
        loader: DolphinDBMinuteLoader,
        day_index: int,
        date: pd.Timestamp,
        arrays: Mapping[str, np.ndarray],
        mask: np.ndarray,
        stock_lookup: Mapping[str, int],
        minute_lookup: Mapping[str, int],
    ) -> dict[str, Any]:
        total_started = time.perf_counter()
        stage_started = total_started
        raw = loader.session.run(loader.build_data_sql(
            date,
            date,
            columns=FAST_LOAD_SOURCE_COLUMNS,
            time_filter_sql=self._ddb_time_filter_sql(),
            order_by=False,
        ))
        ddb_query_s = time.perf_counter() - stage_started

        frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        raw_rows = len(frame)
        ordered_stocks = np.empty(len(stock_lookup), dtype=object)
        for code, index in stock_lookup.items():
            ordered_stocks[int(index)] = str(code)
        day_channels, day_mask, fast_metrics = _build_dense_minute_channels(
            frame, ordered_stocks, minute_lookup
        )

        stage_started = time.perf_counter()
        for channel, array in arrays.items():
            np.copyto(array[day_index], day_channels[channel], casting="unsafe")
        np.copyto(mask[day_index], day_mask, casting="unsafe")
        memory_write_s = time.perf_counter() - stage_started
        return {
            "day_index": day_index,
            "date": str(date.date()),
            "raw_rows": raw_rows,
            "minute_rows": int(fast_metrics["valid_rows"]),
            "excluded_rows": int(fast_metrics["excluded_rows"]),
            "ddb_query_s": ddb_query_s,
            "decode_index_s": float(fast_metrics["decode_index_s"]),
            "base_matrix_s": float(fast_metrics["base_matrix_s"]),
            "numpy_feature_s": float(fast_metrics["numpy_feature_s"]),
            "memory_write_s": memory_write_s,
            "total_s": time.perf_counter() - total_started,
        }

    def _commit_batch(
        self,
        year: int,
        dates: Sequence[pd.Timestamp],
        results: Sequence[Mapping[str, Any]],
        arrays: Mapping[str, np.memmap],
        mask: np.memmap,
        completed_indices: set[int],
        progress_path: Path,
        fingerprint: str,
        shape: Sequence[int],
    ) -> None:
        for array in arrays.values():
            array.flush()
        mask.flush()
        completed_indices.update(int(result["day_index"]) for result in results)
        contiguous_days = 0
        while contiguous_days in completed_indices:
            contiguous_days += 1
        self._write_json(progress_path, {
            "fingerprint": fingerprint,
            "shape": list(shape),
            "completed_days": contiguous_days,
            "completed_day_indices": sorted(completed_indices),
            "complete": len(completed_indices) == len(dates),
        })
        for result in sorted(results, key=lambda item: int(item["day_index"])):
            print(
                f"[MemMapBuild] day_complete year={year} "
                f"day={int(result['day_index']) + 1:03d}/{len(dates):03d} "
                f"date={result['date']} minute_rows={int(result['minute_rows']):,} "
                f"excluded_rows={int(result['excluded_rows']):,} "
                f"timing_s=query:{float(result['ddb_query_s']):.2f},"
                f"decode_index:{float(result['decode_index_s']):.2f},"
                f"base_matrix:{float(result['base_matrix_s']):.2f},"
                f"numpy_feature:{float(result['numpy_feature_s']):.2f},"
                f"write:{float(result['memory_write_s']):.2f},"
                f"total:{float(result['total_s']):.2f}",
                flush=True,
            )
        print(
            f"[MemMapBuild] batch_flushed year={year} "
            f"completed={len(completed_indices):03d}/{len(dates):03d}",
            flush=True,
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


class MinuteMemMapStore:
    """Read-only local MemMap store used by the training process."""

    def __init__(self, config: MinuteMemMapConfig) -> None:
        self.config = config
        manifest_path = config.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Minute MemMap manifest not found: {manifest_path}. Run the build script first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise ValueError("Minute MemMap build is incomplete; resume the build script")
        if int(self.manifest.get("format_version", -1)) != MEMMAP_FORMAT_VERSION:
            raise ValueError("Unsupported minute MemMap format version")
        self.fingerprint = str(self.manifest["fingerprint"])
        self.stocks = np.load(config.root / self.manifest["stocks_file"], allow_pickle=False)
        self.minute_grid = np.load(
            config.root / self.manifest["minute_grid_file"], allow_pickle=False
        )
        self.stock_lookup = {str(code): index for index, code in enumerate(self.stocks)}
        self._arrays: dict[tuple[int, str], np.memmap] = {}
        all_dates = []
        for year, metadata in sorted(self.manifest["years"].items()):
            values = np.load(config.root / metadata["dates_file"], allow_pickle=False)
            all_dates.extend(pd.to_datetime(values).normalize().tolist())
        self.dates = pd.DatetimeIndex(all_dates)

    @property
    def daily_file(self) -> Path:
        return self.config.root / self.manifest["daily_file"]

    def iter_frames(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        stock_tile_size: int | None = None,
    ):
        start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
        tile_size = int(stock_tile_size or self.config.stock_tile_size)
        selected_dates = self.dates[(self.dates >= start) & (self.dates <= end)]
        total_tiles = len(selected_dates) * int(np.ceil(len(self.stocks) / tile_size))
        tile_number = 0
        for date in selected_dates:
            year = int(date.year)
            year_dates = self._year_dates(year)
            day_index = int(year_dates.get_loc(date))
            for stock_start in range(0, len(self.stocks), tile_size):
                stock_end = min(stock_start + tile_size, len(self.stocks))
                tile_number += 1
                frame = self._materialize_frame(
                    year, day_index, date, stock_start, stock_end
                )
                if frame is None:
                    continue
                if tile_number == 1 or tile_number == total_tiles or tile_number % 100 == 0:
                    print(
                        f"[MemMapRead] progress={tile_number}/{total_tiles} "
                        f"date={date.date()} stocks={stock_start}:{stock_end} "
                        f"minute_rows={len(frame):,}",
                        flush=True,
                    )
                yield date, date, frame

    def tile_specs(
        self, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp
    ) -> list[tuple[int, int, int]]:
        start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
        years = sorted({int(date.year) for date in self.dates if start <= date <= end})
        return [
            (year, stock_start, min(stock_start + self.config.stock_tile_size, len(self.stocks)))
            for year in years
            for stock_start in range(0, len(self.stocks), self.config.stock_tile_size)
        ]

    def chunk_specs(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        chunk_days: int | None = None,
    ) -> list[tuple[int, int, int, int, int]]:
        """Return year-local day chunks crossed with stock tiles."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        days = int(chunk_days or self.config.reward_chunk_days)
        specs: list[tuple[int, int, int, int, int]] = []
        for year in sorted(int(value) for value in self.manifest["years"]):
            dates = self._year_dates(year)
            selected = np.flatnonzero((dates >= start) & (dates <= end))
            if not len(selected):
                continue
            for offset in range(0, len(selected), days):
                block = selected[offset:offset + days]
                day_start, day_end = int(block[0]), int(block[-1]) + 1
                for stock_start in range(0, len(self.stocks), self.config.stock_tile_size):
                    specs.append((
                        year, day_start, day_end, stock_start,
                        min(stock_start + self.config.stock_tile_size, len(self.stocks)),
                    ))
        return specs

    def read_numpy_chunk(
        self,
        spec: tuple[int, int, int, int, int],
        channels: Sequence[str],
    ) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, np.ndarray]]:
        """Read only requested channels without constructing a Pandas minute frame."""
        year, day_start, day_end, stock_start, stock_end = spec
        unknown = sorted(set(channels).difference(MEMMAP_CHANNELS))
        if unknown:
            raise KeyError(f"Unknown MemMap channels: {unknown}")
        dates = self._year_dates(year)[day_start:day_end]
        key = np.s_[day_start:day_end, :, stock_start:stock_end]
        mask = np.asarray(self._array(year, "valid_mask")[key], dtype=bool)
        arrays = {
            channel: np.asarray(self._array(year, channel)[key], dtype=np.float32)
            for channel in channels
        }
        return dates, mask, arrays

    def iter_tile_frames(
        self,
        year: int,
        stock_start: int,
        stock_end: int,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ):
        start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
        year_dates = self._year_dates(year)
        for day_index, date in enumerate(year_dates):
            if date < start or date > end:
                continue
            frame = self._materialize_frame(
                year, day_index, date, stock_start, stock_end
            )
            if frame is not None:
                yield date, date, frame

    def _materialize_frame(
        self,
        year: int,
        day_index: int,
        date: pd.Timestamp,
        stock_start: int,
        stock_end: int,
    ) -> pd.DataFrame | None:
        mask = self._array(year, "valid_mask")
        valid = np.asarray(mask[day_index, :, stock_start:stock_end], dtype=bool)
        minute_index, relative_stock_index = np.nonzero(valid)
        if not len(minute_index):
            return None
        absolute_stock_index = relative_stock_index + stock_start
        frame = pd.DataFrame({
            "date": date,
            "datetime": [
                date + pd.Timedelta(str(self.minute_grid[index]))
                for index in minute_index
            ],
            "code": self.stocks[absolute_stock_index].astype(str),
        })
        for channel in MEMMAP_CHANNELS:
            array = self._array(year, channel)
            frame[channel] = np.asarray(
                array[day_index, minute_index, absolute_stock_index], dtype=np.float32
            )
        frame = validate_minute_data(frame)
        grouped = frame.groupby(["date", "code"], observed=True, sort=False)["close"]
        previous = grouped.shift(1)
        ratio = frame["close"].div(previous.where(previous.abs() > 1e-12))
        frame["logret"] = np.log(ratio.where(ratio > 0))
        frame["oc_ret"] = frame["close"].div(
            frame["open"].where(frame["open"].abs() > 1e-12)
        ) - 1.0
        frame.attrs["minute_features_ready"] = True
        return frame

    def _year_dates(self, year: int) -> pd.DatetimeIndex:
        metadata = self.manifest["years"][str(year)]
        values = np.load(self.config.root / metadata["dates_file"], allow_pickle=False)
        return pd.DatetimeIndex(pd.to_datetime(values)).normalize()

    def _array(self, year: int, channel: str):
        key = (year, channel)
        if key not in self._arrays:
            metadata = self.manifest["years"][str(year)]
            relative = metadata["valid_mask"] if channel == "valid_mask" else metadata["channels"][channel]
            self._arrays[key] = np.load(
                self.config.root / relative, mmap_mode="r", allow_pickle=False
            )
        return self._arrays[key]


class DolphinDBMinuteRAMStore(MinuteMemMapStore):
    """Read DDB once into shared process RAM, then expose the MemMap store API.

    Arrays remain split by year so reward chunk specifications and persistent
    factor caches stay compatible with the proven MemMap execution path.  No
    raw-minute file is created and training performs zero remote DDB queries.
    """

    in_memory = True

    def __init__(
        self,
        loader: DolphinDBMinuteLoader,
        config: MinuteMemMapConfig,
        loader_factory: Callable[[], DolphinDBMinuteLoader] | None = None,
        *,
        max_ram_gb: float = 0.0,
        reserve_ram_gb: float = 64.0,
    ) -> None:
        if config.reward_parallel_backend != "threading":
            raise ValueError(
                "DDB RAM mode requires reward_parallel_backend=threading so workers "
                "share arrays instead of copying them"
            )
        if config.build_workers > 1 and loader_factory is None:
            raise ValueError(
                "DDB RAM build_workers > 1 requires an independent loader per thread"
            )
        self.config = config
        self._loader = loader
        self._loader_factory = loader_factory
        self._ram_arrays: dict[tuple[int, str], np.ndarray] = {}
        self._year_date_values: dict[int, pd.DatetimeIndex] = {}
        self._max_ram_gb = float(max_ram_gb)
        self._reserve_ram_gb = float(reserve_ram_gb)
        self.config.root.mkdir(parents=True, exist_ok=True)

        audit = loader.audit()
        if not audit.passed:
            raise ValueError(
                "DDB RAM loading requires adjusted OHLC; verify the source and set "
                "prices_are_adjusted=true"
            )
        dates = loader.load_trade_dates(loader.config.start_date, loader.config.end_date)
        self.dates = pd.DatetimeIndex(dates).normalize()
        self.minute_grid = np.asarray(config.minute_grid, dtype=str)
        helper = DolphinDBMinuteMemMapBuilder(loader, config, loader_factory)
        helper._load_minute_grid(dates)
        daily = loader.build_daily_in_memory(
            loader.config.start_date,
            loader.config.end_date,
            time_filter_sql=helper._ddb_time_filter_sql(),
        )
        self.daily_data = daily
        self.stocks = np.asarray(sorted(daily["code"].astype(str).unique()), dtype=str)
        self.stock_lookup = {str(code): index for index, code in enumerate(self.stocks)}
        self.fingerprint = _source_fingerprint(loader, dates, config.minute_grid)
        helper.ensure_duplicate_audit(dates, self.fingerprint)
        self.manifest = {
            "format_version": MEMMAP_FORMAT_VERSION,
            "complete": True,
            "fingerprint": self.fingerprint,
            "layout": "RAM year/channel -> (n_days, n_minutes, n_stocks)",
            "dtype": "float32",
            "channels": list(MEMMAP_CHANNELS),
            "n_minutes": len(self.minute_grid),
            "n_stocks": len(self.stocks),
            "years": {},
        }
        self._validate_capacity()
        self._load_all_years(helper)

    @property
    def daily_file(self) -> Path:
        raise RuntimeError("DDB RAM mode keeps daily data in memory")

    def _estimated_bytes(self) -> int:
        elements = len(self.dates) * len(self.minute_grid) * len(self.stocks)
        return int(elements * (len(MEMMAP_CHANNELS) * 4 + 1))

    @staticmethod
    def _available_memory_bytes() -> int | None:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except ImportError:
            try:
                return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (AttributeError, OSError, ValueError):
                return None

    def _validate_capacity(self) -> None:
        required = self._estimated_bytes()
        available = self._available_memory_bytes()
        limit = int(self._max_ram_gb * 1024**3) if self._max_ram_gb > 0 else None
        print(
            f"[DDBRAM] capacity_estimate raw_arrays_gb={required / 1024**3:.1f} "
            f"available_gb={available / 1024**3:.1f}" if available is not None else
            f"[DDBRAM] capacity_estimate raw_arrays_gb={required / 1024**3:.1f} available_gb=unknown",
            flush=True,
        )
        if limit is not None and required > limit:
            raise MemoryError(
                f"DDB RAM arrays require {required / 1024**3:.1f}GB, exceeding "
                f"memory.max_ram_gb={self._max_ram_gb:.1f}GB"
            )
        if available is not None:
            usable = available - int(self._reserve_ram_gb * 1024**3)
            if required > usable:
                raise MemoryError(
                    f"DDB RAM arrays require {required / 1024**3:.1f}GB but only "
                    f"{available / 1024**3:.1f}GB is available with "
                    f"reserve_ram_gb={self._reserve_ram_gb:.1f}"
                )

    def _load_all_years(self, helper: DolphinDBMinuteMemMapBuilder) -> None:
        started = time.perf_counter()
        stock_lookup = self.stock_lookup
        minute_lookup = {
            str(value): index for index, value in enumerate(self.minute_grid.tolist())
        }
        worker_local = threading.local()
        worker_loaders: list[DolphinDBMinuteLoader] = []
        worker_lock = threading.Lock()

        def initialize_worker() -> None:
            worker_local.loader = (
                self._loader_factory() if self._loader_factory is not None else self._loader
            )
            if worker_local.loader is not self._loader:
                with worker_lock:
                    worker_loaders.append(worker_local.loader)

        try:
            for year in sorted({int(date.year) for date in self.dates}):
                year_dates = self.dates[self.dates.year == year]
                self._year_date_values[year] = year_dates
                shape = (len(year_dates), len(self.minute_grid), len(self.stocks))
                arrays = {
                    channel: np.empty(shape, dtype=np.float32)
                    for channel in MEMMAP_CHANNELS
                }
                mask = np.empty(shape, dtype=np.uint8)
                for channel, array in arrays.items():
                    self._ram_arrays[(year, channel)] = array
                self._ram_arrays[(year, "valid_mask")] = mask
                self.manifest["years"][str(year)] = {
                    "n_days": len(year_dates), "shape": list(shape)
                }
                print(
                    f"[DDBRAM] year_start year={year} days={len(year_dates)} "
                    f"shape={shape} workers={self.config.build_workers}",
                    flush=True,
                )
                year_started = time.perf_counter()
                stage_totals = {key: 0.0 for key in LOAD_TIMING_KEYS}
                task_total_s = 0.0
                raw_rows_total = 0
                minute_rows_total = 0

                def load_day(day_index: int) -> dict[str, Any]:
                    return helper._write_day(
                        worker_local.loader,
                        day_index,
                        pd.Timestamp(year_dates[day_index]),
                        arrays,
                        mask,
                        stock_lookup,
                        minute_lookup,
                    )

                completed = 0
                with ThreadPoolExecutor(
                    max_workers=self.config.build_workers,
                    thread_name_prefix="ddb-ram",
                    initializer=initialize_worker,
                ) as executor:
                    futures = [executor.submit(load_day, index) for index in range(len(year_dates))]
                    for future in as_completed(futures):
                        result = future.result()
                        completed += 1
                        for key in LOAD_TIMING_KEYS:
                            stage_totals[key] += float(result[key])
                        task_total_s += float(result["total_s"])
                        raw_rows_total += int(result["raw_rows"])
                        minute_rows_total += int(result["minute_rows"])
                        if completed == 1 or completed == len(futures) or completed % 10 == 0:
                            year_elapsed = max(time.perf_counter() - year_started, 1e-9)
                            rate = completed / year_elapsed
                            eta_minutes = (
                                (len(futures) - completed) / max(rate, 1e-9) / 60.0
                            )
                            stage_sum = max(sum(stage_totals.values()), 1e-9)
                            stage_avg = ",".join(
                                f"{key.removesuffix('_s')}:{stage_totals[key] / completed:.2f}"
                                for key in LOAD_TIMING_KEYS
                            )
                            stage_share = ",".join(
                                f"{key.removesuffix('_s')}:{stage_totals[key] / stage_sum * 100:.0f}%"
                                for key in LOAD_TIMING_KEYS
                            )
                            print(
                                f"[DDBRAM] load_progress year={year} "
                                f"days={completed}/{len(futures)} date={result['date']} "
                                f"raw_rows={int(result['raw_rows']):,} "
                                f"minute_rows={int(result['minute_rows']):,} "
                                f"wall_minutes={year_elapsed / 60:.1f} "
                                f"rate_days_per_min={rate * 60:.2f} "
                                f"eta_minutes={eta_minutes:.1f} "
                                f"current_total_s={float(result['total_s']):.2f} "
                                f"stage_avg_s={stage_avg} stage_share={stage_share}",
                                flush=True,
                            )
                year_elapsed = max(time.perf_counter() - year_started, 1e-9)
                print(
                    f"[DDBRAM] year_complete year={year} "
                    f"days={completed} raw_rows={raw_rows_total:,} "
                    f"minute_rows={minute_rows_total:,} wall_seconds={year_elapsed:.1f} "
                    f"worker_task_seconds={task_total_s:.1f} "
                    f"effective_parallelism={task_total_s / year_elapsed:.2f} "
                    f"resident_gb={sum(a.nbytes for a in self._ram_arrays.values()) / 1024**3:.1f}",
                    flush=True,
                )
        finally:
            for worker_loader in worker_loaders:
                worker_loader.session.close()
        print(
            f"[DDBRAM] load_complete dates={len(self.dates):,} stocks={len(self.stocks):,} "
            f"channels={len(MEMMAP_CHANNELS)} seconds={time.perf_counter() - started:.1f} "
            "remote_ddb_queries_during_training=0",
            flush=True,
        )

    def _year_dates(self, year: int) -> pd.DatetimeIndex:
        return self._year_date_values[int(year)]

    def _array(self, year: int, channel: str) -> np.ndarray:
        return self._ram_arrays[(int(year), channel)]
