from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.operators.minute import build_minute_features, validate_minute_data

from .dolphindb_minute import DolphinDBMinuteLoader, normalize_dolphindb_minutes


MEMMAP_CHANNELS = (
    "open", "high", "low", "close", "vol", "amount",
    "ret", "vwap", "hl_pct", "bar_pos", "amihud", "rv",
    "signed_vol", "signed_amt", "typical", "vwap_cum", "twap", "obv", "pvt",
)
MEMMAP_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MinuteMemMapConfig:
    root: Path
    block_cache_dir: Path
    expected_minutes: int = 241
    stock_tile_size: int = 256
    workers: int = 1
    flush_every_days: int = 1
    force_rebuild: bool = False

    def __post_init__(self) -> None:
        if min(
            self.expected_minutes, self.stock_tile_size, self.workers, self.flush_every_days
        ) < 1:
            raise ValueError("MemMap size settings must be positive")

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
        return cls(
            root=Path(str(root)),
            block_cache_dir=cache_path,
            expected_minutes=int(values.get("expected_minutes", 241)),
            stock_tile_size=int(values.get("stock_tile_size", 256)),
            workers=int(values.get("workers", 1)),
            flush_every_days=int(values.get("flush_every_days", 1)),
            force_rebuild=bool(values.get("force_rebuild", False)),
        )


def _time_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.time()
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S.%f")
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Cannot normalize minute time value: {value!r}")
    return parsed.strftime("%H:%M:%S.%f")


def _source_fingerprint(loader: DolphinDBMinuteLoader, dates: Sequence[pd.Timestamp]) -> str:
    payload = {
        "format_version": MEMMAP_FORMAT_VERSION,
        "database": loader.config.database,
        "table": loader.config.table,
        "start_date": str(pd.Timestamp(dates[0]).date()),
        "end_date": str(pd.Timestamp(dates[-1]).date()),
        "prices_are_adjusted": loader.config.prices_are_adjusted,
        "channels": MEMMAP_CHANNELS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class DolphinDBMinuteMemMapBuilder:
    """One-time DDB extraction into local yearly/channel float32 MemMaps."""

    def __init__(
        self,
        loader: DolphinDBMinuteLoader,
        config: MinuteMemMapConfig,
    ) -> None:
        self.loader = loader
        self.config = config

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
        fingerprint = _source_fingerprint(self.loader, dates)
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
            daily = self.loader.build_daily_in_memory(
                self.loader.config.start_date, self.loader.config.end_date
            )
            daily.to_csv(daily_path, index=False, compression="gzip")
            stocks = np.array(sorted(daily["code"].astype(str).unique()), dtype=str)
            np.save(root / "stocks.npy", stocks, allow_pickle=False)
            minute_grid = self._load_minute_grid(dates)
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

    def _load_minute_grid(self, dates: Sequence[pd.Timestamp]) -> tuple[str, ...]:
        sample_dates = []
        for year in sorted({date.year for date in dates}):
            sample_dates.append(next(date for date in dates if date.year == year))
        values: set[str] = set()
        for date in sample_dates:
            script = (
                "select distinct time as minuteTime from "
                f"{self.loader.table_expression} where date = {date:%Y.%m.%d} "
                "order by minuteTime"
            )
            result = pd.DataFrame(self.loader.session.run(script))
            if "minuteTime" not in result:
                raise ValueError("DolphinDB minute-grid query returned no minuteTime column")
            values.update(_time_key(value) for value in result["minuteTime"].dropna())
        grid = tuple(sorted(values))
        if len(grid) != self.config.expected_minutes:
            raise ValueError(
                f"Expected {self.config.expected_minutes} minute points, got {len(grid)}. "
                "Inspect the source time convention and set memmap.expected_minutes only "
                "after confirming the actual grid."
            )
        print(
            f"[MemMapBuild] minute_grid_ready count={len(grid)} "
            f"first={grid[0]} last={grid[-1]}",
            flush=True,
        )
        return grid

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
        completed_days = int(progress.get("completed_days", 0)) if resume else 0
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
            completed_days = 0
        print(
            f"[MemMapBuild] year_start year={year} days={len(dates)} "
            f"resume_from={completed_days} shape={shape}",
            flush=True,
        )
        for day_index in range(completed_days, len(dates)):
            date = pd.Timestamp(dates[day_index])
            raw = self.loader.session.run(self.loader.build_data_sql(date, date))
            frame = build_minute_features(
                normalize_dolphindb_minutes(pd.DataFrame(raw))
            )
            codes = frame["code"].astype(str).map(stock_lookup)
            minutes = frame["datetime"].map(lambda value: minute_lookup.get(_time_key(value)))
            valid_index = codes.notna() & minutes.notna()
            if not valid_index.all():
                raise ValueError(
                    f"MemMap index mismatch on {date.date()}: "
                    f"unmapped_rows={int((~valid_index).sum())}"
                )
            stock_index = codes.to_numpy(dtype=np.int64)
            minute_index = minutes.to_numpy(dtype=np.int64)
            for channel, array in arrays.items():
                array[day_index, :, :] = np.nan
                values = pd.to_numeric(frame[channel], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                array[day_index, minute_index, stock_index] = values
            mask[day_index, :, :] = 0
            # The mask records source-row presence, not close validity. Individual channel
            # NaNs must remain visible so reductions keep exactly the same row-position semantics.
            mask[day_index, minute_index, stock_index] = 1
            if (
                (day_index + 1) % self.config.flush_every_days == 0
                or day_index + 1 == len(dates)
            ):
                for array in arrays.values():
                    array.flush()
                mask.flush()
                self._write_json(progress_path, {
                    "fingerprint": fingerprint,
                    "shape": list(shape),
                    "completed_days": day_index + 1,
                    "complete": day_index + 1 == len(dates),
                })
            print(
                f"[MemMapBuild] day_complete year={year} "
                f"day={day_index + 1:03d}/{len(dates):03d} date={date.date()} "
                f"minute_rows={len(frame):,}",
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
