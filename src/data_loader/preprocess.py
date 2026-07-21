from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


CANONICAL_ALIASES: dict[str, set[str]] = {
    "date": {"date", "datetime", "trade_date", "trading_date", "日期", "交易日期"},
    "code": {"code", "symbol", "ticker", "security_id", "order_book_id", "股票代码", "证券代码"},
    "open": {"open", "open_price", "开盘", "开盘价"},
    "high": {"high", "high_price", "最高", "最高价"},
    "low": {"low", "low_price", "最低", "最低价"},
    "close": {"close", "close_price", "收盘", "收盘价"},
    "volume": {"volume", "vol", "trade_volume", "成交量"},
    "amount": {"amount", "turnover", "total_turnover", "trade_amount", "成交额"},
    "vwap": {"vwap", "avg_price", "average_price", "均价", "成交均价"},
    "adj_factor": {"adj_factor", "adjust_factor", "factor", "复权因子"},
    "industry": {"industry", "industry_code", "申万行业", "行业"},
    "market_cap": {"market_cap", "total_market_cap", "float_market_cap", "市值", "总市值", "流通市值"},
}

REQUIRED = ("date", "code", "open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close", "vwap")
NUMERIC_COLUMNS = PRICE_COLUMNS + ("volume", "amount", "adj_factor", "market_cap")


def _normalise_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", str(value).strip().lower()).strip("_")


@dataclass
class DataQualityReport:
    input_rows: int
    source_rows_scanned: int = 0
    output_rows: int = 0
    column_mapping: dict[str, str] = field(default_factory=dict)
    duplicate_rows_removed: int = 0
    invalid_rows_removed: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    outliers_clipped: dict[str, int] = field(default_factory=dict)
    adjustment_applied: bool = False
    risk_fields_available: dict[str, bool] = field(default_factory=dict)
    date_filter: dict[str, str | None] = field(default_factory=dict)
    universe_selection: dict[str, str | int | None] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


class PriceDataPreprocessor:
    """Prepare point-in-time daily OHLCV without backward filling future data."""

    def __init__(self, winsor_limits: tuple[float, float] = (0.001, 0.999)) -> None:
        self.winsor_limits = winsor_limits

    def infer_mapping(self, columns: Iterable[str]) -> dict[str, str]:
        normalized = {_normalise_name(column): column for column in columns}
        mapping: dict[str, str] = {}
        for canonical, aliases in CANONICAL_ALIASES.items():
            matches = [normalized[a] for a in map(_normalise_name, aliases) if a in normalized]
            if len(matches) > 1:
                warnings.warn(f"Multiple columns map to {canonical}: {matches}; using {matches[0]}")
            if matches:
                mapping[matches[0]] = canonical
        return mapping

    def transform(
        self,
        frame: pd.DataFrame,
        explicit_mapping: Mapping[str, str] | None = None,
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        report = DataQualityReport(input_rows=len(frame))
        mapping = self.infer_mapping(frame.columns)
        if explicit_mapping:
            for source, target in explicit_mapping.items():
                if source not in frame.columns:
                    raise ValueError(f"Explicit source column not found: {source}")
                if target not in CANONICAL_ALIASES:
                    raise ValueError(f"Unsupported canonical column: {target}")
                mapping[source] = target
        report.column_mapping = dict(mapping)
        data = frame.rename(columns=mapping).copy()

        missing_required = [name for name in REQUIRED if name not in data.columns]
        if missing_required:
            raise ValueError(
                f"Missing required columns after automatic mapping: {missing_required}. "
                f"Available columns: {list(data.columns)}"
            )

        keep = [column for column in CANONICAL_ALIASES if column in data.columns]
        data = data[keep]
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
        data["code"] = data["code"].astype("string").str.strip()
        for column in NUMERIC_COLUMNS:
            if column in data:
                data[column] = pd.to_numeric(data[column], errors="coerce")

        invalid_key = data["date"].isna() | data["code"].isna() | data["code"].eq("")
        report.invalid_rows_removed += int(invalid_key.sum())
        data = data.loc[~invalid_key].sort_values(["code", "date"], kind="stable")
        duplicate_mask = data.duplicated(["date", "code"], keep="last")
        report.duplicate_rows_removed = int(duplicate_mask.sum())
        data = data.loc[~duplicate_mask].copy()

        if "vwap" not in data and "amount" in data:
            data["vwap"] = data["amount"] / data["volume"].replace(0, np.nan)
        if "vwap" not in data:
            data["vwap"] = (data["high"] + data["low"] + data["close"]) / 3.0

        # Only forward-fill prices within each security. Back-filling would leak future observations.
        price_present = [column for column in PRICE_COLUMNS if column in data]
        data[price_present] = data.groupby("code", observed=True)[price_present].ffill()
        data["volume"] = data["volume"].fillna(0.0).clip(lower=0.0)
        if "amount" in data:
            data["amount"] = data["amount"].fillna(0.0).clip(lower=0.0)

        if "adj_factor" in data:
            valid_factor = data["adj_factor"].where(data["adj_factor"] > 0)
            valid_factor = valid_factor.groupby(data["code"], observed=True).ffill().fillna(1.0)
            data[price_present] = data[price_present].mul(valid_factor, axis=0)
            data["volume"] = data["volume"].div(valid_factor.replace(0, np.nan)).fillna(0.0)
            report.adjustment_applied = True

        bad_ohlc = (
            (data["low"] > data["high"])
            | (data["open"] <= 0)
            | (data["high"] <= 0)
            | (data["low"] <= 0)
            | (data["close"] <= 0)
        )
        report.invalid_rows_removed += int(bad_ohlc.sum())
        data = data.loc[~bad_ohlc].copy()

        # Cross-sectional winsorization uses only same-date information.
        clip_columns = [*price_present, "volume"] + (["amount"] if "amount" in data else [])
        for column in clip_columns:
            before = data[column].copy()
            grouped = data.groupby("date", observed=True)[column]
            lower = grouped.transform(lambda x: x.quantile(self.winsor_limits[0]))
            upper = grouped.transform(lambda x: x.quantile(self.winsor_limits[1]))
            data[column] = data[column].clip(lower=lower, upper=upper)
            report.outliers_clipped[column] = int((before != data[column]).fillna(False).sum())

        for column in ("open", "high", "low", "close", "volume", "vwap"):
            mean = data.groupby("date", observed=True)[column].transform("mean")
            std = data.groupby("date", observed=True)[column].transform("std").replace(0, np.nan)
            data[f"z_{column}"] = ((data[column] - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        data = data.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        report.output_rows = len(data)
        report.missing_by_column = {column: int(data[column].isna().sum()) for column in data.columns}
        report.risk_fields_available = {
            "industry": "industry" in data,
            "market_cap": "market_cap" in data,
        }
        return data, report


def prepare_price_csv(
    input_path: str | Path = "data/price.csv",
    output_path: str | Path = "data/daily_price.pkl",
    report_path: str | Path = "results/data_quality_report.json",
    explicit_mapping: Mapping[str, str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_stocks: int | None = None,
    universe_start_date: str | None = None,
    universe_end_date: str | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    input_path, output_path, report_path = map(Path, (input_path, output_path, report_path))
    if not input_path.exists():
        raise FileNotFoundError(f"Local input file not found: {input_path.resolve()}")
    preprocessor = PriceDataPreprocessor()
    filtered = any(
        value is not None for value in (start_date, end_date, max_stocks)
    )
    if filtered:
        frame, source_rows, universe_details = _read_filtered_csv(
            input_path,
            preprocessor,
            start_date=start_date,
            end_date=end_date,
            max_stocks=max_stocks,
            universe_start_date=universe_start_date,
            universe_end_date=universe_end_date,
            chunksize=chunksize,
        )
    else:
        frame = pd.read_csv(input_path, low_memory=False)
        source_rows = len(frame)
        universe_details = {}
    prepared, report = preprocessor.transform(frame, explicit_mapping)
    report.source_rows_scanned = source_rows
    report.date_filter = {"start_date": start_date, "end_date": end_date}
    report.universe_selection = universe_details
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_pickle(output_path)
    report.to_json(report_path)
    return prepared


def _read_filtered_csv(
    input_path: Path,
    preprocessor: PriceDataPreprocessor,
    *,
    start_date: str | None,
    end_date: str | None,
    max_stocks: int | None,
    universe_start_date: str | None,
    universe_end_date: str | None,
    chunksize: int,
) -> tuple[pd.DataFrame, int, dict[str, str | int | None]]:
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    if max_stocks is not None and max_stocks < 1:
        raise ValueError("max_stocks must be positive")

    header = pd.read_csv(input_path, nrows=0)
    mapping = preprocessor.infer_mapping(header.columns)
    source_by_canonical = {target: source for source, target in mapping.items()}
    if "date" not in source_by_canonical or "code" not in source_by_canonical:
        raise ValueError("Date/code columns are required for filtered CSV loading")
    date_column = source_by_canonical["date"]
    code_column = source_by_canonical["code"]
    start = pd.Timestamp(start_date) if start_date else None
    end = pd.Timestamp(end_date) if end_date else None
    selected_codes: set[str] | None = None
    universe_details: dict[str, str | int | None] = {}

    if max_stocks is not None:
        score_column = source_by_canonical.get("amount", source_by_canonical.get("volume"))
        if score_column is None:
            raise ValueError("Amount or volume is required for liquid-universe selection")
        universe_start = pd.Timestamp(universe_start_date) if universe_start_date else start
        universe_end = pd.Timestamp(universe_end_date) if universe_end_date else end
        scores = pd.Series(dtype="float64")
        print(
            "[Data] universe_scan_start "
            f"metric={score_column} start={universe_start} end={universe_end} ",
            flush=True,
        )
        for chunk_index, chunk in enumerate(
            pd.read_csv(
                input_path,
                usecols=[date_column, code_column, score_column],
                chunksize=chunksize,
                low_memory=False,
            ),
            start=1,
        ):
            dates = pd.to_datetime(chunk[date_column], errors="coerce")
            mask = dates.notna()
            if universe_start is not None:
                mask &= dates >= universe_start
            if universe_end is not None:
                mask &= dates <= universe_end
            selected = chunk.loc[mask, [code_column, score_column]].copy()
            selected[score_column] = pd.to_numeric(
                selected[score_column], errors="coerce"
            ).clip(lower=0.0)
            chunk_scores = selected.groupby(code_column, observed=True)[score_column].sum()
            scores = scores.add(chunk_scores, fill_value=0.0)
            print(
                f"[Data] universe_scan_chunk={chunk_index:03d} "
                f"candidate_stocks={len(scores)}",
                flush=True,
            )
        selected_codes = set(scores.nlargest(max_stocks).index.astype(str))
        if not selected_codes:
            raise ValueError("No stocks remain after liquid-universe selection")
        universe_details = {
            "max_stocks": max_stocks,
            "selected_stocks": len(selected_codes),
            "metric": score_column,
            "universe_start_date": str(universe_start.date()) if universe_start else None,
            "universe_end_date": str(universe_end.date()) if universe_end else None,
        }
        print(
            f"[Data] universe_scan_complete selected_stocks={len(selected_codes)}",
            flush=True,
        )

    filtered_chunks: list[pd.DataFrame] = []
    source_rows = 0
    kept_rows = 0
    print(f"[Data] filtered_read_start chunksize={chunksize}", flush=True)
    for chunk_index, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize, low_memory=False), start=1
    ):
        source_rows += len(chunk)
        dates = pd.to_datetime(chunk[date_column], errors="coerce")
        mask = dates.notna()
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates <= end
        if selected_codes is not None:
            mask &= chunk[code_column].astype(str).isin(selected_codes)
        selected = chunk.loc[mask].copy()
        if not selected.empty:
            filtered_chunks.append(selected)
            kept_rows += len(selected)
        print(
            f"[Data] filtered_read_chunk={chunk_index:03d} "
            f"source_rows={source_rows:,} kept_rows={kept_rows:,}",
            flush=True,
        )
    if not filtered_chunks:
        raise ValueError("No rows remain after applying the configured data filters")
    print(
        f"[Data] filtered_read_complete source_rows={source_rows:,} "
        f"kept_rows={kept_rows:,}",
        flush=True,
    )
    return pd.concat(filtered_chunks, ignore_index=True), source_rows, universe_details
