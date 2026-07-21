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
    "amount": {"amount", "turnover", "trade_amount", "成交额"},
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
    output_rows: int = 0
    column_mapping: dict[str, str] = field(default_factory=dict)
    duplicate_rows_removed: int = 0
    invalid_rows_removed: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    outliers_clipped: dict[str, int] = field(default_factory=dict)
    adjustment_applied: bool = False
    risk_fields_available: dict[str, bool] = field(default_factory=dict)

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
) -> pd.DataFrame:
    input_path, output_path, report_path = map(Path, (input_path, output_path, report_path))
    if not input_path.exists():
        raise FileNotFoundError(f"Local input file not found: {input_path.resolve()}")
    frame = pd.read_csv(input_path, low_memory=False)
    prepared, report = PriceDataPreprocessor().transform(frame, explicit_mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_pickle(output_path)
    report.to_json(report_path)
    return prepared

