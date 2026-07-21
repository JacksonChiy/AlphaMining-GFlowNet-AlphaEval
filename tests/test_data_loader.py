from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import PriceDataPreprocessor, prepare_price_csv


def test_alias_mapping_missing_and_adjustment() -> None:
    frame = pd.DataFrame({
        "交易日期": ["2024-01-02", "2024-01-03", "2024-01-02"],
        "证券代码": ["000001", "000001", "000002"],
        "开盘价": [10.0, np.nan, 20.0],
        "最高价": [11.0, 11.5, 21.0],
        "最低价": [9.0, 9.5, 19.0],
        "收盘价": [10.5, 11.0, 20.5],
        "成交量": [1000, 1200, 2000],
        "成交额": [10_200, 13_000, 40_500],
        "复权因子": [2.0, 2.0, 1.0],
    })
    data, report = PriceDataPreprocessor().transform(frame)
    assert set(("date", "code", "vwap", "z_close")).issubset(data.columns)
    assert report.adjustment_applied
    assert data.loc[data["code"] == "000001", "open"].iloc[1] == 20.0
    assert pd.api.types.is_datetime64_any_dtype(data["date"])


def test_missing_required_column_is_explicit() -> None:
    frame = pd.DataFrame({"date": ["2024-01-01"], "code": ["1"]})
    try:
        PriceDataPreprocessor().transform(frame)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected a validation error")


def test_total_turnover_mapping_and_fast_universe_filter(tmp_path) -> None:
    rows = []
    for date in pd.date_range("2023-12-01", periods=4):
        for code, turnover in (("A", 100.0), ("B", 300.0), ("C", 200.0)):
            rows.append({
                "order_book_id": code,
                "date": date,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 10.0,
                "total_turnover": turnover,
            })
    source = tmp_path / "price.csv"
    pd.DataFrame(rows).to_csv(source, index=False)

    result = prepare_price_csv(
        source,
        tmp_path / "daily.pkl",
        tmp_path / "report.json",
        start_date="2023-12-03",
        max_stocks=2,
        universe_start_date="2023-12-01",
        universe_end_date="2023-12-02",
        chunksize=5,
    )

    assert set(result["code"]) == {"B", "C"}
    assert set(result["date"]) == set(pd.date_range("2023-12-03", periods=2))
    assert "amount" in result.columns
    assert np.allclose(result["vwap"], result["amount"] / result["volume"])
