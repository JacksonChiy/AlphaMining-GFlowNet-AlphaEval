from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import PriceDataPreprocessor


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

