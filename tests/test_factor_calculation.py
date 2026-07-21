from __future__ import annotations

import numpy as np
import pandas as pd

from src.operators import apply_binary, apply_cross_sectional, apply_time_series


def test_safe_division_and_cross_section() -> None:
    left = pd.Series([1.0, 2.0, 3.0])
    right = pd.Series([1.0, 0.0, 3.0])
    divided = apply_binary("div", left, right)
    assert divided.iloc[0] == 1.0 and np.isnan(divided.iloc[1])
    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 3))
    ranked = apply_cross_sectional("cs_rank", left, dates)
    assert ranked.tolist() == [1 / 3, 2 / 3, 1.0]


def test_rolling_window_is_grouped_by_security() -> None:
    values = pd.Series([1.0, 2.0, 10.0, 20.0])
    codes = pd.Series(["A", "A", "B", "B"])
    result = apply_time_series("ts_mean", values, codes, 2)
    assert np.isnan(result.iloc[0]) and result.iloc[1] == 1.5
    assert np.isnan(result.iloc[2]) and result.iloc[3] == 15.0

