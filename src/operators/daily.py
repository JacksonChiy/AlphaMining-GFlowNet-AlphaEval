from __future__ import annotations

import numpy as np
import pandas as pd

from .torch_timeseries import (
    apply_time_series_torch,
    get_time_series_backend_config,
    record_pandas_time_series_call,
    resolve_time_series_device,
)

EPS = 1e-12


def _finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def apply_unary(name: str, value: pd.Series) -> pd.Series:
    functions = {
        "log": lambda x: np.log(x.abs() + EPS),
        "abs": lambda x: x.abs(),
        "neg": lambda x: -x,
        "sqrt": lambda x: np.sqrt(x.abs()),
        "tanh": np.tanh,
    }
    if name not in functions:
        raise KeyError(f"Unknown unary operator: {name}")
    return _finite(functions[name](value.astype(float)))


def apply_binary(name: str, left: pd.Series, right: pd.Series) -> pd.Series:
    if name == "add":
        value = left + right
    elif name == "sub":
        value = left - right
    elif name == "mul":
        value = left * right
    elif name == "div":
        value = left / right.where(right.abs() > EPS)
    else:
        raise KeyError(f"Unknown binary operator: {name}")
    return _finite(value)


def _apply_time_series_pandas(
    name: str, value: pd.Series, code: pd.Series, window: int
) -> pd.Series:
    grouped = value.groupby(code, observed=True, sort=False)
    min_periods = window
    if name == "ts_delay":
        return grouped.shift(window)
    if name == "ts_delta":
        return value - grouped.shift(window)
    if name == "ts_mean":
        result = grouped.rolling(window, min_periods=min_periods).mean()
    elif name == "ts_std":
        result = grouped.rolling(window, min_periods=min_periods).std(ddof=1)
    elif name == "ts_sum":
        result = grouped.rolling(window, min_periods=min_periods).sum()
    elif name == "ts_max":
        result = grouped.rolling(window, min_periods=min_periods).max()
    elif name == "ts_min":
        result = grouped.rolling(window, min_periods=min_periods).min()
    elif name == "ts_rank":
        result = grouped.rolling(window, min_periods=min_periods).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    elif name == "ts_zscore":
        mean = grouped.rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True)
        std = grouped.rolling(window, min_periods=min_periods).std(ddof=1).reset_index(level=0, drop=True)
        return _finite((value - mean) / std.where(std.abs() > EPS))
    else:
        raise KeyError(f"Unknown time-series operator: {name}")
    return _finite(result.reset_index(level=0, drop=True))


def apply_time_series(name: str, value: pd.Series, code: pd.Series, window: int) -> pd.Series:
    config = get_time_series_backend_config()
    if resolve_time_series_device(config) is not None:
        return apply_time_series_torch(name, value, code, window, config)
    record_pandas_time_series_call()
    return _apply_time_series_pandas(name, value, code, window)


def apply_cross_sectional(name: str, value: pd.Series, date: pd.Series) -> pd.Series:
    grouped = value.groupby(date, observed=True, sort=False)
    if name == "cs_rank":
        result = grouped.rank(pct=True)
    elif name == "cs_demean":
        result = value - grouped.transform("mean")
    elif name == "cs_zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        result = (value - mean) / std
    else:
        raise KeyError(f"Unknown cross-sectional operator: {name}")
    return _finite(result)
