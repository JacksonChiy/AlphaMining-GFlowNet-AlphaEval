from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


EPS = 1e-12
MINUTE_KEYS = ("date", "code")


def _safe_div(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = right.where(right.abs() > EPS)
    return left.div(denominator).replace([np.inf, -np.inf], np.nan)


def _group_transform(values: pd.Series, data: pd.DataFrame, func: Callable) -> pd.Series:
    return values.groupby([data["date"], data["code"]], observed=True, sort=False).transform(func)


def validate_minute_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and stably sort the canonical long-form minute table."""
    required = {"date", "datetime", "code", "open", "high", "low", "close", "vol", "amount"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Minute data is missing canonical columns: {missing}")
    ordered = data.copy()
    ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"])
    ordered = ordered.sort_values(["date", "code", "datetime"], kind="stable").reset_index(drop=True)
    duplicated = ordered.duplicated(["date", "code", "datetime"])
    if duplicated.any():
        raise ValueError(f"Minute data contains {int(duplicated.sum())} duplicated date-code-datetime rows")
    return ordered


def build_minute_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build chart-27 leaves; an existing precomputed column always takes precedence."""
    if data.attrs.get("minute_features_ready") and set(MINUTE_KEYS).issubset(data.columns):
        return data
    frame = validate_minute_data(data)
    numeric = ("open", "high", "low", "close", "vol", "amount")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped_close = frame.groupby(list(MINUTE_KEYS), observed=True, sort=False)["close"]
    previous_close = grouped_close.shift(1)
    fallback: dict[str, pd.Series] = {}
    fallback["ret"] = _safe_div(frame["close"], previous_close) - 1.0
    fallback["vwap"] = _safe_div(frame["amount"], frame["vol"])
    fallback["hl_pct"] = _safe_div(frame["high"] - frame["low"], frame["close"].abs())
    fallback["bar_pos"] = _safe_div(frame["close"] - frame["low"], frame["high"] - frame["low"])
    fallback["amihud"] = _safe_div(fallback["ret"].abs(), frame["amount"].abs())
    fallback["rv"] = fallback["ret"].pow(2)
    direction = np.sign(fallback["ret"]).fillna(0.0)
    fallback["signed_vol"] = direction * frame["vol"]
    fallback["signed_amt"] = direction * frame["amount"]
    fallback["typical"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0

    grouped_amount = frame["amount"].groupby(
        [frame["date"], frame["code"]], observed=True, sort=False
    )
    grouped_vol = frame["vol"].groupby(
        [frame["date"], frame["code"]], observed=True, sort=False
    )
    fallback["vwap_cum"] = _safe_div(grouped_amount.cumsum(), grouped_vol.cumsum())
    fallback["twap"] = _group_transform(frame["close"], frame, lambda value: value.expanding().mean())
    fallback["obv"] = _group_transform(direction * frame["vol"], frame, lambda value: value.cumsum())
    fallback["pvt"] = _group_transform(fallback["ret"] * frame["vol"], frame, lambda value: value.cumsum())
    fallback["logret"] = np.log(_safe_div(frame["close"], previous_close).where(lambda x: x > 0))
    fallback["oc_ret"] = _safe_div(frame["close"], frame["open"]) - 1.0

    for name, values in fallback.items():
        if name not in frame:
            frame[name] = values
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame.attrs["minute_features_ready"] = True
    return frame


def apply_minute_unary(name: str, values: pd.Series, data: pd.DataFrame) -> pd.Series:
    grouped = values.groupby([data["date"], data["code"]], observed=True, sort=False)
    if name == "m_ret":
        return _safe_div(values, grouped.shift(1)) - 1.0
    if name == "m_logret":
        ratio = _safe_div(values, grouped.shift(1))
        return np.log(ratio.where(ratio > 0))
    if name == "m_rank":
        return grouped.rank(method="average", pct=True)
    if name == "m_zscore":
        mean = grouped.transform("mean")
        std = grouped.transform("std").where(lambda x: x > EPS)
        return (values - mean).div(std)
    if name == "m_abs":
        return values.abs()
    if name == "m_sign":
        return np.sign(values)
    if name == "m_log":
        return np.log(values.abs() + EPS)
    raise ValueError(f"Unknown minute unary operator: {name}")


def apply_minute_window(name: str, values: pd.Series, data: pd.DataFrame, window: int) -> pd.Series:
    grouped = values.groupby([data["date"], data["code"]], observed=True, sort=False)
    if name == "m_delay":
        return grouped.shift(window)
    if name == "m_delta":
        return values - grouped.shift(window)
    minimum = max(2, window // 2)
    if name == "m_ma":
        result = grouped.rolling(window, min_periods=minimum).mean()
        return result.reset_index(level=[0, 1], drop=True).reindex(values.index)
    if name == "m_std":
        result = grouped.rolling(window, min_periods=minimum).std()
        return result.reset_index(level=[0, 1], drop=True).reindex(values.index)
    raise ValueError(f"Unknown minute window operator: {name}")


def apply_minute_binary(name: str, left: pd.Series, right: pd.Series) -> pd.Series:
    if name == "m_add":
        return left + right
    if name == "m_sub":
        return left - right
    if name == "m_mul":
        return left * right
    if name == "m_div":
        return _safe_div(left, right)
    raise ValueError(f"Unknown minute binary operator: {name}")


def _exact_rank_mask(values: pd.Series, data: pd.DataFrame, window: int, largest: bool) -> pd.Series:
    rank = values.groupby([data["date"], data["code"]], observed=True, sort=False).rank(
        method="first", ascending=not largest
    )
    return rank <= window


def apply_mask_window(name: str, values: pd.Series, data: pd.DataFrame, window: int) -> pd.Series:
    ordinal = data.groupby(list(MINUTE_KEYS), observed=True, sort=False).cumcount()
    group_size = data.groupby(list(MINUTE_KEYS), observed=True, sort=False)["code"].transform("size")
    if name == "m_head":
        selected = ordinal < window
    elif name == "m_tail":
        selected = ordinal >= (group_size - window).clip(lower=0)
    elif name == "m_mid":
        start = ((group_size - window).clip(lower=0) // 2).astype(int)
        selected = (ordinal >= start) & (ordinal < start + window)
    elif name == "m_top":
        selected = _exact_rank_mask(values, data, window, largest=True)
    elif name == "m_bot":
        selected = _exact_rank_mask(values, data, window, largest=False)
    elif name == "m_xtreme":
        median = _group_transform(values, data, "median")
        selected = _exact_rank_mask((values - median).abs(), data, window, largest=True)
    else:
        raise ValueError(f"Unknown window mask operator: {name}")
    return values.where(selected)


def apply_mask_unary(name: str, values: pd.Series, data: pd.DataFrame) -> pd.Series:
    grouped = values.groupby([data["date"], data["code"]], observed=True, sort=False)
    median = grouped.transform("median")
    if name == "m_above":
        selected = values > median
    elif name == "m_below":
        selected = values < median
    else:
        q1 = grouped.transform(lambda x: x.quantile(0.25))
        q3 = grouped.transform(lambda x: x.quantile(0.75))
        if name == "m_inner":
            selected = values.between(q1, q3, inclusive="both")
        elif name == "m_outer":
            selected = (values < q1) | (values > q3)
        else:
            raise ValueError(f"Unknown unary mask operator: {name}")
    return values.where(selected)


def apply_mask_binary(
    name: str,
    values: pd.Series,
    condition: pd.Series,
    data: pd.DataFrame,
    window: int | None = None,
) -> pd.Series:
    if name in {"m_at_top", "m_at_bot"}:
        if window is None:
            raise ValueError(f"{name} requires a window")
        selected = _exact_rank_mask(condition, data, window, largest=name == "m_at_top")
    elif name == "m_when_pos":
        selected = condition > 0
    elif name == "m_when_gt":
        median = _group_transform(condition, data, "median")
        selected = condition > median
    else:
        raise ValueError(f"Unknown binary mask operator: {name}")
    return values.where(selected)


def _group_reduce(values: pd.Series, data: pd.DataFrame, operation: str) -> pd.Series:
    result = values.groupby([data["date"], data["code"]], observed=True, sort=True).agg(operation)
    result.index.names = ["date", "code"]
    return result.astype(float)


def _numpy_group_layout(data: pd.DataFrame) -> tuple[np.ndarray, pd.MultiIndex]:
    """Return contiguous group boundaries and their (date, code) keys."""
    dates = pd.to_datetime(data["date"]).to_numpy()
    codes = data["code"].astype(str).to_numpy()
    if len(data) == 0:
        return np.array([0], dtype=np.int64), pd.MultiIndex.from_arrays(
            [[], []], names=["date", "code"]
        )
    changes = np.empty(len(data), dtype=bool)
    changes[0] = True
    changes[1:] = (dates[1:] != dates[:-1]) | (codes[1:] != codes[:-1])
    starts = np.flatnonzero(changes)
    bounds = np.r_[starts, len(data)].astype(np.int64)
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates[starts]).normalize(), codes[starts]], names=["date", "code"]
    )
    return bounds, keys


def _numpy_unary_reduce(values: pd.Series, data: pd.DataFrame, name: str) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    array = np.where(np.isfinite(array), array, np.nan)
    bounds, keys = _numpy_group_layout(data)
    output = np.full(len(keys), np.nan, dtype=np.float64)
    for group_index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        raw = array[start:end]
        clean = raw[np.isfinite(raw)]
        n = clean.size
        if n == 0:
            continue
        if name == "r_skew":
            if n < 3:
                continue
            centered = clean - clean.mean()
            m2 = np.mean(centered * centered)
            if m2 <= EPS:
                output[group_index] = 0.0
            else:
                g1 = np.mean(centered ** 3) / (m2 ** 1.5)
                output[group_index] = np.sqrt(n * (n - 1)) / (n - 2) * g1
        elif name == "r_kurt":
            if n < 4:
                continue
            centered = clean - clean.mean()
            m2 = np.mean(centered * centered)
            if m2 <= EPS:
                output[group_index] = 0.0
            else:
                g2 = np.mean(centered ** 4) / (m2 * m2) - 3.0
                output[group_index] = ((n - 1) / ((n - 2) * (n - 3))) * (
                    (n + 1) * g2 + 6.0
                )
        elif name in {"r_slope", "r_rsquare"}:
            if n < 2:
                continue
            x = np.arange(n, dtype=np.float64)
            xc = x - x.mean()
            yc = clean - clean.mean()
            denominator = float(xc @ xc)
            slope = float(xc @ yc / denominator) if denominator > EPS else np.nan
            if name == "r_slope":
                output[group_index] = slope
            else:
                total = float(yc @ yc)
                residual = float(np.square(clean - (clean.mean() + slope * xc)).sum())
                output[group_index] = 1.0 - residual / total if total > EPS else np.nan
        elif name == "r_argmax":
            valid_positions = np.flatnonzero(np.isfinite(raw))
            maximum_position = valid_positions[int(np.argmax(raw[valid_positions]))]
            output[group_index] = maximum_position / max(1, len(raw) - 1)
        else:
            raise ValueError(f"Unknown NumPy unary reduction: {name}")
    return pd.Series(output, index=keys, dtype=float)


def _numpy_binary_reduce(
    left: pd.Series, right: pd.Series, data: pd.DataFrame, name: str
) -> pd.Series:
    left_array = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    right_array = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    bounds, keys = _numpy_group_layout(data)
    output = np.full(len(keys), np.nan, dtype=np.float64)
    for group_index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        x, y = left_array[start:end], right_array[start:end]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if name == "r_wmean":
            denominator = float(y.sum())
            if abs(denominator) > EPS:
                output[group_index] = float(np.dot(x, y) / denominator)
            continue
        if x.size < 2:
            continue
        xc, yc = x - x.mean(), y - y.mean()
        cross = float(np.dot(xc, yc))
        if name == "r_cov":
            output[group_index] = cross / (x.size - 1)
        elif name == "r_corr":
            denominator = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
            if denominator > EPS:
                output[group_index] = cross / denominator
        else:
            raise ValueError(f"Unknown NumPy binary reduction: {name}")
    return pd.Series(output, index=keys, dtype=float)


def apply_reduce_unary(name: str, values: pd.Series, data: pd.DataFrame) -> pd.Series:
    simple = {
        "r_mean": "mean", "r_std": "std", "r_sum": "sum", "r_max": "max",
        "r_min": "min", "r_median": "median", "r_first": "first", "r_last": "last",
    }
    if name in simple:
        return _group_reduce(values, data, simple[name])
    if name in {"r_skew", "r_kurt", "r_slope", "r_rsquare", "r_argmax"}:
        return _numpy_unary_reduce(values, data, name)
    raise ValueError(f"Unknown unary reduction operator: {name}")


def apply_reduce_binary(
    name: str, left: pd.Series, right: pd.Series, data: pd.DataFrame
) -> pd.Series:
    if name in {"r_corr", "r_cov", "r_wmean"}:
        return _numpy_binary_reduce(left, right, data, name)
    raise ValueError(f"Unknown binary reduction operator: {name}")
