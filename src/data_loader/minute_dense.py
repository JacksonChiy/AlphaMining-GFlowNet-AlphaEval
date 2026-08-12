from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.operators.minute import EPS


MINUTE_CHANNELS = (
    "open", "high", "low", "close", "vol", "amount",
    "ret", "vwap", "hl_pct", "bar_pos", "amihud", "rv",
    "signed_vol", "signed_amt", "typical", "vwap_cum", "twap", "obv", "pvt",
)
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


def _configured_minute_numbers(minute_lookup: Mapping[str, int]) -> np.ndarray:
    lookup = np.full(24 * 60, -1, dtype=np.int16)
    for value, index in minute_lookup.items():
        parsed = pd.Timestamp(f"2000-01-01 {value}")
        lookup[parsed.hour * 60 + parsed.minute] = int(index)
    return lookup


def _minute_numbers(values: pd.Series) -> np.ndarray:
    """Decode common DolphinDB time representations without row-wise pandas parsing."""
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

    def decode(value: object) -> int:
        if value is None or value is pd.NaT:
            return -1
        if hasattr(value, "hour") and hasattr(value, "minute"):
            return int(value.hour) * 60 + int(value.minute)
        text = str(value).strip()
        parts = text.rsplit(" ", 1)[-1].split(":")
        if len(parts) >= 2 and parts[0][-2:].isdigit() and parts[1][:2].isdigit():
            return int(parts[0][-2:]) * 60 + int(parts[1][:2])
        parsed = pd.to_datetime(text, errors="coerce")
        return -1 if pd.isna(parsed) else int(parsed.hour) * 60 + int(parsed.minute)

    return np.fromiter(
        (decode(value) for value in values.array), dtype=np.int64, count=len(values)
    )


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
    last_seen = np.maximum.accumulate(np.where(mask, minute_positions, -1), axis=0)
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


def build_dense_minute_channels(
    frame: pd.DataFrame,
    stock_codes: Sequence[str],
    minute_lookup: Mapping[str, int],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float | int]]:
    """Map one raw DolphinDB day to dense arrays and derive stored leaf channels."""
    missing = sorted(set(FAST_LOAD_SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"DolphinDB fast minute query is missing columns: {missing}")
    decode_started = time.perf_counter()
    minute_numbers = _minute_numbers(frame["time"])
    minute_number_lookup = _configured_minute_numbers(minute_lookup)
    minute_index = np.full(len(frame), -1, dtype=np.int64)
    valid_minute_number = (minute_numbers >= 0) & (
        minute_numbers < len(minute_number_lookup)
    )
    minute_index[valid_minute_number] = minute_number_lookup[
        minute_numbers[valid_minute_number]
    ]

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
    for name in MINUTE_CHANNELS:
        channels[name] = np.where(mask, channels[name], np.nan)
    numpy_feature_s = time.perf_counter() - feature_started
    return channels, mask, {
        "decode_index_s": decode_index_s,
        "base_matrix_s": base_matrix_s,
        "numpy_feature_s": numpy_feature_s,
        "valid_rows": int(valid.sum()),
        "excluded_rows": int(len(frame) - valid.sum()),
    }
