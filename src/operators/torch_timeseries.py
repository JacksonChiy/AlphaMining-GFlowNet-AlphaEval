from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd
import torch


SUPPORTED_BACKENDS = {"auto", "pandas", "torch"}
SUPPORTED_DTYPES = {"float32": torch.float32, "float64": torch.float64}
SUPPORTED_OPERATORS = {
    "ts_mean",
    "ts_std",
    "ts_rank",
    "ts_delay",
    "ts_delta",
    "ts_sum",
    "ts_max",
    "ts_min",
    "ts_zscore",
}


@dataclass(frozen=True)
class TimeSeriesBackendConfig:
    backend: str = "auto"
    device: str = "auto"
    chunk_size: int = 64
    dtype: str = "float32"


_config = TimeSeriesBackendConfig()
_config_lock = Lock()
_stats_lock = Lock()
_runtime_stats: dict[str, float | int] = {
    "calls": 0,
    "torch_calls": 0,
    "pandas_calls": 0,
    "torch_seconds": 0.0,
}


def configure_time_series_backend(
    backend: str = "auto",
    device: str = "auto",
    chunk_size: int = 64,
    dtype: str = "float32",
) -> TimeSeriesBackendConfig:
    backend = backend.lower()
    dtype = dtype.lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported time-series backend: {backend}")
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported time-series dtype: {dtype}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if device != "auto":
        torch.device(device)
    configured = TimeSeriesBackendConfig(backend, device, int(chunk_size), dtype)
    global _config
    with _config_lock:
        _config = configured
    return configured


def configure_time_series_from_mapping(
    values: Mapping[str, object] | None,
) -> TimeSeriesBackendConfig:
    values = values or {}
    return configure_time_series_backend(
        backend=str(values.get("time_series_backend", "auto")),
        device=str(values.get("time_series_device", "auto")),
        chunk_size=int(values.get("time_series_chunk_size", 64)),
        dtype=str(values.get("time_series_dtype", "float32")),
    )


def get_time_series_backend_config() -> TimeSeriesBackendConfig:
    with _config_lock:
        return _config


def resolve_time_series_device(
    config: TimeSeriesBackendConfig | None = None,
) -> torch.device | None:
    config = config or get_time_series_backend_config()
    if config.backend == "pandas":
        return None
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if config.backend == "torch":
            return torch.device("cpu")
        return None
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if config.backend == "auto":
            return None
        raise RuntimeError("PyTorch CUDA time-series backend requested but CUDA is unavailable")
    return device


def get_time_series_backend_info() -> dict[str, str | int | bool]:
    config = get_time_series_backend_config()
    device = resolve_time_series_device(config)
    return {
        **asdict(config),
        "resolved_backend": "torch" if device is not None else "pandas",
        "resolved_device": str(device) if device is not None else "cpu",
        "cuda_available": torch.cuda.is_available(),
    }


def get_time_series_runtime_stats() -> dict[str, float | int]:
    with _stats_lock:
        return dict(_runtime_stats)


def record_pandas_time_series_call() -> None:
    with _stats_lock:
        _runtime_stats["calls"] += 1
        _runtime_stats["pandas_calls"] += 1


def _record_torch_call(seconds: float) -> None:
    with _stats_lock:
        _runtime_stats["calls"] += 1
        _runtime_stats["torch_calls"] += 1
        _runtime_stats["torch_seconds"] += seconds


def apply_time_series_torch(
    name: str,
    value: pd.Series,
    code: pd.Series,
    window: int,
    config: TimeSeriesBackendConfig | None = None,
) -> pd.Series:
    if name not in SUPPORTED_OPERATORS:
        raise KeyError(f"Unknown time-series operator: {name}")
    if window <= 0:
        raise ValueError("window must be positive")
    config = config or get_time_series_backend_config()
    device = resolve_time_series_device(config)
    if device is None:
        raise RuntimeError("Torch time-series implementation requires a torch device")
    if len(value) != len(code):
        raise ValueError("value and code must have the same length")
    if value.empty:
        return pd.Series(index=value.index, dtype=float, name=value.name)

    started = perf_counter()
    group_ids, unique_codes = pd.factorize(code, sort=False)
    if np.any(group_ids < 0):
        raise ValueError("code contains missing values")
    blocks = 1 + int(np.count_nonzero(group_ids[1:] != group_ids[:-1]))
    if blocks != len(unique_codes):
        raise ValueError("PyTorch time-series backend requires rows grouped by code")

    lengths = np.bincount(group_ids, minlength=len(unique_codes)).astype(np.int64)
    starts = np.cumsum(np.r_[0, lengths[:-1]])
    rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    columns = np.arange(len(value), dtype=np.int64) - np.repeat(starts, lengths)
    numpy_dtype = np.float32 if config.dtype == "float32" else np.float64
    flat_values = pd.to_numeric(value, errors="coerce").to_numpy(
        dtype=numpy_dtype, copy=True
    )
    padded = np.full((len(lengths), int(lengths.max())), np.nan, dtype=numpy_dtype)
    padded[rows, columns] = flat_values
    cpu_tensor = torch.from_numpy(padded)
    if device.type == "cuda":
        cpu_tensor = cpu_tensor.pin_memory()
    tensor = cpu_tensor.to(device=device, dtype=SUPPORTED_DTYPES[config.dtype], non_blocking=True)
    result = torch.full_like(tensor, torch.nan)

    with torch.inference_mode():
        for start in range(0, tensor.shape[0], config.chunk_size):
            stop = min(start + config.chunk_size, tensor.shape[0])
            chunk = tensor[start:stop]
            if name == "ts_delay":
                if window < chunk.shape[1]:
                    result[start:stop, window:] = chunk[:, :-window]
                continue
            if name == "ts_delta":
                if window < chunk.shape[1]:
                    result[start:stop, window:] = (
                        chunk[:, window:] - chunk[:, :-window]
                    )
                continue
            if window > chunk.shape[1]:
                continue

            rolling = chunk.unfold(dimension=1, size=window, step=1)
            valid = torch.isfinite(rolling).all(dim=-1)
            if name == "ts_mean":
                calculated = rolling.mean(dim=-1)
            elif name == "ts_std":
                calculated = rolling.std(dim=-1, correction=1)
            elif name == "ts_sum":
                calculated = rolling.sum(dim=-1)
            elif name == "ts_max":
                calculated = rolling.amax(dim=-1)
            elif name == "ts_min":
                calculated = rolling.amin(dim=-1)
            elif name == "ts_rank":
                latest = rolling[..., -1:]
                less = (rolling < latest).sum(dim=-1).to(rolling.dtype)
                equal = (rolling == latest).sum(dim=-1).to(rolling.dtype)
                calculated = (less + (equal + 1) * 0.5) / window
            elif name == "ts_zscore":
                mean = rolling.mean(dim=-1)
                std = rolling.std(dim=-1, correction=1)
                calculated = (rolling[..., -1] - mean) / std
                valid = valid & (std.abs() > 1e-12)
            else:
                raise AssertionError(name)
            calculated = torch.where(valid, calculated, torch.nan)
            result[start:stop, window - 1 :] = calculated

    result_numpy = result.to(device="cpu").numpy()
    flat_result = result_numpy[rows, columns]
    _record_torch_call(perf_counter() - started)
    return pd.Series(flat_result, index=value.index, name=value.name)
