from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from threading import Lock
from typing import Callable, Hashable

import pandas as pd


class SubexpressionCache:
    """Thread-safe, bounded LRU cache for expression-node results on one data frame."""

    def __init__(
        self,
        data: pd.DataFrame,
        max_entries: int = 128,
        max_bytes: int = 512 * 1024**2,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        keys = data[["code", "date"]]
        if not keys.equals(keys.sort_values(["code", "date"], kind="stable")):
            raise ValueError("SubexpressionCache data must be sorted by code and date")
        self._data_identity = id(data)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._values: OrderedDict[Hashable, tuple[pd.Series, int]] = OrderedDict()
        self._inflight: dict[Hashable, Future[pd.Series]] = {}
        self._lock = Lock()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._waits = 0
        self._evictions = 0
        self._oversized = 0

    def validate_data(self, data: pd.DataFrame) -> None:
        if id(data) != self._data_identity:
            raise ValueError("SubexpressionCache cannot be shared across different data frames")

    def get_or_compute(
        self,
        key: Hashable,
        compute: Callable[[], pd.Series],
    ) -> pd.Series:
        producer = False
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                self._hits += 1
                return cached[0]
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                self._misses += 1
                producer = True
            else:
                self._waits += 1

        if not producer:
            return future.result()

        try:
            value = compute()
            if not isinstance(value, pd.Series):
                raise TypeError("Subexpression computation must return pandas.Series")
        except BaseException as error:
            with self._lock:
                self._inflight.pop(key, None)
                future.set_exception(error)
            raise

        size = int(value.memory_usage(index=True, deep=True))
        with self._lock:
            if size <= self.max_bytes:
                self._values[key] = (value, size)
                self._bytes += size
                while (
                    len(self._values) > self.max_entries or self._bytes > self.max_bytes
                ):
                    _, (_, removed_size) = self._values.popitem(last=False)
                    self._bytes -= removed_size
                    self._evictions += 1
            else:
                self._oversized += 1
            self._inflight.pop(key, None)
            future.set_result(value)
        return value

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            hits = self._hits
            misses = self._misses
            total = hits + misses
            return {
                "hits": hits,
                "misses": misses,
                "waits": self._waits,
                "evictions": self._evictions,
                "oversized": self._oversized,
                "entries": len(self._values),
                "bytes": self._bytes,
                "memory_mb": self._bytes / 1024**2,
                "hit_rate": hits / total if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            if self._inflight:
                raise RuntimeError("Cannot clear cache while computations are in flight")
            self._values.clear()
            self._bytes = 0
