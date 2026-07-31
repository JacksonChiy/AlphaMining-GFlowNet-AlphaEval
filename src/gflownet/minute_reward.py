from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from src.expression.minute import MinuteExpression
from src.operators.minute import build_minute_features

from .reward import RewardBreakdown, RewardEvaluator


class MinuteRewardEvaluator:
    """Execute an intraday expression, align its daily output, then reuse the daily reward."""

    def __init__(self, minute_data: pd.DataFrame, daily_data: pd.DataFrame, **reward_options) -> None:
        self.minute_data = build_minute_features(minute_data)
        self.daily_evaluator = RewardEvaluator(daily_data, **reward_options)
        self.reward_floor = self.daily_evaluator.reward_floor
        self.min_coverage = self.daily_evaluator.min_coverage
        self.subexpression_cache = None
        self.cache: dict[str, RewardBreakdown] = {}
        self._lock = threading.RLock()

    def evaluate(self, expression: MinuteExpression) -> RewardBreakdown:
        key = str(expression)
        with self._lock:
            cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            daily_factor = expression.execute(self.minute_data)
            if not isinstance(daily_factor.index, pd.MultiIndex):
                raise ValueError("Minute expression output must use a (date, code) MultiIndex")
            normalized = daily_factor.copy()
            normalized.index = pd.MultiIndex.from_arrays(
                [pd.to_datetime(normalized.index.get_level_values(0)).normalize(),
                 normalized.index.get_level_values(1).astype(str)],
                names=["date", "code"],
            )
            daily = self.daily_evaluator.data
            keys = pd.MultiIndex.from_arrays(
                [pd.to_datetime(daily["date"]).dt.normalize(), daily["code"].astype(str)],
                names=["date", "code"],
            )
            aligned = pd.Series(normalized.reindex(keys).to_numpy(dtype=float), index=daily.index)
            result = self.daily_evaluator.evaluate_factor(aligned)
        except (FloatingPointError, ValueError, KeyError, OverflowError, TypeError):
            result = self.daily_evaluator._empty_breakdown()
        with self._lock:
            self.cache[key] = result
        return result

    def cache_stats(self) -> dict[str, int | float | bool]:
        return {
            "enabled": False,
            "hits": 0,
            "misses": len(self.cache),
            "waits": 0,
            "evictions": 0,
            "oversized": 0,
            "entries": len(self.cache),
            "bytes": 0,
            "memory_mb": 0.0,
            "hit_rate": 0.0,
        }
