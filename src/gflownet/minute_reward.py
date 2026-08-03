from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Sequence

import numpy as np
import pandas as pd

from src.expression.minute import MinuteExpression
from src.operators.minute import build_minute_features
from src.data_loader.dolphindb_minute import DolphinDBMinuteLoader

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


class DolphinDBStreamingMinuteRewardEvaluator:
    """Evaluate report-style expressions by streaming DDB once per expression batch."""

    def __init__(
        self,
        loader: DolphinDBMinuteLoader,
        daily_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        block_cache_max_entries: int = 256,
        **reward_options,
    ) -> None:
        if block_cache_max_entries < 1:
            raise ValueError("block_cache_max_entries must be positive")
        self.loader = loader
        self.start_date = start_date
        self.end_date = end_date
        self.daily_evaluator = RewardEvaluator(daily_data, **reward_options)
        self.reward_floor = self.daily_evaluator.reward_floor
        self.min_coverage = self.daily_evaluator.min_coverage
        self.subexpression_cache = None
        self.cache: dict[str, RewardBreakdown] = {}
        self.block_cache: OrderedDict[str, pd.Series] = OrderedDict()
        self.block_cache_max_entries = block_cache_max_entries
        self._block_hits = 0
        self._block_misses = 0
        self._block_evictions = 0
        self._lock = threading.RLock()

    def evaluate(self, expression: MinuteExpression) -> RewardBreakdown:
        return self.evaluate_many([expression])[0]

    def evaluate_many(
        self, expressions: Sequence[MinuteExpression]
    ) -> list[RewardBreakdown]:
        unique: dict[str, MinuteExpression] = {}
        for expression in expressions:
            unique.setdefault(str(expression), expression)
        pending = [expression for key, expression in unique.items() if key not in self.cache]
        required_blocks: dict[str, object] = {}
        for expression in pending:
            for node in expression.block_nodes():
                key = node.render()
                if key in self.block_cache:
                    self._block_hits += 1
                    self.block_cache.move_to_end(key)
                else:
                    required_blocks.setdefault(key, node)
        self._block_misses += len(required_blocks)

        if required_blocks:
            block_parts: dict[str, list[pd.Series]] = {
                key: [] for key in required_blocks
            }
            nodes = list(required_blocks.values())
            executor_expression = pending[0]
            for chunk_index, (_, _, minute) in enumerate(
                self.loader.iter_frames(self.start_date, self.end_date), start=1
            ):
                computed = executor_expression.execute_blocks(nodes, minute)
                for key, values in computed.items():
                    block_parts[key].append(values)
                print(
                    f"[DDBReward] chunk_complete index={chunk_index:03d} "
                    f"new_blocks={len(required_blocks):03d} expressions={len(pending):03d}",
                    flush=True,
                )
            for key, parts in block_parts.items():
                if not parts:
                    continue
                values = pd.concat(parts).sort_index()
                values = values[~values.index.duplicated(keep="last")]
                self._put_block(key, values)

        for expression in pending:
            try:
                blocks = {
                    node.render(): self.block_cache[node.render()]
                    for node in expression.block_nodes()
                }
                factor = expression.execute_from_blocks(blocks)
                result = self._evaluate_factor(factor)
            except (FloatingPointError, ValueError, KeyError, OverflowError, TypeError):
                result = self.daily_evaluator._empty_breakdown()
            self.cache[str(expression)] = result
        return [self.cache[str(expression)] for expression in expressions]

    def _evaluate_factor(self, factor: pd.Series) -> RewardBreakdown:
        if not isinstance(factor.index, pd.MultiIndex):
            raise ValueError("Report minute expression must produce a daily MultiIndex")
        normalized = factor.copy()
        normalized.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(normalized.index.get_level_values(0)).normalize(),
                normalized.index.get_level_values(1).astype(str),
            ],
            names=["date", "code"],
        )
        daily = self.daily_evaluator.data
        keys = pd.MultiIndex.from_arrays(
            [pd.to_datetime(daily["date"]).dt.normalize(), daily["code"].astype(str)],
            names=["date", "code"],
        )
        aligned = pd.Series(normalized.reindex(keys).to_numpy(dtype=float), index=daily.index)
        return self.daily_evaluator.evaluate_factor(aligned)

    def _put_block(self, key: str, values: pd.Series) -> None:
        self.block_cache[key] = values
        self.block_cache.move_to_end(key)
        while len(self.block_cache) > self.block_cache_max_entries:
            self.block_cache.popitem(last=False)
            self._block_evictions += 1

    def cache_stats(self) -> dict[str, int | float | bool]:
        accesses = self._block_hits + self._block_misses
        memory_bytes = int(sum(value.memory_usage(deep=True) for value in self.block_cache.values()))
        return {
            "enabled": True,
            "hits": self._block_hits,
            "misses": self._block_misses,
            "waits": 0,
            "evictions": self._block_evictions,
            "oversized": 0,
            "entries": len(self.block_cache),
            "bytes": memory_bytes,
            "memory_mb": memory_bytes / 1024**2,
            "hit_rate": self._block_hits / accesses if accesses else 0.0,
        }
