from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError

from src.data_loader.minute_memmap import MinuteMemMapStore
from src.expression.minute import MinuteExpression, MinuteNode

from .numpy_minute_executor import (
    NUMPY_MINUTE_EXECUTOR_VERSION,
    NumpyMinuteBlockExecutor,
    UnsupportedNumpyNode,
    required_memmap_channels,
    validate_numpy_nodes,
    )

from .reward import RewardBreakdown, RewardEvaluator


_NUMPY_WORKER_STORES: dict[str, MinuteMemMapStore] = {}


def _compute_memmap_tile(
    store_config,
    nodes: Sequence[MinuteNode],
    spec: tuple[int, int, int],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    store = MinuteMemMapStore(store_config)
    year, stock_start, stock_end = spec
    executor = MinuteExpression(nodes[0])
    parts: dict[str, list[pd.Series]] = {node.render(): [] for node in nodes}
    for _, _, frame in store.iter_tile_frames(
        year, stock_start, stock_end, start_date, end_date
    ):
        computed = executor.execute_blocks(nodes, frame)
        for key, values in computed.items():
            parts[key].append(values)
    return {
        key: pd.concat(values).sort_index() if values else pd.Series(dtype=float)
        for key, values in parts.items()
    }


def _compute_numpy_chunk(
    store_config,
    nodes: Sequence[MinuteNode],
    spec: tuple[int, int, int, int, int],
) -> tuple[tuple[int, int, int, int, int], dict[str, np.ndarray], tuple[str, ...]]:
    store_key = str(store_config.root.resolve())
    store = _NUMPY_WORKER_STORES.get(store_key)
    if store is None:
        store = MinuteMemMapStore(store_config)
        _NUMPY_WORKER_STORES[store_key] = store
    channels = required_memmap_channels(nodes)
    _, mask, arrays = store.read_numpy_chunk(spec, channels)
    values = NumpyMinuteBlockExecutor(mask, arrays).execute(nodes)
    return spec, values, channels


class PersistentMinuteBlockCache:
    """Dense daily float32 block cache keyed by source/date scope/expression."""

    def __init__(self, store: MinuteMemMapStore, root: Path) -> None:
        self.store = store
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.disk_hits = 0
        self.disk_misses = 0
        self.writes = 0

    def load(self, expression: str, start_date: str, end_date: str) -> pd.Series | None:
        data_path, meta_path = self._paths(expression, start_date, end_date)
        if not data_path.exists() or not meta_path.exists():
            self.disk_misses += 1
            return None
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("source_fingerprint") != self.store.fingerprint:
            self.disk_misses += 1
            return None
        dates = self._dates(start_date, end_date)
        expected_shape = (len(dates), len(self.store.stocks))
        values = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if values.shape != expected_shape:
            self.disk_misses += 1
            return None
        row_index, stock_index = np.nonzero(np.isfinite(values))
        index = pd.MultiIndex.from_arrays(
            [dates[row_index], self.store.stocks[stock_index].astype(str)],
            names=["date", "code"],
        )
        self.disk_hits += 1
        return pd.Series(
            np.asarray(values[row_index, stock_index], dtype=float),
            index=index,
            name=expression,
        )

    def save(
        self,
        expression: str,
        start_date: str,
        end_date: str,
        values: pd.Series,
    ) -> None:
        data_path, meta_path = self._paths(expression, start_date, end_date)
        dates = self._dates(start_date, end_date)
        date_lookup = {date: index for index, date in enumerate(dates)}
        date_values = pd.to_datetime(values.index.get_level_values(0)).normalize()
        stock_values = values.index.get_level_values(1).astype(str)
        row_index = np.array([date_lookup.get(date, -1) for date in date_values], dtype=np.int64)
        stock_index = np.array(
            [self.store.stock_lookup.get(code, -1) for code in stock_values], dtype=np.int64
        )
        valid = (row_index >= 0) & (stock_index >= 0)
        temporary = data_path.with_suffix(".npy.tmp")
        array = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(len(dates), len(self.store.stocks)),
        )
        array[:] = np.nan
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
        array[row_index[valid], stock_index[valid]] = numeric[valid]
        array.flush()
        del array
        temporary.replace(data_path)
        temporary_meta = meta_path.with_suffix(".json.tmp")
        temporary_meta.write_text(json.dumps({
            "source_fingerprint": self.store.fingerprint,
            "expression": expression,
            "start_date": str(pd.Timestamp(start_date).date()),
            "end_date": str(pd.Timestamp(end_date).date()),
            "shape": [len(dates), len(self.store.stocks)],
            "dtype": "float32",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_meta.replace(meta_path)
        self.writes += 1

    def _dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        return self.store.dates[(self.store.dates >= start) & (self.store.dates <= end)]

    def _paths(self, expression: str, start_date: str, end_date: str) -> tuple[Path, Path]:
        payload = "|".join((self.store.fingerprint, str(start_date), str(end_date), expression))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return self.root / f"{digest}.npy", self.root / f"{digest}.json"


class PartialMinuteBlockCache:
    """Small daily chunk files make long reward scans resumable after interruption."""

    def __init__(
        self,
        store: MinuteMemMapStore,
        root: Path,
        start_date: str,
        end_date: str,
    ) -> None:
        scope = "|".join((
            store.fingerprint, str(pd.Timestamp(start_date).date()),
            str(pd.Timestamp(end_date).date()), str(NUMPY_MINUTE_EXECUTOR_VERSION),
        ))
        self.root = root / "partials" / hashlib.sha256(scope.encode()).hexdigest()[:20]
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.writes = 0

    def path(self, expression: str, spec: tuple[int, int, int, int, int]) -> Path:
        expression_key = hashlib.sha256(expression.encode()).hexdigest()[:20]
        return self.root / expression_key / ("_".join(map(str, spec)) + ".npy")

    def load(
        self, expression: str, spec: tuple[int, int, int, int, int]
    ) -> np.ndarray | None:
        path = self.path(expression, spec)
        if not path.exists():
            return None
        expected = (spec[2] - spec[1], spec[4] - spec[3])
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != expected:
            return None
        self.hits += 1
        return values

    def save(
        self,
        expression: str,
        spec: tuple[int, int, int, int, int],
        values: np.ndarray,
    ) -> None:
        path = self.path(expression, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
        temporary.replace(path)
        self.writes += 1


def _run_pandas_blocks(
    store: MinuteMemMapStore,
    nodes: Sequence[MinuteNode],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    parts: dict[str, list[pd.Series]] = {node.render(): [] for node in nodes}
    specs = store.tile_specs(start_date, end_date)
    print(
        f"[MemMapReward] pandas_fallback_start workers={store.config.workers} "
        f"year_stock_tiles={len(specs)}",
        flush=True,
    )
    if store.config.workers > 1:
        try:
            results = Parallel(n_jobs=store.config.workers, backend="loky", verbose=10)(
                delayed(_compute_memmap_tile)(
                    store.config, nodes, spec, start_date, end_date
                )
                for spec in specs
            )
        except (PermissionError, NotImplementedError) as error:
            print(
                f"[MemMapReward] pandas_loky_unavailable sequential_fallback=true "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            results = [
                _compute_memmap_tile(store.config, nodes, spec, start_date, end_date)
                for spec in specs
            ]
    else:
        results = [
            _compute_memmap_tile(store.config, nodes, spec, start_date, end_date)
            for spec in specs
        ]
    for completed, computed in enumerate(results, start=1):
        for key, values in computed.items():
            if not values.empty:
                parts[key].append(values)
        if completed == 1 or completed == len(results) or completed % 10 == 0:
            print(f"[MemMapReward] pandas_progress={completed}/{len(results)}", flush=True)
    return {
        key: pd.concat(values).sort_index()
        for key, values in parts.items() if values
    }


def _execute_numpy_blocks(
    store: MinuteMemMapStore,
    nodes: Sequence[MinuteNode],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    validate_numpy_nodes(nodes)
    specs = store.chunk_specs(start_date, end_date)
    if not specs:
        raise ValueError("MemMap store returned no dates for the requested range")
    partial = PartialMinuteBlockCache(
        store, store.config.block_cache_dir, start_date, end_date
    )
    missing_tasks: list[tuple[tuple[int, int, int, int, int], list[MinuteNode]]] = []
    cached_parts = 0
    for spec in specs:
        missing = []
        for node in nodes:
            if partial.load(node.render(), spec) is None:
                missing.append(node)
            else:
                cached_parts += 1
        for offset in range(0, len(missing), store.config.reward_blocks_per_task):
            task_nodes = missing[offset:offset + store.config.reward_blocks_per_task]
            if task_nodes:
                missing_tasks.append((spec, task_nodes))
    channels = required_memmap_channels(nodes)
    total_parts = len(specs) * len(nodes)
    max_channels = max(
        (len(required_memmap_channels(task_nodes)) for _, task_nodes in missing_tasks),
        default=0,
    )
    max_complexity = max(
        (sum(node.complexity() for node in task_nodes) for _, task_nodes in missing_tasks),
        default=0,
    )
    elements = (
        store.config.reward_chunk_days * len(store.minute_grid) *
        min(store.config.stock_tile_size, len(store.stocks))
    )
    estimated_mb = elements * (max_channels * 4 + max_complexity * 8 + 1) / 1024**2
    print(
        f"[MemMapReward] numpy_start workers={store.config.workers} "
        f"parallel_backend={store.config.reward_parallel_backend} "
        f"date_chunk_days={store.config.reward_chunk_days} chunks={len(specs)} "
        f"blocks={len(nodes)} blocks_per_task={store.config.reward_blocks_per_task} "
        f"required_channels={len(channels)}/{len(store.manifest['channels'])} "
        f"estimated_peak_mb_per_worker<={estimated_mb:.0f} "
        f"partial_hits={cached_parts}/{total_parts} pending_tasks={len(missing_tasks)}",
        flush=True,
    )
    started = time.monotonic()
    if missing_tasks:
        if store.config.workers > 1:
            try:
                results = Parallel(
                    n_jobs=store.config.workers,
                    backend=store.config.reward_parallel_backend,
                    batch_size=1,
                    # Ordered streaming is supported by both the pinned project
                    # joblib and newer releases, and still lets the parent save
                    # each completed chunk instead of buffering the full scan.
                    return_as="generator",
                )(
                    delayed(_compute_numpy_chunk)(store.config, task_nodes, spec)
                    for spec, task_nodes in missing_tasks
                )
            except (PermissionError, NotImplementedError, TerminatedWorkerError) as error:
                print(
                    f"[MemMapReward] loky_unavailable sequential_fallback=true "
                    f"error={type(error).__name__}: {error}",
                    flush=True,
                )
                results = (
                    _compute_numpy_chunk(store.config, task_nodes, spec)
                    for spec, task_nodes in missing_tasks
                )
        else:
            results = (
                _compute_numpy_chunk(store.config, task_nodes, spec)
                for spec, task_nodes in missing_tasks
            )
        completed = 0

        def persist_result(result) -> None:
            nonlocal completed
            spec, computed, _ = result
            completed += 1
            for expression, values in computed.items():
                partial.save(expression, spec, values)
            elapsed = max(time.monotonic() - started, 1e-9)
            if completed == 1 or completed == len(missing_tasks) or completed % 10 == 0:
                rate = completed / elapsed
                eta = (len(missing_tasks) - completed) / max(rate, 1e-9)
                print(
                    f"[MemMapReward] numpy_progress tasks={completed}/{len(missing_tasks)} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}m partial_writes={partial.writes}",
                    flush=True,
                )

        try:
            for result in results:
                persist_result(result)
        except TerminatedWorkerError as error:
            print(
                f"[MemMapReward] worker_terminated sequential_resume=true "
                f"completed={completed}/{len(missing_tasks)} error={error}",
                flush=True,
            )
            for spec, task_nodes in missing_tasks:
                remaining = [
                    node for node in task_nodes
                    if partial.load(node.render(), spec) is None
                ]
                if remaining:
                    persist_result(_compute_numpy_chunk(store.config, remaining, spec))

    selected_dates = store.dates[
        (store.dates >= pd.Timestamp(start_date)) & (store.dates <= pd.Timestamp(end_date))
    ]
    date_lookup = {date: index for index, date in enumerate(selected_dates)}
    output: dict[str, pd.Series] = {}
    for node in nodes:
        dense = np.full((len(selected_dates), len(store.stocks)), np.nan, dtype=np.float32)
        for spec in specs:
            year, day_start, day_end, stock_start, stock_end = spec
            dates = store._year_dates(year)[day_start:day_end]
            row_start = date_lookup[dates[0]]
            values = partial.load(node.render(), spec)
            if values is None:
                raise RuntimeError(f"Missing partial cache after computation: {node.render()} {spec}")
            dense[row_start:row_start + len(dates), stock_start:stock_end] = values
        row_index, stock_index = np.nonzero(np.isfinite(dense))
        index = pd.MultiIndex.from_arrays(
            [selected_dates[row_index], store.stocks[stock_index].astype(str)],
            names=["date", "code"],
        )
        output[node.render()] = pd.Series(
            dense[row_index, stock_index].astype(float), index=index, name=node.render()
        )
    print(
        f"[MemMapReward] numpy_complete seconds={time.monotonic()-started:.1f} "
        f"partial_hits={partial.hits} partial_writes={partial.writes} pandas_rows_built=0",
        flush=True,
    )
    return output


def execute_memmap_blocks(
    store: MinuteMemMapStore,
    nodes: Sequence[MinuteNode],
    start_date: str,
    end_date: str,
    cache: PersistentMinuteBlockCache,
) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    pending: list[MinuteNode] = []
    for node in nodes:
        cached = cache.load(node.render(), start_date, end_date)
        if cached is None:
            pending.append(node)
        else:
            output[node.render()] = cached
    if not pending:
        return output
    if store.config.reward_backend == "pandas":
        computed_blocks = _run_pandas_blocks(store, pending, start_date, end_date)
    else:
        try:
            computed_blocks = _execute_numpy_blocks(store, pending, start_date, end_date)
        except UnsupportedNumpyNode as error:
            if not store.config.numpy_fallback:
                raise
            print(
                f"[MemMapReward] numpy_unsupported pandas_fallback=true error={error}",
                flush=True,
            )
            computed_blocks = _run_pandas_blocks(store, pending, start_date, end_date)
    for node in pending:
        key = node.render()
        if key not in computed_blocks or computed_blocks[key].empty:
            raise ValueError(f"MemMap produced no values for block: {key}")
        values = computed_blocks[key].sort_index()
        values = values[~values.index.duplicated(keep="last")]
        cache.save(key, start_date, end_date, values)
        output[key] = values
    return output


class MemMapMinuteRewardEvaluator:
    def __init__(
        self,
        store: MinuteMemMapStore,
        daily_data: pd.DataFrame,
        start_date: str,
        end_date: str,
        block_cache_max_entries: int = 16,
        **reward_options,
    ) -> None:
        if block_cache_max_entries < 1:
            raise ValueError("block_cache_max_entries must be positive")
        self.store = store
        self.start_date = start_date
        self.end_date = end_date
        self.daily_evaluator = RewardEvaluator(daily_data, **reward_options)
        self.reward_floor = self.daily_evaluator.reward_floor
        self.min_coverage = self.daily_evaluator.min_coverage
        self.subexpression_cache = None
        self.cache: dict[str, RewardBreakdown] = {}
        self.block_cache: OrderedDict[str, pd.Series] = OrderedDict()
        self.block_cache_max_entries = block_cache_max_entries
        self.persistent_cache = PersistentMinuteBlockCache(
            store, store.config.block_cache_dir
        )
        self._block_hits = 0
        self._block_misses = 0
        self._block_evictions = 0
        self._lock = threading.RLock()

    def evaluate(self, expression: MinuteExpression) -> RewardBreakdown:
        return self.evaluate_many([expression])[0]

    def evaluate_many(
        self, expressions: Sequence[MinuteExpression]
    ) -> list[RewardBreakdown]:
        unique = {str(expression): expression for expression in expressions}
        pending_expressions = [
            expression for key, expression in unique.items() if key not in self.cache
        ]
        required: dict[str, MinuteNode] = {}
        for expression in pending_expressions:
            for node in expression.block_nodes():
                key = node.render()
                if key in self.block_cache:
                    self._block_hits += 1
                    self.block_cache.move_to_end(key)
                else:
                    required.setdefault(key, node)
        self._block_misses += len(required)
        if required:
            print(
                f"[MemMapReward] execution_plan new_blocks={len(required):03d} "
                "remote_ddb_queries=0 coarse_screen=false",
                flush=True,
            )
            computed = execute_memmap_blocks(
                self.store,
                list(required.values()),
                self.start_date,
                self.end_date,
                self.persistent_cache,
            )
            for key, values in computed.items():
                self._put_block(key, values)
        for expression in pending_expressions:
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
        normalized = factor.copy()
        normalized.index = pd.MultiIndex.from_arrays([
            pd.to_datetime(normalized.index.get_level_values(0)).normalize(),
            normalized.index.get_level_values(1).astype(str),
        ], names=["date", "code"])
        daily = self.daily_evaluator.data
        keys = pd.MultiIndex.from_arrays([
            pd.to_datetime(daily["date"]).dt.normalize(), daily["code"].astype(str)
        ], names=["date", "code"])
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
            "hits": self._block_hits + self.persistent_cache.disk_hits,
            "misses": self._block_misses + self.persistent_cache.disk_misses,
            "waits": 0,
            "evictions": self._block_evictions,
            "oversized": 0,
            "entries": len(self.block_cache),
            "bytes": memory_bytes,
            "memory_mb": memory_bytes / 1024**2,
            "hit_rate": self._block_hits / accesses if accesses else 0.0,
            "persistent_hits": self.persistent_cache.disk_hits,
            "persistent_writes": self.persistent_cache.writes,
        }
