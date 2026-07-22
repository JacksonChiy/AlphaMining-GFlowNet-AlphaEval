from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.expression import Expression, Node, SubexpressionCache, expression_from_tokens


def test_expression_round_trip_and_execution(daily_prices) -> None:
    expression = Expression(Node("cs", "cs_rank", (Node("ts", "ts_mean", (Node("feature", "close"),), 5),)))
    restored = expression_from_tokens(expression.to_tokens())
    values = restored.execute(daily_prices)
    assert str(restored) == "cs_rank(ts_mean(close,5))"
    assert restored.complexity() == 3
    assert values.notna().sum() > 0
    assert values.dropna().between(0, 1).all()


def test_time_series_operator_has_no_future_leakage(daily_prices) -> None:
    expression = Expression(Node("ts", "ts_mean", (Node("feature", "close"),), 5))
    baseline = expression.execute(daily_prices)
    changed = daily_prices.copy()
    cutoff = changed["date"].sort_values().unique()[70]
    changed.loc[changed["date"] > cutoff, "close"] *= 100.0
    modified = expression.execute(changed)
    before = changed["date"] <= cutoff
    assert np.allclose(baseline[before], modified[before], equal_nan=True)


def test_random_generation_respects_depth() -> None:
    expression = Expression.generate(max_depth=3, seed=3)
    assert expression.depth() <= 3


def test_repeated_subexpression_is_reused_across_expressions(
    daily_prices, monkeypatch
) -> None:
    ordered = daily_prices.sort_values(["code", "date"], kind="stable").copy()
    cache = SubexpressionCache(ordered, max_entries=16, max_bytes=32 * 1024**2)
    shared = Node("ts", "ts_mean", (Node("feature", "close"),), 5)
    first = Expression(shared)
    second = Expression(Node("binary", "add", (shared, Node("feature", "volume"))))

    import src.expression.tree as tree_module

    original = tree_module.apply_time_series
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tree_module, "apply_time_series", counted)
    first_values = first.execute(ordered, cache=cache)
    second_values = second.execute(ordered, cache=cache)

    assert calls == 1
    assert np.allclose(
        second_values,
        first_values + ordered["volume"].astype(float),
        equal_nan=True,
    )
    assert cache.stats()["hits"] >= 1


def test_subexpression_cache_single_flight_for_concurrent_workers(daily_prices) -> None:
    ordered = daily_prices.sort_values(["code", "date"], kind="stable").copy()
    cache = SubexpressionCache(ordered, max_entries=8, max_bytes=32 * 1024**2)
    barrier = threading.Barrier(4)
    compute_lock = threading.Lock()
    calls = 0

    def task() -> pd.Series:
        barrier.wait()

        def compute() -> pd.Series:
            nonlocal calls
            with compute_lock:
                calls += 1
            time.sleep(0.05)
            return ordered["close"].astype(float)

        return cache.get_or_compute("shared-close", compute)

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _: task(), range(4)))

    assert calls == 1
    assert cache.stats()["waits"] == 3
    assert all(value.equals(values[0]) for value in values)


def test_subexpression_cache_evicts_least_recently_used(daily_prices) -> None:
    ordered = daily_prices.sort_values(["code", "date"], kind="stable").copy()
    cache = SubexpressionCache(ordered, max_entries=2, max_bytes=32 * 1024**2)
    for name in ("open", "high", "low"):
        cache.get_or_compute(name, lambda name=name: ordered[name].astype(float))

    assert cache.stats()["entries"] == 2
    assert cache.stats()["evictions"] == 1
    misses_before = cache.stats()["misses"]
    cache.get_or_compute("open", lambda: ordered["open"].astype(float))
    assert cache.stats()["misses"] == misses_before + 1
