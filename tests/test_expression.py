from __future__ import annotations

import numpy as np

from src.expression import Expression, Node, expression_from_tokens


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

