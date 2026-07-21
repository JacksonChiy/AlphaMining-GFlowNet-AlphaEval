from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig, greedy_dpp_select


def test_greedy_dpp_prefers_non_degenerate_items() -> None:
    kernel = np.array([[2.0, 1.8, 0.0], [1.8, 2.0, 0.0], [0.0, 0.0, 1.0]])
    selected = greedy_dpp_select(kernel, 2)
    assert selected[0] in (0, 1)
    assert 2 in selected


def test_alpha_eval_output_contract(daily_prices) -> None:
    factors = daily_prices[["date", "code"]].copy()
    grouped = daily_prices.groupby("code", observed=True)["close"]
    factors["factor_001"] = grouped.pct_change(5)
    factors["factor_002"] = -factors["factor_001"]
    factors["factor_003"] = daily_prices["volume"]
    metadata = pd.DataFrame({
        "factor": ["factor_001", "factor_002", "factor_003"],
        "complexity": [3, 3, 1],
        "depth": [3, 3, 1],
    })
    config = AlphaEvalConfig(
        horizon=5, rolling_window=20, min_cross_section=10, dpp_k=2, seed=11
    )
    result = AlphaEval(daily_prices, factors, config).evaluate(metadata, output_path=None)
    required = {"factor", "IC", "RankIC", "ICIR", "Sharpe", "complexity", "score"}
    assert required.issubset(result.columns)
    assert result["dpp_selected"].sum() <= 2
    assert np.isfinite(result["score"]).all()

