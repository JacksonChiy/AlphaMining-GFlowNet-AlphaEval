from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig, greedy_dpp_select
from src.alpha_eval.evaluator import _daily_corr


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


def test_vectorized_daily_corr_matches_pandas() -> None:
    rng = np.random.default_rng(7)
    work = pd.DataFrame({
        "date": np.repeat(pd.date_range("2024-01-01", periods=8), 30),
        "factor": rng.normal(size=240),
        "target": rng.normal(size=240),
    })
    work.loc[[2, 31, 99], "factor"] = np.nan
    work.loc[[5, 61], "target"] = np.nan
    for method in ("pearson", "spearman"):
        expected = (
            work.dropna()
            .groupby("date", observed=True)[["factor", "target"]]
            .apply(lambda group: group["factor"].corr(group["target"], method=method))
            .dropna()
        )
        actual = _daily_corr(work, method, min_cross_section=20)
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-11, atol=1e-11)


def test_alpha_eval_prints_progress(daily_prices, capsys) -> None:
    factors = daily_prices[["date", "code"]].copy()
    factors["factor_001"] = daily_prices.groupby("code", observed=True)["close"].pct_change(3)
    config = AlphaEvalConfig(
        horizon=5, rolling_window=10, min_cross_section=10, dpp_k=1,
        dpp_max_rows=100, verbose=True,
    )
    AlphaEval(daily_prices, factors, config).evaluate(output_path=None)
    output = capsys.readouterr().out
    assert "因子 1/1 factor_001 [1/4] IC/RankIC 完成" in output
    assert "构建 DPP 多样性矩阵" in output
    assert "全部完成" in output
