from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.index_enhancement.portfolio_optimizer import (
    PortfolioOptimizerConfig,
    latest_weight_date,
    load_weight_history_for_strategy,
    optimize_benchmark_portfolio,
)


def test_optimizer_respects_weight_active_and_turnover_constraints() -> None:
    ranked = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "smoothed_rank_score": [1.0, 0.75, 0.5, 0.25],
        }
    )
    benchmark = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    current = {"A": 0.39, "B": 0.29, "C": 0.20, "D": 0.10}
    config = PortfolioOptimizerConfig(
        max_names=4,
        alpha_top_n=4,
        cash_buffer=0.98,
        alpha_strength=0.02,
        risk_aversion=1.0,
        turnover_penalty=0.01,
        max_active_weight=0.03,
        max_stock_weight=0.5,
        max_rebalance_turnover=0.10,
    )

    result, diagnostics = optimize_benchmark_portfolio(
        ranked, benchmark, current, config
    )

    assert result["target_weight"].sum() == pytest.approx(0.98)
    assert result["active_weight"].abs().max() <= 0.03 + 1e-8
    assert diagnostics["gross_target_turnover"] <= 0.10 + 1e-7
    assert result.set_index("code").loc["A", "active_weight"] > 0


def test_optimizer_penalizes_high_cost_turnover() -> None:
    ranked = pd.DataFrame(
        {"code": ["A", "B"], "smoothed_rank_score": [1.0, 0.0]}
    )
    benchmark = pd.Series({"A": 0.5, "B": 0.5})
    current = {"A": 0.49, "B": 0.49}
    common = dict(
        max_names=2,
        alpha_top_n=2,
        cash_buffer=0.98,
        alpha_strength=0.05,
        risk_aversion=0.1,
        turnover_penalty=0.0,
        max_active_weight=0.2,
        max_stock_weight=0.8,
        max_rebalance_turnover=0.4,
    )
    low, _ = optimize_benchmark_portfolio(
        ranked,
        benchmark,
        current,
        PortfolioOptimizerConfig(**common, estimated_buy_cost=0, estimated_sell_cost=0),
    )
    high, _ = optimize_benchmark_portfolio(
        ranked,
        benchmark,
        current,
        PortfolioOptimizerConfig(
            **common, estimated_buy_cost=0.20, estimated_sell_cost=0.20
        ),
    )

    low_active = low.set_index("code").loc["A", "active_weight"]
    high_active = high.set_index("code").loc["A", "active_weight"]
    assert high_active < low_active


def test_projected_gradient_fallback_improves_failed_slsqp_run() -> None:
    ranked = pd.DataFrame(
        {
            "code": [f"S{i}" for i in range(20)],
            "smoothed_rank_score": np.linspace(1.0, 0.0, 20),
        }
    )
    benchmark = pd.Series(1 / 20, index=ranked["code"])
    current = {code: 0.049 for code in ranked["code"]}
    result, diagnostics = optimize_benchmark_portfolio(
        ranked,
        benchmark,
        current,
        PortfolioOptimizerConfig(
            max_names=20,
            alpha_top_n=20,
            alpha_strength=0.05,
            max_active_weight=0.02,
            max_stock_weight=0.2,
            max_iterations=1,
        ),
    )

    assert diagnostics["solver_method"] == "projected_gradient"
    assert diagnostics["success"] is True
    assert diagnostics["fallback_iterations"] > 0
    assert result["target_weight"].sum() == pytest.approx(0.98)
    assert diagnostics["gross_target_turnover"] <= 0.20 + 1e-7


def test_local_weight_loader_filters_one_index_and_uses_point_in_time_date(tmp_path) -> None:
    path = tmp_path / "weights.csv.gz"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "index_key": ["csi300", "csi300", "csi500"],
            "code": ["600000.SH", "000001.SZ", "000002.SZ"],
            "benchmark_weight": [0.6, 0.4, 1.0],
        }
    ).to_csv(path, index=False, compression="gzip")

    history = load_weight_history_for_strategy(path, "csi300")

    assert list(history) == [pd.Timestamp("2024-01-02").date()]
    assert history[next(iter(history))].sum() == pytest.approx(1.0)
    assert latest_weight_date(sorted(history), pd.Timestamp("2024-01-03").date()) == pd.Timestamp(
        "2024-01-02"
    ).date()
