import pandas as pd
import pytest

from src.index_enhancement.factor_attribution import (
    AttributionConfig,
    compute_active_exposure,
    estimate_factor_returns,
)


def test_compute_active_exposure_uses_normalized_equity_weights() -> None:
    date = pd.Timestamp("2024-01-02")
    exposure = pd.DataFrame(
        {
            "date": [date, date],
            "order_book_id": ["A", "B"],
            "size": [1.0, -1.0],
        }
    )
    portfolio = pd.DataFrame(
        {
            "date": [date, date],
            "order_book_id": ["A", "B"],
            "portfolio_weight": [0.75, 0.25],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": [date, date],
            "order_book_id": ["A", "B"],
            "benchmark_weight": [0.5, 0.5],
        }
    )

    active, coverage = compute_active_exposure(
        portfolio, benchmark, exposure, ["size"]
    )

    assert active.loc[date, "size"] == pytest.approx(0.5)
    assert coverage.loc[date, "portfolio_coverage"] == pytest.approx(1.0)
    assert coverage.loc[date, "benchmark_coverage"] == pytest.approx(1.0)


def test_factor_return_regression_uses_previous_day_exposure() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-03")
    codes = ["A", "B", "C", "D"]
    exposure = pd.DataFrame(
        {
            "date": [first] * 4 + [second] * 4,
            "order_book_id": codes + codes,
            "factor": [1.0, 2.0, 3.0, 4.0] + [10.0, 20.0, 30.0, 40.0],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [second] * 4,
            "order_book_id": codes,
            "return": [0.01, 0.02, 0.03, 0.04],
        }
    )
    config = AttributionConfig(
        minimum_cross_section=2,
        ridge_alpha=0.0,
        return_winsor_lower=0.0,
        return_winsor_upper=1.0,
    )

    factor_return, diagnostics = estimate_factor_returns(
        exposure, returns, ["factor"], config
    )

    assert factor_return.loc[second, "factor"] == pytest.approx(0.01)
    assert diagnostics.loc[0, "observations"] == 4
