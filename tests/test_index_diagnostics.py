from __future__ import annotations

import pandas as pd

from src.index_enhancement.diagnostics import (
    evaluate_backtest_periods,
    evaluate_index_signal,
    evaluate_universe_coverage,
    make_t5_t1_labels,
)


def _prices() -> pd.DataFrame:
    rows = []
    for code, closes in {"000001.XSHE": [10, 11, 12, 13, 14, 22], "600000.XSHG": [20, 18, 17, 16, 15, 9]}.items():
        for offset, close in enumerate(closes):
            rows.append({"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset), "code": code, "close": close})
    return pd.DataFrame(rows)


def test_t5_t1_label_uses_entry_day_one_and_exit_day_five() -> None:
    labels = make_t5_t1_labels(_prices())
    first = labels[labels["date"] == pd.Timestamp("2024-01-01")].set_index("code")
    assert first.loc["000001.XSHE", "future_return_t5_t1"] == 1.0
    assert first.loc["600000.XSHG", "future_return_t5_t1"] == -0.5


def test_signal_diagnostics_compute_index_rank_ic_and_quantiles() -> None:
    labels = make_t5_t1_labels(_prices())
    predictions = pd.DataFrame({
        "signal_date": ["2024-01-01", "2024-01-01"],
        "code": ["000001.XSHE", "600000.XSHG"],
        "prediction_score": [1.0, -1.0],
        "index_key": ["csi300", "csi300"],
    })
    daily, annual, quantiles = evaluate_index_signal(predictions, labels, quantiles=2)
    assert daily.loc[0, "rank_ic"] == 1.0
    assert annual.loc[0, "rank_ic_mean"] == 1.0
    assert set(quantiles["quantile"].astype(int)) == {1, 2}


def test_universe_coverage_uses_point_in_time_component_count() -> None:
    predictions = pd.DataFrame({
        "signal_date": ["2024-01-01"], "code": ["000001.XSHE"],
        "prediction_score": [1.0], "index_key": ["csi300"]
    })
    components = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "index_key": ["csi300", "csi300"],
        "code": ["000001.XSHE", "600000.XSHG"],
    })
    result = evaluate_universe_coverage(predictions, components)
    assert result.loc[0, "component_count"] == 2
    assert result.loc[0, "prediction_coverage"] == 0.5


def test_backtest_periods_report_cash_drag_and_cost(tmp_path) -> None:
    pd.DataFrame({
        "date": pd.date_range("2024-01-02", periods=4),
        "cash": [100.0, 100.0, 1.0, 1.0],
        "market_value": [0.0, 0.0, 105.0, 107.0],
        "total_value": [100.0, 100.0, 106.0, 108.0],
        "unit_net_value": [1.0, 1.0, 1.06, 1.08],
        "benchmark_unit_net_value": [1.0, 1.01, 1.02, 1.03],
    }).to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame({
        "datetime": ["2024-01-04 15:00:00"],
        "last_quantity": [10], "last_price": [10.0],
        "transaction_cost": [0.2], "commission": [0.2], "tax": [0.0],
    }).to_csv(tmp_path / "trades.csv", index=False)
    annual, monthly, turnover, cash = evaluate_backtest_periods(tmp_path, "csi300")
    assert len(annual) == 1 and len(monthly) == 1
    assert turnover.loc[0, "transaction_cost"] == 0.2
    assert cash.loc[0, "initial_cash_days"] == 2
    assert cash.loc[0, "first_invested_date"] == "2024-01-04"
