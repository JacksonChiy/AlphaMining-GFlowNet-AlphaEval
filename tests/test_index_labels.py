from __future__ import annotations

import numpy as np
import pandas as pd

from src.index_enhancement.labels import build_forward_labels, build_index_labels


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=7)
    rows = []
    closes = {"000001.XSHE": [10, 11, 12, 13, 14, 15, 16], "600000.XSHG": [20, 20, 22, 24, 26, 28, 30]}
    for code, values in closes.items():
        for date, close in zip(dates, values):
            rows.append({
                "date": date,
                "order_book_id": code,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100.0,
                "total_turnover": close * 100.0,
                "limit_up": close * 1.1,
                "limit_down": close * 0.9,
            })
    return pd.DataFrame(rows)


def test_forward_label_is_tplus5_over_tplus1() -> None:
    result = build_forward_labels(_prices(), horizon=5)
    first = result[(result["date"] == pd.Timestamp("2024-01-02")) & (result["code"] == "000001.XSHE")].iloc[0]

    assert first["entry_date"] == pd.Timestamp("2024-01-03")
    assert first["exit_date"] == pd.Timestamp("2024-01-09")
    assert np.isclose(first["target_raw_return"], 15 / 11 - 1)
    assert bool(first["tradable"])


def test_real_limit_and_suspension_fields_control_tradability() -> None:
    price = _prices()
    entry = (price["date"] == pd.Timestamp("2024-01-03")) & (price["order_book_id"] == "000001.XSHE")
    price.loc[entry, "limit_up"] = price.loc[entry, "close"]
    exit_ = (price["date"] == pd.Timestamp("2024-01-09")) & (price["order_book_id"] == "600000.XSHG")
    price.loc[exit_, ["volume", "total_turnover"]] = 0.0

    result = build_forward_labels(price, horizon=5)
    day = result[result["date"] == pd.Timestamp("2024-01-02")].set_index("code")

    assert bool(day.loc["000001.XSHE", "entry_at_limit_up"])
    assert not bool(day.loc["000001.XSHE", "entry_buyable"])
    assert bool(day.loc["600000.XSHG", "exit_suspended"])
    assert not bool(day.loc["600000.XSHG", "tradable"])


def test_processed_pickle_without_limits_has_explicit_degraded_mode() -> None:
    processed = _prices().rename(columns={"order_book_id": "code", "total_turnover": "amount"}).drop(
        columns=["limit_up", "limit_down"]
    )
    result = build_forward_labels(processed, horizon=5)
    first = result.iloc[0]

    assert not bool(first["entry_limit_data_available"])
    assert not bool(first["exit_limit_data_available"])
    assert bool(first["tradable"])


def test_missing_calendar_row_does_not_shift_to_next_stock_quote() -> None:
    price = _prices()
    missing_entry = (price["date"] == pd.Timestamp("2024-01-03")) & (price["order_book_id"] == "000001.XSHE")
    result = build_forward_labels(price.loc[~missing_entry], horizon=5)
    first = result[(result["date"] == pd.Timestamp("2024-01-02")) & (result["code"] == "000001.XSHE")].iloc[0]

    assert pd.isna(first["entry_close"])
    assert pd.isna(first["target_raw_return"])
    assert not bool(first["tradable"])


def test_index_labels_use_point_in_time_weights_and_coverage() -> None:
    price = _prices()
    signal_date = pd.Timestamp("2024-01-02")
    components = pd.DataFrame({
        "date": [signal_date, signal_date],
        "index_key": ["csi300", "csi300"],
        "code": ["000001.XSHE", "600000.XSHG"],
    })
    weights = pd.DataFrame({
        "date": [signal_date, signal_date],
        "index_key": ["csi300", "csi300"],
        "order_book_id": ["000001.XSHE", "600000.XSHG"],
        "weight": [30.0, 70.0],
    })

    result = build_index_labels(
        price, components, weights, "csi300", horizon=5, min_weight_coverage=0.95
    ).set_index("code")
    ret_a = 15 / 11 - 1
    ret_b = 28 / 20 - 1
    benchmark = 0.3 * ret_a + 0.7 * ret_b

    assert np.isclose(result.loc["000001.XSHE", "benchmark_weight"], 0.3)
    assert np.isclose(result.loc["000001.XSHE", "benchmark_return"], benchmark)
    assert np.isclose(result.loc["000001.XSHE", "target_excess_return"], ret_a - benchmark)
    assert result.loc["600000.XSHG", "target_cross_sectional_rank"] == 1.0
    assert result.loc["000001.XSHE", "target_cross_sectional_rank"] == 0.5
    assert np.isclose(result.loc["000001.XSHE", "weight_coverage"], 1.0)


def test_low_price_coverage_invalidates_benchmark_and_targets() -> None:
    price = _prices()
    price = price[price["order_book_id"] != "600000.XSHG"]
    signal_date = pd.Timestamp("2024-01-02")
    components = pd.DataFrame({
        "date": [signal_date, signal_date],
        "index_key": ["csi300", "csi300"],
        "code": ["000001.XSHE", "600000.XSHG"],
    })
    weights = pd.DataFrame({
        "date": [signal_date, signal_date],
        "index_key": ["csi300", "csi300"],
        "code": ["000001.XSHE", "600000.XSHG"],
        "benchmark_weight": [0.3, 0.7],
    })

    result = build_index_labels(
        price, components, weights, "csi300", horizon=5, min_weight_coverage=0.95
    )

    assert np.allclose(result["weight_coverage"], 0.3)
    assert result["benchmark_return"].isna().all()
    assert result["target_excess_return"].isna().all()
