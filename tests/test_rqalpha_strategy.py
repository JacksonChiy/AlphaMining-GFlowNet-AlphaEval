import pandas as pd
import pytest

from rqalpha_strategy.run_backtest import build_config, load_backtest_settings
from rqalpha_strategy.strategy import (
    build_smoothed_scores,
    normalize_order_book_id,
    parse_smoothing_weights,
    select_turnover_controlled_targets,
)


def test_order_book_id_mapping() -> None:
    assert normalize_order_book_id("600000.SH") == "600000.XSHG"
    assert normalize_order_book_id("000001") == "000001.XSHE"
    assert normalize_order_book_id("430001.BJ") == "430001.XBSE"


def test_stock_only_backtest_disables_unused_instrument_modules(tmp_path) -> None:
    predictions = tmp_path / "prediction_score.csv"
    pd.DataFrame({"signal_date": ["2023-01-03", "2023-01-04"]}).to_csv(
        predictions, index=False
    )

    config = build_config(predictions, tmp_path / "bundle", tmp_path / "report")

    assert config["base"]["rqdatac_uri"] == "disabled"
    assert config["base"]["auto_update_bundle"] is False
    assert config["base"]["capital_gain_tax_rate"] == 0.0
    transaction_cost = config["mod"]["sys_transaction_cost"]
    assert transaction_cost["stock_min_commission"] == 5
    assert "cn_stock_min_commission" not in transaction_cost
    for module in ("option", "fund", "convertible", "spot", "rqfactor"):
        assert config["mod"][module]["enabled"] is False


def test_backtest_settings_are_loaded_from_yaml_and_cli_can_override(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
backtest:
  initial_cash: 10000000
  benchmark: 000852.XSHG
  top_n: 20
  rebalance_days: 10
  slippage: 0.0015
  cash_buffer: 0.97
  rank_smoothing_weights: [5, 3, 2]
  hold_buffer_rank: 50
  max_replacement_ratio: 0.2
  min_holding_days: 8
""".strip(),
        encoding="utf-8",
    )

    settings = load_backtest_settings(config_path, top_n=25)

    expected = {
        "initial_cash": 10_000_000.0,
        "benchmark": "000852.XSHG",
        "top_n": 25,
        "rebalance_days": 10,
        "slippage": 0.0015,
        "cash_buffer": 0.97,
        "rank_smoothing_weights": [0.5, 0.3, 0.2],
        "hold_buffer_rank": 50,
        "max_replacement_ratio": 0.2,
        "min_holding_days": 8,
    }
    assert {key: settings[key] for key in expected} == expected
    assert settings["stock_commission_multiplier"] == 1.0
    assert settings["pit_tax"] is True
    assert settings["portfolio_mode"] == "equal_weight"


def test_parse_smoothing_weights_normalizes_and_validates() -> None:
    assert parse_smoothing_weights("5,3,2") == pytest.approx((0.5, 0.3, 0.2))
    assert parse_smoothing_weights(None) == (0.5, 0.3, 0.2)
    for invalid in ("0,0", "1,-1", "not-a-number"):
        with pytest.raises(ValueError):
            parse_smoothing_weights(invalid)


def test_rank_smoothing_combines_current_and_historical_cross_section() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-02"] * 3 + ["2024-01-03"] * 3
            ).date,
            "code": ["A", "B", "C", "A", "B", "C"],
            "prediction_score": [3.0, 2.0, 1.0, 1.0, 3.0, 2.0],
        }
    )

    smoothed = build_smoothed_scores(scores, weights=(0.5, 0.5))
    second = smoothed[pd.Timestamp("2024-01-03").date()]

    assert second["code"].tolist() == ["B", "A", "C"]
    assert second.set_index("code")["smoothed_rank_score"].to_dict() == pytest.approx(
        {"A": 2 / 3, "B": 5 / 6, "C": 1 / 2}
    )


def test_rank_smoothing_uses_current_universe_and_reweights_missing_history() -> None:
    scores = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ).date,
            "code": ["A", "B", "B", "C"],
            "prediction_score": [2.0, 1.0, 1.0, 2.0],
        }
    )

    second = build_smoothed_scores(scores, weights=(0.5, 0.5))[
        pd.Timestamp("2024-01-03").date()
    ]

    assert set(second["code"]) == {"B", "C"}
    assert second.set_index("code").loc["C", "smoothed_rank_score"] == pytest.approx(1.0)


def test_holding_buffer_retains_existing_names_outside_top_n() -> None:
    targets = select_turnover_controlled_targets(
        ranked_codes=["X", "A", "B", "Y"],
        current_holdings=["A", "B"],
        holding_days={"A": 20, "B": 20},
        top_n=2,
        hold_buffer_rank=3,
        max_replacement_ratio=1.0,
        min_holding_days=10,
    )

    assert targets == ["A", "B"]


def test_replacement_cap_limits_names_changed_per_rebalance() -> None:
    current = ["A", "B", "C", "D"]
    targets = select_turnover_controlled_targets(
        ranked_codes=["X", "Y", "Z", "W", "A", "B", "C", "D"],
        current_holdings=current,
        holding_days={code: 20 for code in current},
        top_n=4,
        hold_buffer_rank=4,
        max_replacement_ratio=0.25,
        min_holding_days=10,
    )

    assert len(set(current) - set(targets)) == 1
    assert len(set(targets) - set(current)) == 1


def test_minimum_holding_days_protects_young_position() -> None:
    common = {
        "ranked_codes": ["X", "Y", "A", "Z"],
        "current_holdings": ["A", "Z"],
        "top_n": 2,
        "hold_buffer_rank": 3,
        "max_replacement_ratio": 1.0,
        "min_holding_days": 10,
    }

    young = select_turnover_controlled_targets(holding_days={"A": 20, "Z": 9}, **common)
    mature = select_turnover_controlled_targets(holding_days={"A": 20, "Z": 10}, **common)

    assert "Z" in young
    assert "Z" not in mature
