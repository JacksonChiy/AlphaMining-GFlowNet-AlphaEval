import pandas as pd

from rqalpha_strategy.run_backtest import build_config
from rqalpha_strategy.strategy import normalize_order_book_id


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
    for module in ("option", "fund", "convertible", "spot"):
        assert config["mod"][module]["enabled"] is False
