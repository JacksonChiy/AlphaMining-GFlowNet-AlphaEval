from rqalpha_strategy.strategy import normalize_order_book_id


def test_order_book_id_mapping() -> None:
    assert normalize_order_book_id("600000.SH") == "600000.XSHG"
    assert normalize_order_book_id("000001") == "000001.XSHE"
    assert normalize_order_book_id("430001.BJ") == "430001.XBSE"

