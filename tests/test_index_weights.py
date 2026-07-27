from __future__ import annotations

import json

import pandas as pd
import pytest

from src.index_enhancement.weights import (
    fetch_and_save_weights,
    load_index_weights,
    normalize_weight_history,
)


def _raw_weights() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "600000.XSHG"),
            (pd.Timestamp("2024-01-02"), "000001.XSHE"),
            (pd.Timestamp("2024-01-03"), "600000.XSHG"),
            (pd.Timestamp("2024-01-03"), "000002.XSHE"),
        ],
        names=["date", "order_book_id"],
    )
    return pd.DataFrame({"weight": [0.6001, 0.4, 0.25, 0.75]}, index=index)


def test_normalize_weight_history_normalizes_rqdata_rounding() -> None:
    result = normalize_weight_history(
        "csi300", "000300.XSHG", "沪深300", _raw_weights()
    )

    assert result.columns.tolist() == [
        "date",
        "index_key",
        "index_code",
        "index_name",
        "code",
        "benchmark_weight",
    ]
    assert result.groupby("date")["benchmark_weight"].sum().tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_fetch_weights_calls_provider_once_per_index_and_reuses_file(tmp_path) -> None:
    calls = []

    def provider(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        calls.append((index_code, start_date, end_date))
        return _raw_weights()

    output = tmp_path / "index_weights.csv.gz"
    specs = {"csi300": {"order_book_id": "000300.XSHG", "name": "沪深300"}}
    fetch_and_save_weights(
        output,
        "2024-01-02",
        "2024-01-03",
        index_specs=specs,
        provider=provider,
    )

    assert calls == [("000300.XSHG", "2024-01-02", "2024-01-03")]
    assert len(load_index_weights(output)) == 4
    metadata = json.loads(
        output.with_name(output.name + ".metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["api_calls"] == 1
    assert metadata["daily_weights_normalized"] is True
    with pytest.raises(FileExistsError):
        fetch_and_save_weights(
            output,
            "2024-01-02",
            "2024-01-03",
            index_specs=specs,
            provider=provider,
        )
    assert len(calls) == 1


def test_load_index_weights_rejects_non_unit_daily_weights(tmp_path) -> None:
    output = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "index_key": ["csi300"],
            "index_code": ["000300.XSHG"],
            "index_name": ["沪深300"],
            "code": ["600000.XSHG"],
            "benchmark_weight": [0.5],
        }
    ).to_csv(output, index=False)

    with pytest.raises(ValueError, match="sum to one"):
        load_index_weights(output)
