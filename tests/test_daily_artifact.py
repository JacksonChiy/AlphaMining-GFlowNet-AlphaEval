from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data_loader.daily_artifact import save_daily_price_artifact
from src.data_loader.minute_memmap import build_ddb_minute_time_filter


def test_save_daily_price_artifact_is_independent_and_auditable(
    daily_prices: pd.DataFrame, tmp_path
) -> None:
    daily_prices = daily_prices.assign(
        amount=daily_prices["volume"] * daily_prices["vwap"]
    )
    output = tmp_path / "postprocess" / "daily_price.pkl"
    result = save_daily_price_artifact(
        daily_prices,
        output,
        source="test",
        minute_grid=("09:25:00", "09:31:00"),
    )

    restored = pd.read_pickle(result)
    metadata = json.loads(
        output.with_suffix(".pkl.metadata.json").read_text(encoding="utf-8")
    )
    assert result == output
    assert restored[["date", "code"]].duplicated().sum() == 0
    assert metadata["rows"] == len(daily_prices)
    assert metadata["dates"] == daily_prices["date"].nunique()
    assert metadata["minute_grid"] == ["09:25:00", "09:31:00"]
    assert len(metadata["sha256"]) == 64


def test_save_daily_price_artifact_rejects_duplicate_keys(
    daily_prices: pd.DataFrame, tmp_path
) -> None:
    daily_prices = daily_prices.assign(
        amount=daily_prices["volume"] * daily_prices["vwap"]
    )
    duplicated = pd.concat([daily_prices, daily_prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate date/code"):
        save_daily_price_artifact(
            duplicated, tmp_path / "daily_price.pkl", source="test"
        )


def test_build_ddb_minute_time_filter_matches_ppu_grid() -> None:
    result = build_ddb_minute_time_filter(
        (("09:31:00", "11:30:00"), ("13:01:00", "15:00:00")),
        ("09:25:00",),
    )
    assert "minute(time) = 09:25m" in result
    assert "minute(time) >= 09:31m" in result
    assert "minute(time) <= 15:00m" in result
