from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from rqalpha_strategy.run_index_enhancement import build_backtest_command
from src.index_enhancement.builder import build_all_index_inputs, build_index_input
from src.index_enhancement.universe import (
    fetch_and_save_components,
    load_components,
    normalize_component_history,
)


def _component_frame() -> pd.DataFrame:
    history = {
        datetime(2024, 1, 2): ["600000.XSHG", "000001.XSHE"],
        datetime(2024, 1, 3): ["600000.XSHG", "000002.XSHE"],
    }
    return normalize_component_history(
        "csi300", "000300.XSHG", "沪深300", history
    )


def test_normalize_component_history_creates_point_in_time_long_table() -> None:
    frame = _component_frame()

    assert frame.columns.tolist() == [
        "date",
        "index_key",
        "index_code",
        "index_name",
        "code",
    ]
    assert len(frame) == 4
    assert set(frame.loc[frame["date"] == pd.Timestamp("2024-01-03"), "code"]) == {
        "600000.XSHG",
        "000002.XSHE",
    }


def test_component_download_is_persisted_once_and_reused(tmp_path) -> None:
    calls = []

    def provider(index_code: str, start_date: str, end_date: str):
        calls.append((index_code, start_date, end_date))
        return {
            datetime(2024, 1, 2): ["600000.XSHG", "000001.XSHE"],
            datetime(2024, 1, 3): ["600000.XSHG", "000002.XSHE"],
        }

    output = tmp_path / "index_components.csv.gz"
    specs = {
        "csi300": {"order_book_id": "000300.XSHG", "name": "沪深300"}
    }
    fetch_and_save_components(
        output,
        "2024-01-02",
        "2024-01-03",
        specs,
        provider=provider,
    )

    assert calls == [("000300.XSHG", "2024-01-02", "2024-01-03")]
    assert len(load_components(output)) == 4
    assert output.with_name(output.name + ".metadata.json").exists()
    with pytest.raises(FileExistsError):
        fetch_and_save_components(
            output,
            "2024-01-02",
            "2024-01-03",
            specs,
            provider=provider,
        )
    assert len(calls) == 1


def test_index_input_uses_exact_historical_membership() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "code": [
                "600000.XSHG",
                "000001.XSHE",
                "000002.XSHE",
                "600000.XSHG",
                "000001.XSHE",
                "000002.XSHE",
            ],
            "prediction_score": [0.2, 0.3, 9.0, 0.5, 9.0, 0.4],
        }
    )

    result = build_index_input(predictions, _component_frame(), "csi300")

    day_one = result[result["signal_date"] == pd.Timestamp("2024-01-02")]
    day_two = result[result["signal_date"] == pd.Timestamp("2024-01-03")]
    assert set(day_one["code"]) == {"600000.XSHG", "000001.XSHE"}
    assert set(day_two["code"]) == {"600000.XSHG", "000002.XSHE"}
    assert day_one.iloc[0]["code"] == "000001.XSHE"
    assert day_two.iloc[0]["code"] == "600000.XSHG"


def test_build_all_index_inputs_writes_separate_local_files(tmp_path) -> None:
    predictions_path = tmp_path / "prediction_score.csv"
    pd.DataFrame(
        {
            "signal_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "code": ["600000.XSHG", "000001.XSHE", "600000.XSHG", "000002.XSHE"],
            "prediction_score": [0.2, 0.3, 0.5, 0.4],
        }
    ).to_csv(predictions_path, index=False)
    component_path = tmp_path / "components.csv.gz"
    _component_frame().assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_csv(
        component_path, index=False, compression="gzip"
    )
    specs = {
        "csi300": {
            "order_book_id": "000300.XSHG",
            "name": "沪深300",
            "top_n": 1,
            "hold_buffer_rank": 2,
        }
    }

    outputs = build_all_index_inputs(
        predictions_path, component_path, tmp_path / "outputs", specs
    )

    assert outputs["csi300"].exists()
    assert (tmp_path / "outputs" / "manifest.json").exists()
    assert len(pd.read_csv(outputs["csi300"])) == 4


def test_index_backtest_command_passes_benchmark_and_universe_parameters(tmp_path) -> None:
    command = build_backtest_command(
        "csi500",
        {
            "order_book_id": "000905.XSHG",
            "top_n": 100,
            "hold_buffer_rank": 200,
        },
        tmp_path / "predictions.csv.gz",
        tmp_path / "report",
        "configs/training_config.yaml",
        "~/.rqalpha-plus/bundle",
    )

    assert command[command.index("--benchmark") + 1] == "000905.XSHG"
    assert command[command.index("--top-n") + 1] == "100"
    assert command[command.index("--hold-buffer-rank") + 1] == "200"
