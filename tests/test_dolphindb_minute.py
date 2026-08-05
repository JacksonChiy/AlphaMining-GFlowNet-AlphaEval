from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_loader.dolphindb_minute import (
    DolphinDBMinuteLoader,
    MinuteDolphinDBConfig,
    normalize_dolphindb_minutes,
)
from src.expression.minute import minute_expression_from_tokens
from src.expression.dolphindb_minute import DolphinDBMinuteCompiler
from src.gflownet.minute_factor_pool import (
    save_minute_alpha_pool_from_cache,
    save_minute_alpha_pool_from_dolphindb_stream,
)
from src.gflownet.minute_reward import DolphinDBStreamingMinuteRewardEvaluator


def _source_minutes() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for code_index, code in enumerate(("000001.SZ", "000002.SZ")):
            for minute in range(3):
                price = 10.0 + code_index + minute * 0.1
                rows.append({
                    "sym": code,
                    "date": date,
                    "time": dt.time(9, 30 + minute),
                    "open": price,
                    "high": price + 0.05,
                    "low": price - 0.05,
                    "close": price + 0.01,
                    "volume": 100 + minute,
                    "amount": (100 + minute) * price,
                    "tradeCount": 5 + minute,
                })
    return pd.DataFrame(rows)


class FakeDolphinDBSession:
    def __init__(self, frame: pd.DataFrame, bad_close_type: bool = False) -> None:
        self.frame = frame
        self.closed = False
        self.scripts: list[str] = []
        self.bad_close_type = bad_close_type

    def run(self, script: str):
        self.scripts.append(script)
        if script.startswith("schema("):
            if '"TradeDays"' in script:
                return pd.DataFrame({"name": ["tradeDate"], "typeString": ["DATE"]})
            types = {
                "sym": "SYMBOL", "date": "DATE", "time": "MINUTE",
                "open": "DOUBLE", "high": "DOUBLE", "low": "DOUBLE",
                "close": "STRING" if self.bad_close_type else "DOUBLE",
                "volume": "LONG", "amount": "DOUBLE", "tradeCount": "INT",
            }
            return pd.DataFrame({"name": list(types), "typeString": list(types.values())})
        if script.startswith("select distinct") and '"TradeDays"' in script:
            dates = re.findall(r"tradeDate [<>]= (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 2
            start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
            selected = self.frame.loc[self.frame["date"].between(start, end), "date"]
            return pd.DataFrame({"tradeDate": sorted(selected.drop_duplicates())})
        if "ALPHAMINING_MINUTE_PUSHDOWN_V1" in script:
            dates = re.findall(r"date [<>]= (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 2
            start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
            selected = self.frame[self.frame["date"].between(start, end)]
            alias = re.search(r"avg\(__ddb_vec_000_0\) as (ddb_[a-f0-9]+)", script)
            assert alias is not None
            result = selected.groupby(["date", "sym"], observed=True, sort=True)["close"].mean().reset_index()
            return result.rename(columns={"close": alias.group(1)})
        if script.startswith("select min(date)"):
            dates = re.findall(r"date [<>]= (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 2
            start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
            selected = self.frame[self.frame["date"].between(start, end)]
            return pd.DataFrame({
                "minDate": [selected["date"].min()],
                "maxDate": [selected["date"].max()],
                "rows": [len(selected)],
            })
        if "select first(open)" in script:
            dates = re.findall(r"date [<>]= (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 2
            start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
            selected = self.frame[self.frame["date"].between(start, end)].copy()
            if selected.empty:
                return pd.DataFrame()
            return selected.groupby(["date", "sym"], observed=True, sort=True).agg(
                open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"),
                amount=("amount", "sum"), trade_count=("tradeCount", "sum"),
            ).reset_index()
        dates = re.findall(r"date [<>]= (\d{4}\.\d{2}\.\d{2})", script)
        assert len(dates) == 2
        start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
        return self.frame[self.frame["date"].between(start, end)].copy()

    def close(self) -> None:
        self.closed = True


def _config(tmp_path, adjusted: bool = True) -> MinuteDolphinDBConfig:
    return MinuteDolphinDBConfig(
        database="dfs://minuteBars",
        table="minuteKline",
        start_date="2024-01-02",
        end_date="2024-01-03",
        cache_dir=tmp_path / "minute_cache",
        daily_file=tmp_path / "daily.pkl",
        chunk_days=1,
        audit_chunk_days=1,
        daily_aggregate_chunk_days=1,
        prices_are_adjusted=adjusted,
    )


def test_normalize_user_supplied_dolphindb_schema() -> None:
    normalized = normalize_dolphindb_minutes(_source_minutes())
    assert normalized.columns.tolist() == [
        "date", "datetime", "code", "open", "high", "low", "close",
        "vol", "amount", "trade_count",
    ]
    assert normalized["datetime"].dt.strftime("%H:%M").iloc[0] == "09:30"
    assert normalized["code"].nunique() == 2
    assert normalized["vol"].min() == 100


def test_field_audit_checks_names_types_dates_and_adjustment(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    audit = DolphinDBMinuteLoader(_config(tmp_path), session).audit()
    assert audit.passed
    assert audit.source_rows == 12
    assert audit.source_min_date == "2024-01-02"
    assert {item["status"] for item in audit.required_fields} == {"direct", "derived"}
    stats_scripts = [script for script in session.scripts if script.startswith("select min(date)")]
    assert stats_scripts
    assert all("where date >=" in script for script in stats_scripts)


def test_field_audit_rejects_incompatible_dolphindb_type(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes(), bad_close_type=True)
    with pytest.raises(ValueError, match="missing fields"):
        DolphinDBMinuteLoader(_config(tmp_path), session).audit()


def test_field_audit_reports_actual_date_when_requested_end_is_unavailable(tmp_path) -> None:
    config = MinuteDolphinDBConfig(
        database="dfs://minuteBars",
        table="minuteKline",
        start_date="2024-01-02",
        end_date="2024-02-29",
        cache_dir=tmp_path / "minute_cache",
        daily_file=tmp_path / "daily.pkl",
        chunk_days=20,
        prices_are_adjusted=True,
    )
    with pytest.raises(
        ValueError,
        match=r"requested_end=2024-02-29, actual_last_trade_date=2024-01-03",
    ):
        DolphinDBMinuteLoader(config, FakeDolphinDBSession(_source_minutes())).audit()


def test_unadjusted_source_is_auditable_but_cannot_extract(tmp_path) -> None:
    loader = DolphinDBMinuteLoader(
        _config(tmp_path, adjusted=False), FakeDolphinDBSession(_source_minutes())
    )
    assert loader.audit().passed is False
    with pytest.raises(ValueError, match="prices_are_adjusted=true"):
        loader.extract()
    assert (tmp_path / "minute_cache" / "field_audit.json").exists()


def test_server_side_daily_aggregation_does_not_fetch_raw_minutes(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    daily = DolphinDBMinuteLoader(_config(tmp_path), session).build_daily_in_memory()
    assert len(daily) == 4
    assert np.allclose(daily["volume"], 303)
    daily_scripts = [script for script in session.scripts if "select first(open)" in script]
    assert len(daily_scripts) == 2
    assert all("group by date, sym" in script for script in daily_scripts)
    assert all("order by date, sym, time;" in script for script in daily_scripts)


def test_streaming_reward_batch_shares_ddb_scan_and_daily_block(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    loader = DolphinDBMinuteLoader(_config(tmp_path), session)
    daily = loader.build_daily_in_memory()
    session.scripts.clear()
    evaluator = DolphinDBStreamingMinuteRewardEvaluator(
        loader,
        daily,
        start_date="2024-01-02",
        end_date="2024-01-03",
        block_cache_max_entries=8,
        horizon=2,
        min_cross_section=1,
        min_coverage=0.5,
        subexpression_cache_enabled=False,
    )
    first = minute_expression_from_tokens(["r_mean", "close"])
    second = minute_expression_from_tokens(["neg", "r_mean", "close"])
    results = evaluator.evaluate_many([first, second])
    pushdown_scripts = [script for script in session.scripts if "ALPHAMINING_MINUTE_PUSHDOWN_V1" in script]
    assert len(results) == 2
    assert len(pushdown_scripts) == 2
    assert evaluator.cache_stats()["entries"] == 1
    evaluator.evaluate_many([first, second])
    assert len([script for script in session.scripts if "ALPHAMINING_MINUTE_PUSHDOWN_V1" in script]) == 2


def test_trade_days_drive_stream_and_skip_non_trading_calendar_dates(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    loader = DolphinDBMinuteLoader(_config(tmp_path), session)
    chunks = list(loader.iter_frames("2024-01-01", "2024-01-07"))
    assert [(start.date(), end.date()) for start, end, _ in chunks] == [
        (dt.date(2024, 1, 2), dt.date(2024, 1, 2)),
        (dt.date(2024, 1, 3), dt.date(2024, 1, 3)),
    ]
    assert len([script for script in session.scripts if '"TradeDays"' in script]) == 2


def test_trade_days_can_be_loaded_from_a_separate_database(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    config = replace(_config(tmp_path), trade_days_database="dfs://calendar")
    loader = DolphinDBMinuteLoader(config, session)
    assert len(loader.load_trade_dates()) == 2
    calendar_scripts = [script for script in session.scripts if '"TradeDays"' in script]
    assert calendar_scripts
    assert all('loadTable("dfs://calendar", "TradeDays")' in script for script in calendar_scripts)


def test_ddb_compiler_pushes_supported_blocks_and_marks_complex_fallback() -> None:
    compiler = DolphinDBMinuteCompiler('loadTable("dfs://minuteBars", "minuteKline")')
    supported = minute_expression_from_tokens(["r_mean", "m_ma", "W5", "close"]).block_nodes()[0]
    unsupported = minute_expression_from_tokens(["r_slope", "close"]).block_nodes()[0]
    compiled = compiler.compile([supported], pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02"))
    assert "mavg(close, 5, 2)" in compiled.script
    assert "group by date, sym" in compiled.script
    assert "where date >= 2024.01.02, date <= 2024.01.02" in compiled.script
    assert compiler.supports(unsupported) is False


def test_streaming_factor_pool_writes_daily_csv_without_minute_pickle(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    loader = DolphinDBMinuteLoader(_config(tmp_path), session)
    daily = loader.build_daily_in_memory()
    expression = minute_expression_from_tokens(["r_mean", "close"])
    pool = [{
        "expression": expression,
        "tokens": expression.to_tokens(),
        "coverage": 1.0,
        "valid_date_coverage": 1.0,
        "reward": 0.1,
    }]
    metadata, matrix = save_minute_alpha_pool_from_dolphindb_stream(
        pool,
        loader,
        daily,
        start_date="2024-01-02",
        end_date="2024-01-03",
        metadata_path=tmp_path / "alpha_pool.csv",
        matrix_path=tmp_path / "alpha_factor_matrix.csv.gz",
    )
    assert metadata["expression"].tolist() == ["r_mean(close)"]
    assert matrix["minute_factor_001"].notna().all()
    assert (tmp_path / "alpha_factor_matrix.csv.gz").exists()
    assert not list(tmp_path.glob("minute_*.pkl"))


def test_chunked_extract_daily_aggregation_and_partitioned_factor_pool(tmp_path) -> None:
    session = FakeDolphinDBSession(_source_minutes())
    loader = DolphinDBMinuteLoader(_config(tmp_path), session)
    cache_dir, daily_path = loader.extract()
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["rows"] == 12
    assert len(manifest["files"]) == 2
    data_scripts = [
        script for script in session.scripts
        if "where date" in script and "select sym, date, time" in script
    ]
    assert data_scripts
    assert all("order by date, sym, time" in script for script in data_scripts)

    daily = pd.read_pickle(daily_path)
    assert len(daily) == 4
    assert np.allclose(daily["volume"], 303)
    expression = minute_expression_from_tokens(["r_mean", "close"])
    pool = [{
        "expression": expression,
        "tokens": expression.to_tokens(),
        "coverage": 1.0,
        "valid_date_coverage": 1.0,
        "reward": 0.1,
    }]
    metadata, matrix = save_minute_alpha_pool_from_cache(
        pool,
        cache_dir,
        daily,
        metadata_path=tmp_path / "alpha_pool.csv",
        matrix_path=tmp_path / "factor_matrix.pkl",
    )
    assert metadata["factor"].tolist() == ["minute_factor_001"]
    assert matrix["minute_factor_001"].notna().all()
