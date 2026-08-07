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
from src.data_loader.minute_memmap import (
    DolphinDBMinuteMemMapBuilder,
    MEMMAP_CHANNELS,
    MinuteMemMapConfig,
    MinuteMemMapStore,
)
from src.data_loader.minute_quality_audit import (
    DolphinDBMinuteQualityAuditor,
    MinuteQualityAuditConfig,
)
from src.gflownet.memmap_reward import (
    PartialMinuteBlockCache,
    PersistentMinuteBlockCache,
    execute_memmap_blocks,
    pack_nodes_by_channel_dependency,
)
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
        if "ALPHAMINING_QUALITY_" in script:
            dates = re.findall(r"date [<>]= (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 2
            start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
            selected = self.frame[self.frame["date"].between(start, end)].copy()
            if "ALPHAMINING_QUALITY_TIME_V1" in script:
                return selected.groupby(["date", "time"], observed=True).size().rename(
                    "rowCount"
                ).reset_index()
            if "ALPHAMINING_QUALITY_VALUES_V1" in script:
                return pd.DataFrame({
                    "date": sorted(selected["date"].drop_duplicates()),
                    "invalidKeyRows": 0,
                    "nullValueRows": 0,
                    "nonPositivePriceRows": 0,
                    "invalidOhlcRows": 0,
                    "negativeActivityRows": 0,
                })
            if "ALPHAMINING_QUALITY_DUPLICATES_V1" in script:
                counts = selected.groupby(
                    ["date", "sym", "time"], observed=True
                ).size().rename("duplicateCount").reset_index()
                return counts.loc[counts["duplicateCount"] > 1]
            if "ALPHAMINING_QUALITY_SYMBOL_V1" in script:
                return selected.groupby(["date", "sym"], observed=True).agg(
                    rowCount=("time", "size"),
                    firstTime=("time", "min"),
                    lastTime=("time", "max"),
                ).reset_index()
            raise AssertionError(f"Unknown quality audit query: {script}")
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
        if script.startswith("select distinct time as minuteTime"):
            dates = re.findall(r"date = (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 1
            date = pd.Timestamp(dates[0].replace(".", "-"))
            values = self.frame.loc[self.frame["date"] == date, "time"]
            return pd.DataFrame({"minuteTime": sorted(values.drop_duplicates())})
        if "ALPHAMINING_MINUTE_PUSHDOWN_V1" in script:
            dates = re.findall(r"date [<>]= (\d{4}\.\d{2}\.\d{2})", script)
            assert len(dates) == 2
            start, end = (pd.Timestamp(value.replace(".", "-")) for value in dates)
            selected = self.frame[self.frame["date"].between(start, end)]
            alias = re.search(r"avg\(am_vec_000_0\) as (ddb_[a-f0-9]+)", script)
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
    assert "am_source = select" in compiled.script
    assert "am_vectors = select" in compiled.script
    assert "__ddb" not in compiled.script
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


def test_ddb_to_local_memmap_build_read_and_persistent_block_cache(
    tmp_path, capsys
) -> None:
    base_source = _source_minutes()
    excluded = base_source.groupby(["date", "sym"], observed=True).head(1).copy()
    excluded["time"] = dt.time(9, 29)
    session = FakeDolphinDBSession(
        pd.concat([base_source, excluded], ignore_index=True)
    )
    loader = DolphinDBMinuteLoader(_config(tmp_path), session)
    memmap_config = MinuteMemMapConfig(
        root=tmp_path / "minute_memmap",
        block_cache_dir=tmp_path / "block_cache",
        expected_minutes=3,
        minute_sessions=(("09:30:00", "09:32:00"),),
        minute_extra_times=(),
        stock_tile_size=1,
        build_workers=2,
        flush_every_days=2,
    )
    worker_sessions: list[FakeDolphinDBSession] = []

    def loader_factory() -> DolphinDBMinuteLoader:
        worker_session = FakeDolphinDBSession(session.frame.copy())
        worker_sessions.append(worker_session)
        return DolphinDBMinuteLoader(_config(tmp_path), worker_session)

    manifest_path = DolphinDBMinuteMemMapBuilder(
        loader, memmap_config, loader_factory=loader_factory
    ).build()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert (memmap_config.root / "field_audit.json").exists()
    assert manifest["n_minutes"] == 3
    assert manifest["n_stocks"] == 2
    assert set(manifest["channels"]) == set(MEMMAP_CHANNELS)
    assert not list((tmp_path / "minute_memmap").rglob("minute_*.pkl"))
    grid_audit = json.loads(
        (memmap_config.root / "minute_grid_audit.json").read_text(encoding="utf-8")
    )
    assert grid_audit["sample_dates"][0]["excluded_source_times"] == ["09:29:00"]
    daily_scripts = [script for script in session.scripts if "select first(open)" in script]
    assert daily_scripts
    assert all("minute(time)" in script for script in daily_scripts)
    assert worker_sessions
    assert all(worker.closed for worker in worker_sessions)
    assert sum(
        "select sym, date, time" in script
        for worker in worker_sessions for script in worker.scripts
    ) == 2

    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    session.scripts.clear()
    DolphinDBMinuteMemMapBuilder(
        loader, memmap_config, loader_factory=loader_factory
    ).build()
    assert not [script for script in session.scripts if "select first(open)" in script]

    store = MinuteMemMapStore(memmap_config)
    frames = [frame for _, _, frame in store.iter_frames("2024-01-02", "2024-01-03")]
    assert len(frames) == 4
    assert all(frame.attrs.get("minute_features_ready") for frame in frames)
    expression = minute_expression_from_tokens(["r_mean", "close"])
    cache = PersistentMinuteBlockCache(store, memmap_config.block_cache_dir)
    blocks = execute_memmap_blocks(
        store, list(expression.block_nodes()), "2024-01-02", "2024-01-03", cache
    )
    performance_log = capsys.readouterr().out
    assert "[MemMapReward] stage_complete stage=partial_cache_scan" in performance_log
    assert "[MemMapReward] stage_complete stage=task_execution" in performance_log
    assert "worker_read_sum_seconds=" in performance_log
    assert "worker_compute_sum_seconds=" in performance_log
    assert "[MemMapReward] stage_summary" in performance_log
    assert "[MemMapBlockPipeline] stage_summary" in performance_log
    actual = expression.execute_from_blocks(blocks).sort_index()
    expected = expression.execute(normalize_dolphindb_minutes(base_source)).sort_index()
    assert np.allclose(actual, expected)
    assert cache.writes == 1
    expression_dirs = list(
        (memmap_config.block_cache_dir / "partials_v2").glob("*/*")
    )
    assert len(expression_dirs) == 1
    assert {path.name for path in expression_dirs[0].iterdir()} == {
        "values.npy", "completed.npy", "metadata.json",
    }

    second_cache = PersistentMinuteBlockCache(store, memmap_config.block_cache_dir)
    cached_blocks = execute_memmap_blocks(
        store, list(expression.block_nodes()), "2024-01-02", "2024-01-03", second_cache
    )
    assert np.allclose(cached_blocks["r_mean(close)"].sort_index(), expected)
    assert second_cache.disk_hits == 1

    empty_expression = minute_expression_from_tokens(
        ["r_kurt", "m_head", "W5", "m_ret", "hl_pct"]
    )
    empty_cache = PersistentMinuteBlockCache(
        store, memmap_config.block_cache_dir
    )
    empty_blocks = execute_memmap_blocks(
        store,
        list(empty_expression.block_nodes()),
        "2024-01-02",
        "2024-01-03",
        empty_cache,
    )
    empty_key = "r_kurt(m_head(m_ret(hl_pct),5))"
    assert empty_blocks[empty_key].empty
    empty_log = capsys.readouterr().out
    assert f"empty_block_cached expression={empty_key}" in empty_log
    assert "action=reward_floor" in empty_log

    reused_empty_cache = PersistentMinuteBlockCache(
        store, memmap_config.block_cache_dir
    )
    reused_empty = execute_memmap_blocks(
        store,
        list(empty_expression.block_nodes()),
        "2024-01-02",
        "2024-01-03",
        reused_empty_cache,
    )
    assert reused_empty[empty_key].empty
    assert reused_empty_cache.disk_hits == 1

    parallel_config = replace(
        memmap_config,
        workers=2,
        block_cache_dir=tmp_path / "parallel_block_cache",
    )
    parallel_store = MinuteMemMapStore(parallel_config)
    parallel_cache = PersistentMinuteBlockCache(
        parallel_store, parallel_config.block_cache_dir
    )
    parallel_blocks = execute_memmap_blocks(
        parallel_store,
        list(expression.block_nodes()),
        "2024-01-02",
        "2024-01-03",
        parallel_cache,
    )
    assert np.allclose(parallel_blocks["r_mean(close)"].sort_index(), expected)


def test_consolidated_partial_cache_migrates_legacy_and_survives_rechunking(
    tmp_path,
) -> None:
    source = _source_minutes()
    loader = DolphinDBMinuteLoader(
        _config(tmp_path), FakeDolphinDBSession(source)
    )
    config = MinuteMemMapConfig(
        root=tmp_path / "minute_memmap",
        block_cache_dir=tmp_path / "block_cache",
        expected_minutes=3,
        minute_sessions=(("09:30:00", "09:32:00"),),
        minute_extra_times=(),
        stock_tile_size=1,
    )
    DolphinDBMinuteMemMapBuilder(loader, config).build()
    store = MinuteMemMapStore(config)
    expression = "r_mean(close)"
    cache = PartialMinuteBlockCache(
        store, config.block_cache_dir, "2024-01-02", "2024-01-03"
    )
    old_specs = store.chunk_specs("2024-01-02", "2024-01-03")
    assert len(old_specs) == 2
    expected = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    for index, spec in enumerate(old_specs):
        legacy_path = cache.legacy_path(expression, spec)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(legacy_path, expected[:, index:index + 1], allow_pickle=False)
        migrated = cache.load(expression, spec)
        assert np.allclose(migrated, expected[:, index:index + 1])
    cache.commit()
    assert cache.migrated_legacy_parts == 2
    assert len(list(cache.expression_dir(expression).iterdir())) == 3

    rechunked_config = replace(config, stock_tile_size=2, reward_chunk_days=1)
    rechunked_store = MinuteMemMapStore(rechunked_config)
    rechunked_cache = PartialMinuteBlockCache(
        rechunked_store,
        rechunked_config.block_cache_dir,
        "2024-01-02",
        "2024-01-03",
    )
    new_specs = rechunked_store.chunk_specs("2024-01-02", "2024-01-03")
    assert len(new_specs) == 2
    assert np.allclose(rechunked_cache.load(expression, new_specs[0]), expected[:1])
    assert np.allclose(rechunked_cache.load(expression, new_specs[1]), expected[1:])


def test_channel_dependency_packing_groups_compatible_blocks() -> None:
    nodes = [
        minute_expression_from_tokens(tokens).block_nodes()[0]
        for tokens in (
            ["r_mean", "close"],
            ["r_corr", "ret", "vol"],
            ["r_std", "close"],
            ["r_wmean", "ret", "vol"],
            ["r_mean", "amount"],
        )
    ]
    batches = pack_nodes_by_channel_dependency(nodes, max_blocks=2, enabled=True)
    rendered = [{node.render() for node in batch} for batch in batches]
    assert {"r_mean(close)", "r_std(close)"} in rendered
    assert {"r_corr(ret,vol)", "r_wmean(ret,vol)"} in rendered
    assert {"r_mean(amount)"} in rendered
    ungrouped = pack_nodes_by_channel_dependency(nodes, max_blocks=2, enabled=False)
    assert [node.render() for node in ungrouped[0]] == [
        "r_mean(close)", "r_corr(ret,vol)",
    ]


def test_minute_quality_audit_reports_extra_time_and_duplicate_key(tmp_path) -> None:
    source = _source_minutes()
    extra = source.iloc[[0]].copy()
    extra["time"] = dt.time(9, 29)
    duplicate = source.iloc[[1]].copy()
    source = pd.concat([source, extra, duplicate], ignore_index=True)
    session = FakeDolphinDBSession(source)
    loader = DolphinDBMinuteLoader(_config(tmp_path), session)
    audit_config = MinuteQualityAuditConfig(
        output_dir=tmp_path / "quality",
        expected_minutes=3,
        sessions=(("09:30:00", "09:32:00"),),
        extra_times=(),
        chunk_days=1,
        scope="full",
    )
    summary_path = DolphinDBMinuteQualityAuditor(loader, audit_config).run()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dates = pd.read_csv(tmp_path / "quality" / "date_quality.csv")
    presence = pd.read_csv(tmp_path / "quality" / "time_presence.csv")
    duplicates = pd.read_csv(tmp_path / "quality" / "duplicate_keys.csv")
    assert summary["status"] == "failed"
    assert summary["counts"]["problem_dates"] == 1
    assert dates.loc[0, "extra_times"] == "09:29:00"
    assert dates.loc[0, "duplicate_extra_rows"] == 1
    assert presence.loc[presence["time"] == "09:29:00", "in_expected_session"].eq(False).all()
    assert len(duplicates) == 1
    assert (tmp_path / "quality" / "problem_symbols.csv").exists()
