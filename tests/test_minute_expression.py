from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import warnings

from src.expression.minute import (
    MASK_BINARY_OPS,
    MASK_BINARY_WINDOW_OPS,
    MASK_UNARY_OPS,
    MASK_WINDOW_OPS,
    MINUTE_BINARY_OPS,
    MINUTE_FEATURES,
    MINUTE_UNARY_OPS,
    MINUTE_WINDOW_OPS,
    REDUCE_BINARY_OPS,
    REDUCE_UNARY_OPS,
    MinuteExpression,
    minute_expression_from_tokens,
)
from src.gflownet.minute_grammar import MINUTE_ACTION_TOKENS, MinuteGrammarState, MinuteVocabulary
from src.gflownet.model import GFlowNetPolicy, PolicyConfig
from src.gflownet.numpy_minute_executor import (
    NumpyMinuteBlockExecutor,
    required_memmap_channels,
)
from src.data_loader.minute_memmap import MEMMAP_CHANNELS
from src.operators.minute import build_minute_features
from src.operators.minute import apply_reduce_binary, apply_reduce_unary


def _minute_prices() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.bdate_range("2024-01-02", periods=2)):
        for code_index, code in enumerate(("000001.XSHE", "000002.XSHE")):
            for minute in range(70):
                price = 10.0 + code_index + date_index * 0.2 + minute * 0.01
                volume = 100.0 + minute
                rows.append({
                    "date": date,
                    "datetime": date + pd.Timedelta(hours=9, minutes=30 + minute),
                    "code": code,
                    "open": price - 0.01,
                    "high": price + 0.03,
                    "low": price - 0.03,
                    "close": price,
                    "vol": volume,
                    "amount": volume * price,
                })
    return pd.DataFrame(rows)


def test_chart_27_feature_set_and_fallback_calculation() -> None:
    prepared = build_minute_features(_minute_prices())
    assert len(MINUTE_FEATURES) == 21
    assert set(MINUTE_FEATURES).issubset(prepared.columns)
    first_rows = prepared.groupby(["date", "code"], observed=True).head(1)
    assert first_rows["ret"].isna().all()
    assert np.allclose(prepared["vwap"], prepared["close"])
    assert prepared["vwap_cum"].notna().all()


def test_chart_28_to_30_operator_inventory_is_complete() -> None:
    assert len(MINUTE_UNARY_OPS + MINUTE_WINDOW_OPS + MINUTE_BINARY_OPS) == 15
    assert len(MASK_WINDOW_OPS + MASK_UNARY_OPS + MASK_BINARY_WINDOW_OPS + MASK_BINARY_OPS) == 14
    assert len(REDUCE_UNARY_OPS + REDUCE_BINARY_OPS) == 16
    assert len(MINUTE_ACTION_TOKENS) == 92


def test_all_chart_28_to_30_operator_paths_execute() -> None:
    data = _minute_prices()
    token_sets: list[list[str]] = []
    token_sets.extend([[operator, "close"] for operator in REDUCE_UNARY_OPS])
    token_sets.extend([[operator, "close", "vol"] for operator in REDUCE_BINARY_OPS])
    token_sets.extend([["r_mean", operator, "close"] for operator in MINUTE_UNARY_OPS])
    token_sets.extend([["r_mean", operator, "W5", "close"] for operator in MINUTE_WINDOW_OPS])
    token_sets.extend([["r_mean", operator, "close", "vol"] for operator in MINUTE_BINARY_OPS])
    token_sets.extend([["r_mean", operator, "W5", "close"] for operator in MASK_WINDOW_OPS])
    token_sets.extend([["r_mean", operator, "close"] for operator in MASK_UNARY_OPS])
    token_sets.extend(
        [["r_mean", operator, "W5", "close", "vol"] for operator in MASK_BINARY_WINDOW_OPS]
    )
    token_sets.extend([["r_mean", operator, "close", "vol"] for operator in MASK_BINARY_OPS])
    for tokens in token_sets:
        result = minute_expression_from_tokens(tokens).execute(data)
        assert result.index.names == ["date", "code"]
        assert len(result) == 4


def test_numpy_memmap_executor_matches_pandas_for_all_minute_operators() -> None:
    data = _minute_prices().drop(index=[3, 78, 201]).reset_index(drop=True)
    data.loc[4, "close"] = data.loc[5, "close"]  # exercise average-rank tie fallback
    prepared = build_minute_features(data)
    dates = pd.DatetimeIndex(sorted(prepared["date"].unique()))
    stocks = np.array(sorted(prepared["code"].unique()))
    minutes = sorted(prepared["datetime"].dt.time.astype(str).unique())
    date_lookup = {value: index for index, value in enumerate(dates)}
    stock_lookup = {value: index for index, value in enumerate(stocks)}
    minute_lookup = {value: index for index, value in enumerate(minutes)}
    shape = (len(dates), len(minutes), len(stocks))
    mask = np.zeros(shape, dtype=bool)
    channels = {name: np.full(shape, np.nan, dtype=np.float32) for name in MEMMAP_CHANNELS}
    for row in prepared.itertuples():
        key = (
            date_lookup[pd.Timestamp(row.date)],
            minute_lookup[str(row.datetime.time())],
            stock_lookup[str(row.code)],
        )
        mask[key] = True
        for name in MEMMAP_CHANNELS:
            channels[name][key] = getattr(row, name)
    expected_data = prepared.copy()
    expected_data[list(MEMMAP_CHANNELS)] = expected_data[list(MEMMAP_CHANNELS)].astype(
        np.float32
    )
    previous = expected_data.groupby(["date", "code"], observed=True)["close"].shift(1)
    ratio = expected_data["close"].div(previous.where(previous.abs() > 1e-12))
    expected_data["logret"] = np.log(ratio.where(ratio > 0))
    expected_data["oc_ret"] = expected_data["close"].div(
        expected_data["open"].where(expected_data["open"].abs() > 1e-12)
    ) - 1.0
    expected_data.attrs["minute_features_ready"] = True

    token_sets: list[list[str]] = []
    token_sets.extend([[operator, "close"] for operator in REDUCE_UNARY_OPS])
    token_sets.extend([[operator, "close", "vol"] for operator in REDUCE_BINARY_OPS])
    token_sets.extend([["r_mean", operator, "close"] for operator in MINUTE_UNARY_OPS])
    token_sets.extend([["r_mean", operator, "W5", "close"] for operator in MINUTE_WINDOW_OPS])
    token_sets.extend([["r_mean", operator, "close", "vol"] for operator in MINUTE_BINARY_OPS])
    token_sets.extend([["r_mean", operator, "W5", "close"] for operator in MASK_WINDOW_OPS])
    token_sets.extend([["r_mean", operator, "close"] for operator in MASK_UNARY_OPS])
    token_sets.extend(
        [["r_mean", operator, "W5", "close", "vol"] for operator in MASK_BINARY_WINDOW_OPS]
    )
    token_sets.extend([["r_mean", operator, "close", "vol"] for operator in MASK_BINARY_OPS])
    token_sets.extend([
        ["r_mean", "m_logret", "close"],
        ["r_mean", "m_abs", "logret"],
        ["r_mean", "m_add", "oc_ret", "ret"],
    ])
    full_index = pd.MultiIndex.from_product([dates, stocks], names=["date", "code"])
    for tokens in token_sets:
        expression = minute_expression_from_tokens(tokens)
        node = expression.block_nodes()[0]
        selected = required_memmap_channels([node])
        actual = NumpyMinuteBlockExecutor(
            mask, {name: channels[name] for name in selected}
        ).execute([node])[node.render()].reshape(-1)
        expected = expression.execute_block(node, expected_data).reindex(full_index).to_numpy()
        assert np.allclose(actual, expected, rtol=2e-4, atol=2e-5, equal_nan=True), str(expression)


def test_vectorized_reductions_are_stable_for_sparse_near_constant_groups() -> None:
    days, minutes, stocks = 2, 70, 3
    minute_axis = np.arange(minutes, dtype=np.float64)
    close = np.empty((days, minutes, stocks), dtype=np.float32)
    volume = np.empty_like(close)
    for day in range(days):
        for stock in range(stocks):
            close[day, :, stock] = (
                10_000.0 + stock * 100 + day + np.sin(minute_axis / 9.0) * 0.1
            )
            volume[day, :, stock] = 1_000.0 + minute_axis * (stock + 1)
    mask = np.ones_like(close, dtype=bool)
    mask[1, 5:10, 1] = False
    mask[:, :, 2] = False
    close[0, 12, 0] = np.nan
    tokens = [
        ["r_skew", "close"], ["r_kurt", "close"], ["r_slope", "close"],
        ["r_rsquare", "close"], ["r_argmax", "close"],
        ["r_corr", "close", "vol"], ["r_cov", "close", "vol"],
        ["r_wmean", "close", "vol"],
    ]
    nodes = [minute_expression_from_tokens(value).block_nodes()[0] for value in tokens]
    actual = NumpyMinuteBlockExecutor(
        mask, {"close": close, "vol": volume}
    ).execute(nodes)

    dates = pd.bdate_range("2024-01-02", periods=days)
    codes = [f"S{index}" for index in range(stocks)]
    rows = []
    for day in range(days):
        for stock in range(stocks):
            for minute in range(minutes):
                if mask[day, minute, stock]:
                    rows.append({
                        "date": dates[day], "code": codes[stock],
                        "close": close[day, minute, stock],
                        "vol": volume[day, minute, stock],
                    })
    frame = pd.DataFrame(rows)
    frame.attrs["minute_features_ready"] = True
    full_index = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    for node in nodes:
        expected = MinuteExpression(node).execute_block(node, frame).reindex(full_index)
        assert np.allclose(
            actual[node.render()].reshape(-1), expected.to_numpy(),
            rtol=2e-5, atol=2e-6, equal_nan=True,
        ), node.render()


def test_minute_delay_resets_at_each_trading_day() -> None:
    data = _minute_prices()
    expression = minute_expression_from_tokens(
        ["r_mean", "m_head", "W5", "m_delay", "W5", "close"]
    )
    values = expression.execute(data)
    assert values.isna().all()


def test_mask_and_reduction_semantics() -> None:
    data = _minute_prices()
    head = minute_expression_from_tokens(["r_mean", "m_head", "W5", "close"]).execute(data)
    expected = data.groupby(["date", "code"], observed=True)["close"].head(5)
    expected = expected.groupby([data.loc[expected.index, "date"], data.loc[expected.index, "code"]]).mean()
    assert np.allclose(head.sort_index(), expected.sort_index())
    argmax = minute_expression_from_tokens(["r_argmax", "close"]).execute(data)
    assert np.allclose(argmax, 1.0)


def test_minute_grammar_round_trip_and_policy_output_shape() -> None:
    state = MinuteGrammarState(max_depth=6, max_nodes=18)
    for token in ("r_mean", "m_at_top", "W5", "ret", "signed_amt"):
        state = state.step(token)
    assert state.terminal
    assert str(state.expression) == "r_mean(m_at_top(ret,signed_amt,5))"
    vocabulary = MinuteVocabulary()
    model = GFlowNetPolicy(
        PolicyConfig(hidden_dim=32, num_layers=1, num_heads=4, max_sequence_length=16),
        vocabulary,
    )
    token_ids = torch.tensor([[vocabulary.bos_id]], dtype=torch.long)
    logits = model(token_ids, torch.zeros((1, 3)))
    assert logits.shape == (1, len(MINUTE_ACTION_TOKENS))


def test_report_daily_tree_executes_after_intraday_reduction() -> None:
    data = _minute_prices()
    expression = minute_expression_from_tokens(
        ["neg", "ts_mean", "W5", "r_corr", "ret", "signed_amt"]
    )
    assert str(expression) == "neg(ts_mean(r_corr(ret,signed_amt),5))"
    assert [node.render() for node in expression.block_nodes()] == [
        "r_corr(ret,signed_amt)"
    ]
    result = expression.execute(data)
    assert result.index.names == ["date", "code"]
    assert len(result) == 4


def test_daily_array_execution_matches_grouped_pandas_for_nested_sparse_blocks() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    full_index = pd.MultiIndex.from_product(
        [dates, ["A", "B", "C"]], names=["date", "code"]
    )
    left = pd.Series(np.arange(len(full_index), dtype=float), index=full_index)
    # Preserve both a genuinely absent row and an observed NaN: they have
    # different rolling-window semantics.
    left = left.drop((dates[2], "B"))
    left.loc[(dates[4], "A")] = np.nan
    expression = minute_expression_from_tokens(
        ["cs_zscore", "ts_mean", "W5", "neg", "r_mean", "close"]
    )

    frame = left.rename("value").reset_index()
    frame = frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    rolling = frame["value"].groupby(frame["code"], observed=True, sort=False).rolling(
        5, min_periods=5
    ).mean().reset_index(level=0, drop=True)
    frame["value"] = -rolling
    frame = frame.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    grouped = frame["value"].groupby(frame["date"], observed=True, sort=False)
    expected = (frame["value"] - grouped.transform("mean")) / grouped.transform("std")
    expected_index = pd.MultiIndex.from_frame(frame[["date", "code"]])
    expected = pd.Series(expected.to_numpy(), index=expected_index).sort_index()

    actual = expression.execute_from_blocks({"r_mean(close)": left})
    pd.testing.assert_series_equal(
        actual,
        expected.rename(str(expression)),
        check_dtype=False,
        check_names=True,
    )


def test_daily_binary_array_execution_keeps_pandas_union_alignment() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    left_index = pd.MultiIndex.from_tuples(
        [(dates[0], "A"), (dates[1], "A")], names=["date", "code"]
    )
    right_index = pd.MultiIndex.from_tuples(
        [(dates[0], "A"), (dates[0], "B")], names=["date", "code"]
    )
    left = pd.Series([1.0, 2.0], index=left_index)
    right = pd.Series([10.0, 20.0], index=right_index)
    expression = minute_expression_from_tokens(
        ["add", "r_mean", "close", "r_sum", "vol"]
    )

    actual = expression.execute_from_blocks(
        {"r_mean(close)": left, "r_sum(vol)": right}
    )
    expected = (left + right).sort_index().rename(str(expression))
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)


def test_binary_reductions_handle_constant_inputs_without_warnings() -> None:
    data = _minute_prices()
    constant = pd.Series(1.0, index=data.index)
    weights = pd.Series(2.0, index=data.index)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", FutureWarning)
        correlation = apply_reduce_binary("r_corr", constant, constant, data)
        covariance = apply_reduce_binary("r_cov", constant, constant, data)
        weighted = apply_reduce_binary("r_wmean", constant, weights, data)
    assert correlation.isna().all()
    assert np.allclose(covariance, 0.0)
    assert np.allclose(weighted, 1.0)


def test_complex_numpy_reductions_match_pandas_group_semantics() -> None:
    data = _minute_prices()
    values = data["close"] + np.sin(np.arange(len(data), dtype=float) / 7.0)
    expected_skew = values.groupby([data["date"], data["code"]], observed=True).skew()
    expected_kurt = values.groupby([data["date"], data["code"]], observed=True).apply(
        lambda group: group.kurt()
    )
    actual_skew = apply_reduce_unary("r_skew", values, data)
    actual_kurt = apply_reduce_unary("r_kurt", values, data)
    assert np.allclose(actual_skew.sort_index(), expected_skew.sort_index(), atol=1e-10)
    assert np.allclose(actual_kurt.sort_index(), expected_kurt.sort_index(), atol=1e-10)
