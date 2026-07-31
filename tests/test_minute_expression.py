from __future__ import annotations

import numpy as np
import pandas as pd
import torch

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
    minute_expression_from_tokens,
)
from src.gflownet.minute_grammar import MINUTE_ACTION_TOKENS, MinuteGrammarState, MinuteVocabulary
from src.gflownet.model import GFlowNetPolicy, PolicyConfig
from src.operators.minute import build_minute_features


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
    assert len(MINUTE_ACTION_TOKENS) == 71


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
