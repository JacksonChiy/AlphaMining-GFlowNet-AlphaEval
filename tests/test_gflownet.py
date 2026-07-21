from __future__ import annotations

import torch

from src.gflownet import GFlowNetPolicy, GrammarState, PolicyConfig, Vocabulary
from src.gflownet.grammar import ACTION_TOKENS


def test_grammar_produces_parseable_terminal_expression() -> None:
    state = GrammarState(max_depth=3, max_nodes=5)
    for action in ("cs_rank", "ts_mean", "W5", "close"):
        state = state.step(action)
    assert state.terminal
    assert str(state.expression) == "cs_rank(ts_mean(close,5))"


def test_transformer_policy_output_shape() -> None:
    config = PolicyConfig(hidden_dim=32, num_layers=1, num_heads=4, max_sequence_length=16)
    vocabulary = Vocabulary()
    model = GFlowNetPolicy(config, vocabulary)
    ids = torch.tensor([[vocabulary.bos_id]], dtype=torch.long)
    features = torch.zeros((1, 3))
    assert model(ids, features).shape == (1, len(ACTION_TOKENS))

