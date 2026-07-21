from __future__ import annotations

import pandas as pd
import torch

from src.gflownet import (
    GFlowNetPolicy,
    GFlowNetTrainer,
    GrammarState,
    PolicyConfig,
    RewardEvaluator,
    TrainerConfig,
    Vocabulary,
)
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


def test_training_prints_epoch_metrics(
    daily_prices: pd.DataFrame,
    tmp_path,
    capsys,
) -> None:
    policy = GFlowNetPolicy(
        PolicyConfig(hidden_dim=16, num_layers=1, num_heads=4, max_sequence_length=16)
    )
    evaluator = RewardEvaluator(daily_prices, horizon=5, min_cross_section=5)
    trainer = GFlowNetTrainer(
        policy,
        evaluator,
        TrainerConfig(
            epochs=1,
            trajectories_per_epoch=1,
            mixed_precision=False,
            max_depth=2,
            max_nodes=3,
        ),
        device="cpu",
    )
    checkpoint = tmp_path / "gflownet.pt"

    metrics = trainer.train(checkpoint)
    output = capsys.readouterr().out

    assert "[GFlowNet] training_start" in output
    assert "epoch=001/001" in output
    assert "mean_reward=" in output
    assert "checkpoint=saved" in output
    assert "[GFlowNet] training_complete" in output
    assert checkpoint.exists()
    assert {
        "learning_rate",
        "gradient_norm",
        "epoch_seconds",
        "elapsed_seconds",
        "gpu_allocated_gb",
        "gpu_reserved_gb",
        "gpu_peak_gb",
    }.issubset(metrics.columns)
