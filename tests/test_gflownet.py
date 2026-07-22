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
from src.gflownet.trainer import save_alpha_pool
from src.expression import expression_from_tokens


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


def test_batched_trajectory_sampling_preserves_gradients(daily_prices: pd.DataFrame) -> None:
    policy = GFlowNetPolicy(
        PolicyConfig(hidden_dim=16, num_layers=1, num_heads=4, max_sequence_length=16)
    )
    trainer = GFlowNetTrainer(
        policy,
        RewardEvaluator(daily_prices, horizon=5, min_cross_section=5),
        TrainerConfig(
            epochs=1,
            trajectories_per_epoch=4,
            mixed_precision=False,
            max_depth=2,
            max_nodes=3,
            reward_workers=1,
        ),
        device="cpu",
    )

    trajectories = trainer.sample_trajectories(4)
    objective = -torch.stack([log_pf for _, log_pf, _ in trajectories]).mean()
    objective.backward()

    assert len(trajectories) == 4
    assert all(expression is not None and tokens for expression, _, tokens in trajectories)
    assert any(parameter.grad is not None for parameter in policy.parameters())


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
            reward_workers=1,
        ),
        device="cpu",
    )
    checkpoint = tmp_path / "gflownet.pt"

    metrics = trainer.train(checkpoint)
    output = capsys.readouterr().out

    assert "[GFlowNet] training_start" in output
    assert "[GFlowNet] epoch_start epoch=001/001" in output
    assert "[GFlowNet] batch_sampling_start epoch=001" in output
    assert "[GFlowNet] batch_sampling_complete epoch=001" in output
    assert "[GFlowNet] reward_progress completed=001/001" in output
    assert "cache_hit_rate=" in output
    assert "[GFlowNet] trajectory epoch=001/001 step=001/001" in output
    assert "global_step=00001/00001" in output
    assert "progress=100.00%" in output
    assert "expression=" in output
    assert "tb_loss=" in output
    assert "epoch=001/001" in output
    assert "mean_reward=" in output
    assert "checkpoint=saved" in output
    assert "[GFlowNet] training_complete" in output
    assert checkpoint.exists()
    assert len(trainer.trajectory_history) == 1
    assert {
        "epoch",
        "step",
        "global_step",
        "progress_pct",
        "expression",
        "action_count",
        "reward",
        "rank_ic",
        "long_ir",
        "risk_penalty",
        "coverage",
        "valid_date_coverage",
        "coverage_penalty",
        "log_pf",
        "tb_loss",
        "trajectory_seconds",
        "elapsed_seconds",
    }.issubset(trainer.trajectory_history[0])
    assert {
        "learning_rate",
        "gradient_norm",
        "sampling_seconds",
        "reward_seconds",
        "epoch_seconds",
        "elapsed_seconds",
        "gpu_allocated_gb",
        "gpu_reserved_gb",
        "gpu_peak_gb",
        "mean_coverage",
        "cache_hits",
        "cache_misses",
        "cache_waits",
        "cache_evictions",
        "cache_hit_rate",
        "cache_entries",
        "cache_memory_mb",
        "ts_torch_calls",
        "ts_pandas_calls",
        "ts_torch_seconds",
    }.issubset(metrics.columns)


def test_save_alpha_pool_excludes_low_coverage_expression(
    daily_prices: pd.DataFrame, tmp_path
) -> None:
    expression = expression_from_tokens(["close"])
    common = {
        "expression": expression,
        "tokens": ["close"],
        "reward": 0.1,
        "valid_date_coverage": 1.0,
    }
    pool = [
        {**common, "coverage": 0.50},
        {**common, "coverage": 0.95, "reward": 0.05},
    ]

    metadata, matrix = save_alpha_pool(
        pool,
        daily_prices,
        metadata_path=tmp_path / "alpha_pool.csv",
        matrix_path=tmp_path / "alpha_matrix.pkl",
        min_coverage=0.80,
    )

    assert metadata["coverage"].tolist() == [0.95]
    assert [column for column in matrix if column.startswith("factor_")] == ["factor_001"]
