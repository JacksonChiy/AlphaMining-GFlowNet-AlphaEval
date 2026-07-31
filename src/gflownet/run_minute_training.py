from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.gflownet.minute_factor_pool import save_minute_alpha_pool
from src.gflownet.minute_grammar import MinuteVocabulary
from src.gflownet.minute_reward import MinuteRewardEvaluator
from src.gflownet.minute_trainer import MinuteGFlowNetTrainer
from src.gflownet.model import GFlowNetPolicy, PolicyConfig
from src.gflownet.run_training import gpu_report
from src.gflownet.trainer import TrainerConfig
from src.utils import create_experiment, load_config, seed_everything, slice_date_range


def _load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Minute training input does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported minute input format: {suffix}; use pkl, parquet or csv")


def run(config_path: str, require_a100: bool = True, pool_size: int | None = None) -> Path:
    config = load_config(config_path)
    dataset = config["dataset"]
    pool_size = int(pool_size or config.get("pipeline", {}).get("pool_size", 100))
    if pool_size <= 0:
        raise ValueError("pipeline.pool_size must be positive")
    hardware = gpu_report(require_a100)
    print(f"[MinuteGFlowNet] hardware={hardware}", flush=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(int(config["training"]["seed"]))
    experiment_dir = create_experiment(config_path)
    print(f"[MinuteGFlowNet] experiment_id={experiment_dir.name}", flush=True)

    minute_data = _load_frame(dataset["minute_file"])
    daily_data = _load_frame(dataset["daily_file"])
    for frame, label in ((minute_data, "minute"), (daily_data, "daily")):
        if not {"date", "code"}.issubset(frame.columns):
            raise ValueError(f"{label} input must contain date and code")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["code"] = frame["code"].astype(str)
    mining_start = dataset.get("mining_start_date")
    mining_end = dataset.get("mining_end_date")
    minute_mining = slice_date_range(
        minute_data, mining_start, mining_end, label="minute mining data"
    )
    daily_mining = slice_date_range(
        daily_data, mining_start, mining_end, label="daily reward data"
    )
    print(
        f"[MinuteGFlowNet] minute_data rows={len(minute_mining):,} "
        f"dates={minute_mining['date'].nunique():,} stocks={minute_mining['code'].nunique():,} "
        f"start={minute_mining['date'].min()} end={minute_mining['date'].max()}",
        flush=True,
    )
    print(
        f"[MinuteGFlowNet] daily_reward_data rows={len(daily_mining):,} "
        f"dates={daily_mining['date'].nunique():,} stocks={daily_mining['code'].nunique():,}",
        flush=True,
    )
    evaluator = MinuteRewardEvaluator(minute_mining, daily_mining, **config["reward"])
    policy_values = dict(config["model"])
    policy_values.pop("name", None)
    model = GFlowNetPolicy(PolicyConfig(**policy_values), MinuteVocabulary())
    print(
        f"[MinuteGFlowNet] model_parameters={sum(p.numel() for p in model.parameters()):,} "
        f"actions={len(model.vocabulary.action_tokens)}",
        flush=True,
    )
    training_values = dict(config["training"])
    training_values.pop("seed", None)
    trainer = MinuteGFlowNetTrainer(
        model,
        evaluator,
        TrainerConfig(seed=int(config["training"]["seed"]), **training_values),
    )
    checkpoint = Path(config.get("outputs", {}).get("checkpoint", "checkpoints/gflownet_minute_best.pt"))
    metrics = trainer.train(checkpoint)
    outputs = config.get("outputs", {})
    metrics_path = Path(outputs.get("metrics", "results/minute_gflownet_training_metrics.csv"))
    trajectory_path = Path(outputs.get("trajectory_metrics", "results/minute_gflownet_trajectory_metrics.csv"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(trainer.trajectory_history).to_csv(trajectory_path, index=False)
    metrics.to_csv(experiment_dir / "model_metrics.csv", index=False)

    loaded = MinuteGFlowNetTrainer.load_checkpoint(checkpoint, evaluator)
    print(f"[MinuteGFlowNet] alpha_pool_generation_start target_size={pool_size}", flush=True)
    pool = loaded.generate_pool(size=pool_size)
    metadata, matrix = save_minute_alpha_pool(
        pool,
        minute_data,
        daily_data,
        metadata_path=outputs.get("alpha_pool", "results/minute_alpha_pool.csv"),
        matrix_path=outputs.get("factor_matrix", "results/minute_alpha_factor_matrix.pkl"),
        min_coverage=evaluator.min_coverage,
    )
    metadata.to_csv(experiment_dir / "factor_results.csv", index=False)
    print(
        f"[MinuteGFlowNet] complete factors={len(metadata)} matrix_rows={len(matrix):,} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the report chart-27~30 minute grammar")
    parser.add_argument("--config", default="configs/minute_training_config.yaml")
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--pool-size", type=int, default=None)
    args = parser.parse_args()
    run(args.config, require_a100=not args.allow_non_a100, pool_size=args.pool_size)


if __name__ == "__main__":
    main()
