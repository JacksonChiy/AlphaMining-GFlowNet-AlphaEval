from __future__ import annotations

import argparse
import platform
from pathlib import Path

import pandas as pd
import torch

from src.gflownet.model import GFlowNetPolicy, PolicyConfig
from src.gflownet.reward import RewardEvaluator
from src.gflownet.trainer import GFlowNetTrainer, TrainerConfig, save_alpha_pool
from src.utils import create_experiment, load_config, seed_everything


def gpu_report(require_a100: bool = False) -> dict[str, str | bool | float]:
    available = torch.cuda.is_available()
    result: dict[str, str | bool | float] = {
        "cuda_available": available,
        "torch_version": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "platform": platform.platform(),
    }
    if available:
        properties = torch.cuda.get_device_properties(0)
        result.update({
            "gpu_name": properties.name,
            "gpu_memory_gb": round(properties.total_memory / 1024**3, 2),
        })
        if require_a100 and "A100" not in properties.name.upper():
            raise RuntimeError(f"A100 required, but detected {properties.name}")
    elif require_a100:
        raise RuntimeError("A100 CUDA GPU is required but CUDA is unavailable")
    return result


def run(config_path: str, require_a100: bool = True, pool_size: int = 100) -> Path:
    config = load_config(config_path)
    hardware = gpu_report(require_a100)
    print(f"[GFlowNet] hardware={hardware}", flush=True)
    seed_everything(int(config["training"]["seed"]))
    experiment_dir = create_experiment(config_path)
    print(f"[GFlowNet] experiment_id={experiment_dir.name}", flush=True)
    data = pd.read_pickle(config["dataset"]["output"])
    print(
        f"[GFlowNet] data_loaded rows={len(data)} dates={data['date'].nunique()} "
        f"stocks={data['code'].nunique()}",
        flush=True,
    )
    evaluator = RewardEvaluator(data, **config["reward"])
    policy_values = dict(config["model"])
    policy_values.pop("name", None)
    model = GFlowNetPolicy(PolicyConfig(**policy_values))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[GFlowNet] model_parameters={parameter_count:,}", flush=True)
    training_values = dict(config["training"])
    training_values.pop("seed", None)
    trainer_config = TrainerConfig(seed=int(config["training"]["seed"]), **training_values)
    trainer = GFlowNetTrainer(model, evaluator, trainer_config)
    metrics = trainer.train("checkpoints/gflownet_best.pt")
    Path("results").mkdir(exist_ok=True)
    metrics.to_csv("results/gflownet_training_metrics.csv", index=False)
    trajectory_metrics = pd.DataFrame(trainer.trajectory_history)
    trajectory_metrics.to_csv("results/gflownet_trajectory_metrics.csv", index=False)
    metrics.to_csv(experiment_dir / "model_metrics.csv", index=False)
    trajectory_metrics.to_csv(experiment_dir / "trajectory_metrics.csv", index=False)
    print("[GFlowNet] reloading_best_checkpoint", flush=True)
    loaded = GFlowNetTrainer.load_checkpoint("checkpoints/gflownet_best.pt", evaluator)
    print(f"[GFlowNet] alpha_pool_generation_start target_size={pool_size}", flush=True)
    pool = loaded.generate_pool(size=pool_size)
    metadata, _ = save_alpha_pool(pool, data)
    metadata.to_csv(experiment_dir / "factor_results.csv", index=False)
    print(
        f"[GFlowNet] alpha_pool_generation_complete factors={len(metadata)} "
        "metadata=results/alpha_pool.csv matrix=results/alpha_factor_matrix.pkl",
        flush=True,
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Only for small CPU/GPU smoke tests; formal training requires A100.",
    )
    parser.add_argument("--pool-size", type=int, default=100)
    args = parser.parse_args()
    run(args.config, not args.allow_non_a100, args.pool_size)


if __name__ == "__main__":
    main()
