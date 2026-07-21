from __future__ import annotations

import argparse
import platform
from pathlib import Path

import pandas as pd
import torch

from src.gflownet.model import GFlowNetPolicy, PolicyConfig
from src.gflownet.reward import RewardEvaluator
from src.gflownet.trainer import GFlowNetTrainer, TrainerConfig, save_alpha_pool
from src.utils import load_config, seed_everything


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


def run(config_path: str, require_a100: bool = False, pool_size: int = 100) -> None:
    config = load_config(config_path)
    print(gpu_report(require_a100))
    seed_everything(int(config["training"]["seed"]))
    data = pd.read_pickle(config["dataset"]["output"])
    evaluator = RewardEvaluator(data, **config["reward"])
    policy_values = dict(config["model"])
    policy_values.pop("name", None)
    model = GFlowNetPolicy(PolicyConfig(**policy_values))
    training_values = dict(config["training"])
    training_values.pop("seed", None)
    trainer_config = TrainerConfig(seed=int(config["training"]["seed"]), **training_values)
    trainer = GFlowNetTrainer(model, evaluator, trainer_config)
    metrics = trainer.train("checkpoints/gflownet_best.pt")
    Path("results").mkdir(exist_ok=True)
    metrics.to_csv("results/gflownet_training_metrics.csv", index=False)
    loaded = GFlowNetTrainer.load_checkpoint("checkpoints/gflownet_best.pt", evaluator)
    pool = loaded.generate_pool(size=pool_size)
    save_alpha_pool(pool, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--pool-size", type=int, default=100)
    args = parser.parse_args()
    run(args.config, args.require_a100, args.pool_size)


if __name__ == "__main__":
    main()
