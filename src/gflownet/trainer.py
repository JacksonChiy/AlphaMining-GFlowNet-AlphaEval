from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.expression import Expression

from .grammar import ACTION_TOKENS, GrammarState, Vocabulary
from .model import GFlowNetPolicy, PolicyConfig
from .reward import RewardEvaluator


@dataclass
class TrainerConfig:
    epochs: int = 100
    trajectories_per_epoch: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    reward_temperature: float = 1.0
    mixed_precision: bool = True
    max_depth: int = 5
    max_nodes: int = 15
    gradient_clip: float = 1.0
    seed: int = 42


class GFlowNetTrainer:
    def __init__(
        self,
        model: GFlowNetPolicy,
        reward_evaluator: RewardEvaluator,
        config: TrainerConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.reward_evaluator = reward_evaluator
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.log_z = nn.Parameter(torch.zeros((), device=self.device))
        self.optimizer = torch.optim.AdamW(
            [*self.model.parameters(), self.log_z],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.amp_enabled = bool(config.mixed_precision and self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:  # PyTorch 2.2 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.vocabulary = Vocabulary()
        self.rng = np.random.default_rng(config.seed)
        self.history: list[dict[str, float]] = []

    def _state_tensors(self, state: GrammarState) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocabulary.bos_id, *self.vocabulary.encode(state.tokens)]
        token_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        features = torch.tensor(state.handcrafted_features(), dtype=torch.float32, device=self.device).unsqueeze(0)
        return token_ids, features

    def sample_trajectory(self, greedy: bool = False) -> tuple[Expression, torch.Tensor, list[str]]:
        state = GrammarState(max_depth=self.config.max_depth, max_nodes=self.config.max_nodes)
        log_forward: list[torch.Tensor] = []
        while not state.terminal:
            token_ids, features = self._state_tensors(state)
            logits = self.model(token_ids, features).squeeze(0)
            mask = torch.tensor(state.action_mask(), dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask, -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            action_index = torch.argmax(logits) if greedy else distribution.sample()
            log_forward.append(distribution.log_prob(action_index))
            state = state.step(ACTION_TOKENS[int(action_index.item())])
        assert state.expression is not None
        return state.expression, torch.stack(log_forward).sum(), list(state.tokens)

    def trajectory_balance_loss(self, log_pf: torch.Tensor, reward: float) -> torch.Tensor:
        # Prefix-tree construction has exactly one parent per non-root state, so sum(log PB) = 0.
        log_reward = math.log(max(reward, self.reward_evaluator.reward_floor)) / self.config.reward_temperature
        target = torch.tensor(log_reward, dtype=log_pf.dtype, device=self.device)
        return torch.square(self.log_z + log_pf - target)

    def train(self, checkpoint_path: str | Path = "checkpoints/gflownet_best.pt") -> pd.DataFrame:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        training_started = time.perf_counter()
        print(
            "[GFlowNet] training_start "
            f"device={self.device} epochs={self.config.epochs} "
            f"trajectories_per_epoch={self.config.trajectories_per_epoch} "
            f"amp={self.amp_enabled} checkpoint={checkpoint_path}",
            flush=True,
        )
        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.model.train()
            losses: list[torch.Tensor] = []
            rewards: list[float] = []
            rank_ics: list[float] = []
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                for _ in range(self.config.trajectories_per_epoch):
                    expression, log_pf, _ = self.sample_trajectory()
                    breakdown = self.reward_evaluator.evaluate(expression)
                    losses.append(self.trajectory_balance_loss(log_pf, breakdown.reward))
                    rewards.append(breakdown.reward)
                    rank_ics.append(breakdown.rank_ic)
                loss = torch.stack(losses).mean()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_seconds = time.perf_counter() - epoch_started
            elapsed_seconds = time.perf_counter() - training_started
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            gpu_allocated_gb = 0.0
            gpu_reserved_gb = 0.0
            gpu_peak_gb = 0.0
            if self.device.type == "cuda":
                gpu_allocated_gb = torch.cuda.memory_allocated(self.device) / 1024**3
                gpu_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                gpu_peak_gb = torch.cuda.max_memory_allocated(self.device) / 1024**3
            record = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "mean_reward": float(np.mean(rewards)),
                "max_reward": float(np.max(rewards)),
                "mean_rank_ic": float(np.mean(rank_ics)),
                "log_z": float(self.log_z.detach().cpu()),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "epoch_seconds": float(epoch_seconds),
                "elapsed_seconds": float(elapsed_seconds),
                "gpu_allocated_gb": float(gpu_allocated_gb),
                "gpu_reserved_gb": float(gpu_reserved_gb),
                "gpu_peak_gb": float(gpu_peak_gb),
            }
            self.history.append(record)
            is_best = record["loss"] < best_loss
            if is_best:
                best_loss = record["loss"]
                self.save_checkpoint(checkpoint_path, best_loss)
            gpu_log = ""
            if self.device.type == "cuda":
                gpu_log = (
                    f" gpu_allocated={gpu_allocated_gb:.2f}GB"
                    f" gpu_reserved={gpu_reserved_gb:.2f}GB"
                    f" gpu_peak={gpu_peak_gb:.2f}GB"
                )
            print(
                f"[GFlowNet] epoch={epoch:03d}/{self.config.epochs:03d}"
                f" loss={record['loss']:.6f}"
                f" mean_reward={record['mean_reward']:.6f}"
                f" max_reward={record['max_reward']:.6f}"
                f" mean_rank_ic={record['mean_rank_ic']:.6f}"
                f" log_z={record['log_z']:.6f}"
                f" grad_norm={record['gradient_norm']:.4f}"
                f" lr={learning_rate:.2e}"
                f" epoch_seconds={epoch_seconds:.2f}"
                f" elapsed_seconds={elapsed_seconds:.2f}"
                f" checkpoint={'saved' if is_best else '-'}"
                f"{gpu_log}",
                flush=True,
            )
        print(
            f"[GFlowNet] training_complete best_loss={best_loss:.6f} "
            f"elapsed_seconds={time.perf_counter() - training_started:.2f} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
        return pd.DataFrame(self.history)

    def save_checkpoint(self, path: str | Path, best_loss: float | None = None) -> None:
        payload: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "log_z": self.log_z.detach().cpu(),
            "policy_config": self.model.config.to_dict(),
            "trainer_config": asdict(self.config),
            "action_tokens": ACTION_TOKENS,
            "best_loss": best_loss,
            "history": self.history,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        reward_evaluator: RewardEvaluator,
        device: str | torch.device | None = None,
    ) -> "GFlowNetTrainer":
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
        if tuple(payload["action_tokens"]) != ACTION_TOKENS:
            raise ValueError("Checkpoint action vocabulary is incompatible with this code version")
        model = GFlowNetPolicy(PolicyConfig(**payload["policy_config"]))
        trainer = cls(model, reward_evaluator, TrainerConfig(**payload["trainer_config"]), target_device)
        trainer.model.load_state_dict(payload["model_state"])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.log_z.data.copy_(payload["log_z"].to(target_device))
        trainer.history = list(payload.get("history", []))
        return trainer

    @torch.no_grad()
    def generate_pool(self, size: int = 100, attempts: int = 2000) -> list[dict[str, Any]]:
        self.model.eval()
        unique: dict[str, dict[str, Any]] = {}
        for attempt in range(1, attempts + 1):
            expression, _, tokens = self.sample_trajectory()
            key = str(expression)
            if key in unique:
                if attempt % 100 == 0:
                    print(
                        f"[GFlowNet] alpha_pool_progress attempt={attempt}/{attempts} "
                        f"unique={len(unique)} target_candidates={size * 5}",
                        flush=True,
                    )
                continue
            breakdown = self.reward_evaluator.evaluate(expression)
            unique[key] = {
                "expression": expression,
                "tokens": tokens,
                **breakdown.to_dict(),
                "complexity": expression.complexity(),
                "depth": expression.depth(),
            }
            if attempt % 100 == 0:
                print(
                    f"[GFlowNet] alpha_pool_progress attempt={attempt}/{attempts} "
                    f"unique={len(unique)} target_candidates={size * 5} "
                    f"latest_reward={breakdown.reward:.6f}",
                    flush=True,
                )
            if len(unique) >= size * 5:
                break
        return sorted(unique.values(), key=lambda item: item["reward"], reverse=True)[:size]


def save_alpha_pool(
    pool: list[dict[str, Any]],
    data: pd.DataFrame,
    metadata_path: str | Path = "results/alpha_pool.csv",
    matrix_path: str | Path = "results/alpha_factor_matrix.pkl",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows: list[dict[str, Any]] = []
    matrix = data[["date", "code"]].copy()
    for index, item in enumerate(pool, start=1):
        name = f"factor_{index:03d}"
        expression: Expression = item["expression"]
        matrix[name] = expression.execute(data).to_numpy()
        metadata_rows.append({
            "factor": name,
            "expression": str(expression),
            **{key: value for key, value in item.items() if key not in {"expression", "tokens"}},
            "tokens": json.dumps(item["tokens"], ensure_ascii=False),
        })
    metadata = pd.DataFrame(metadata_rows)
    metadata_path, matrix_path = Path(metadata_path), Path(matrix_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    matrix.to_pickle(matrix_path)
    return metadata, matrix
