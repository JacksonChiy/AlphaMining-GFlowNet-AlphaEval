from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.expression import Expression

from .grammar import ACTION_TOKENS, GrammarState, Vocabulary
from .model import GFlowNetPolicy, PolicyConfig
from .reward import RewardBreakdown, RewardEvaluator


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
    reward_workers: int = 4
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
        self.trajectory_history: list[dict[str, float | str]] = []

    def _state_tensors(self, state: GrammarState) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.vocabulary.bos_id, *self.vocabulary.encode(state.tokens)]
        token_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        features = torch.tensor(state.handcrafted_features(), dtype=torch.float32, device=self.device).unsqueeze(0)
        return token_ids, features

    def _batch_state_tensors(
        self, states: list[GrammarState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            torch.tensor(
                [self.vocabulary.bos_id, *self.vocabulary.encode(state.tokens)],
                dtype=torch.long,
                device=self.device,
            )
            for state in states
        ]
        token_ids = pad_sequence(
            sequences, batch_first=True, padding_value=self.vocabulary.pad_id
        )
        features = torch.tensor(
            [state.handcrafted_features() for state in states],
            dtype=torch.float32,
            device=self.device,
        )
        return token_ids, features

    def sample_trajectory(self, greedy: bool = False) -> tuple[Expression, torch.Tensor, list[str]]:
        return self.sample_trajectories(1, greedy=greedy)[0]

    def sample_trajectories(
        self, batch_size: int, greedy: bool = False
    ) -> list[tuple[Expression, torch.Tensor, list[str]]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        states = [
            GrammarState(max_depth=self.config.max_depth, max_nodes=self.config.max_nodes)
            for _ in range(batch_size)
        ]
        log_forward: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        while active_indices:
            active_states = [states[index] for index in active_indices]
            token_ids, features = self._batch_state_tensors(active_states)
            logits = self.model(token_ids, features)
            masks = torch.tensor(
                [state.action_mask() for state in active_states],
                dtype=torch.bool,
                device=self.device,
            )
            logits = logits.masked_fill(~masks, -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            action_indices = logits.argmax(dim=-1) if greedy else distribution.sample()
            step_log_prob = distribution.log_prob(action_indices)
            actions = action_indices.detach().cpu().tolist()
            next_active: list[int] = []
            for local_index, state_index in enumerate(active_indices):
                log_forward[state_index].append(step_log_prob[local_index])
                states[state_index] = states[state_index].step(ACTION_TOKENS[actions[local_index]])
                if not states[state_index].terminal:
                    next_active.append(state_index)
            active_indices = next_active
        trajectories: list[tuple[Expression, torch.Tensor, list[str]]] = []
        for state, log_probabilities in zip(states, log_forward):
            assert state.expression is not None
            trajectories.append(
                (state.expression, torch.stack(log_probabilities).sum(), list(state.tokens))
            )
        return trajectories

    def _evaluate_expressions(
        self,
        expressions: list[Expression],
        executor: Executor | None = None,
        log_progress: bool = False,
    ) -> list[RewardBreakdown]:
        unique: dict[str, Expression] = {}
        for expression in expressions:
            unique.setdefault(str(expression), expression)
        unique_expressions = list(unique.values())
        evaluated_by_expression: dict[str, RewardBreakdown] = {}
        if executor is None:
            for completed, expression in enumerate(unique_expressions, start=1):
                key = str(expression)
                evaluated_by_expression[key] = self.reward_evaluator.evaluate(expression)
                if log_progress:
                    print(
                        f"[GFlowNet] reward_progress completed={completed:03d}/"
                        f"{len(unique_expressions):03d} expression={expression}",
                        flush=True,
                    )
        else:
            future_to_expression = {
                executor.submit(self.reward_evaluator.evaluate, expression): expression
                for expression in unique_expressions
            }
            for completed, future in enumerate(as_completed(future_to_expression), start=1):
                expression = future_to_expression[future]
                evaluated_by_expression[str(expression)] = future.result()
                if log_progress:
                    print(
                        f"[GFlowNet] reward_progress completed={completed:03d}/"
                        f"{len(unique_expressions):03d} expression={expression}",
                        flush=True,
                    )
        return [evaluated_by_expression[str(expression)] for expression in expressions]

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
        total_trajectory_steps = self.config.epochs * self.config.trajectories_per_epoch
        print(
            "[GFlowNet] training_start "
            f"device={self.device} epochs={self.config.epochs} "
            f"trajectories_per_epoch={self.config.trajectories_per_epoch} "
            f"reward_workers={self.config.reward_workers} "
            f"amp={self.amp_enabled} checkpoint={checkpoint_path}",
            flush=True,
        )
        reward_executor = (
            ThreadPoolExecutor(max_workers=self.config.reward_workers)
            if self.config.reward_workers > 1
            else None
        )
        for epoch in range(1, self.config.epochs + 1):
            epoch_started = time.perf_counter()
            print(
                f"[GFlowNet] epoch_start epoch={epoch:03d}/{self.config.epochs:03d} "
                f"trajectories={self.config.trajectories_per_epoch} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
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
                print(
                    f"[GFlowNet] batch_sampling_start epoch={epoch:03d} "
                    f"batch_size={self.config.trajectories_per_epoch}",
                    flush=True,
                )
                sampling_started = time.perf_counter()
                trajectories = self.sample_trajectories(self.config.trajectories_per_epoch)
                sampling_seconds = time.perf_counter() - sampling_started
                expressions = [trajectory[0] for trajectory in trajectories]
                log_probabilities = [trajectory[1] for trajectory in trajectories]
                token_sequences = [trajectory[2] for trajectory in trajectories]
                print(
                    f"[GFlowNet] batch_sampling_complete epoch={epoch:03d} "
                    f"seconds={sampling_seconds:.2f}",
                    flush=True,
                )
                reward_started = time.perf_counter()
                breakdowns = self._evaluate_expressions(
                    expressions, reward_executor, log_progress=True
                )
                reward_seconds = time.perf_counter() - reward_started
                for log_pf, breakdown in zip(log_probabilities, breakdowns):
                    losses.append(self.trajectory_balance_loss(log_pf, breakdown.reward))
                    rewards.append(breakdown.reward)
                    rank_ics.append(breakdown.rank_ic)
                loss = torch.stack(losses).mean()
                step_values = torch.stack(
                    [torch.stack(log_probabilities), torch.stack(losses)]
                ).detach().float().cpu().numpy()
                average_step_seconds = (
                    sampling_seconds + reward_seconds
                ) / self.config.trajectories_per_epoch
                for trajectory_index, (expression, tokens, breakdown) in enumerate(
                    zip(expressions, token_sequences, breakdowns), start=1
                ):
                    global_step = (
                        (epoch - 1) * self.config.trajectories_per_epoch + trajectory_index
                    )
                    progress_pct = 100.0 * global_step / total_trajectory_steps
                    trajectory_record: dict[str, float | str] = {
                        "epoch": float(epoch),
                        "step": float(trajectory_index),
                        "global_step": float(global_step),
                        "progress_pct": float(progress_pct),
                        "expression": str(expression),
                        "action_count": float(len(tokens)),
                        "reward": float(breakdown.reward),
                        "rank_ic": float(breakdown.rank_ic),
                        "long_ir": float(breakdown.long_ir),
                        "risk_penalty": float(breakdown.risk_penalty),
                        "log_pf": float(step_values[0, trajectory_index - 1]),
                        "tb_loss": float(step_values[1, trajectory_index - 1]),
                        "trajectory_seconds": float(average_step_seconds),
                        "sampling_batch_seconds": float(sampling_seconds),
                        "reward_batch_seconds": float(reward_seconds),
                        "elapsed_seconds": float(time.perf_counter() - training_started),
                    }
                    self.trajectory_history.append(trajectory_record)
                    print(
                        f"[GFlowNet] trajectory"
                        f" epoch={epoch:03d}/{self.config.epochs:03d}"
                        f" step={trajectory_index:03d}/{self.config.trajectories_per_epoch:03d}"
                        f" global_step={global_step:05d}/{total_trajectory_steps:05d}"
                        f" progress={progress_pct:6.2f}%"
                        f" actions={len(tokens)}"
                        f" expression={expression}"
                        f" reward={breakdown.reward:.6f}"
                        f" rank_ic={breakdown.rank_ic:.6f}"
                        f" long_ir={breakdown.long_ir:.6f}"
                        f" risk_penalty={breakdown.risk_penalty:.6f}"
                        f" log_pf={trajectory_record['log_pf']:.6f}"
                        f" tb_loss={trajectory_record['tb_loss']:.6f}"
                        f" step_seconds={average_step_seconds:.2f}"
                        f" elapsed_seconds={trajectory_record['elapsed_seconds']:.2f}",
                        flush=False,
                    )
                sys.stdout.flush()
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
                "sampling_seconds": float(sampling_seconds),
                "reward_seconds": float(reward_seconds),
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
                f" sampling_seconds={sampling_seconds:.2f}"
                f" reward_seconds={reward_seconds:.2f}"
                f" epoch_seconds={epoch_seconds:.2f}"
                f" elapsed_seconds={elapsed_seconds:.2f}"
                f" checkpoint={'saved' if is_best else '-'}"
                f"{gpu_log}",
                flush=True,
            )
        if reward_executor is not None:
            reward_executor.shutdown(wait=True)
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
            "trajectory_history": self.trajectory_history,
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
        trainer.trajectory_history = list(payload.get("trajectory_history", []))
        return trainer

    @torch.no_grad()
    def generate_pool(self, size: int = 100, attempts: int = 2000) -> list[dict[str, Any]]:
        self.model.eval()
        unique: dict[str, dict[str, Any]] = {}
        reward_executor = (
            ThreadPoolExecutor(max_workers=self.config.reward_workers)
            if self.config.reward_workers > 1
            else None
        )
        attempt = 0
        target_candidates = size * 5
        while attempt < attempts and len(unique) < target_candidates:
            batch_size = min(self.config.trajectories_per_epoch, attempts - attempt)
            trajectories = self.sample_trajectories(batch_size)
            accepted_indices: list[int] = []
            seen = set(unique)
            for index, (expression, _, _) in enumerate(trajectories):
                key = str(expression)
                if key not in seen:
                    accepted_indices.append(index)
                    seen.add(key)
            accepted_expressions = [trajectories[index][0] for index in accepted_indices]
            accepted_breakdowns = self._evaluate_expressions(
                accepted_expressions, reward_executor, log_progress=True
            )
            breakdown_by_index = dict(zip(accepted_indices, accepted_breakdowns))
            for index, (expression, _, tokens) in enumerate(trajectories):
                attempt += 1
                key = str(expression)
                if index not in breakdown_by_index:
                    print(
                        f"[GFlowNet] alpha_pool_step attempt={attempt:04d}/{attempts:04d} "
                        f"status=duplicate unique={len(unique)} "
                        f"target_candidates={target_candidates} expression={expression}",
                        flush=False,
                    )
                    continue
                breakdown = breakdown_by_index[index]
                unique[key] = {
                    "expression": expression,
                    "tokens": tokens,
                    **breakdown.to_dict(),
                    "complexity": expression.complexity(),
                    "depth": expression.depth(),
                }
                print(
                    f"[GFlowNet] alpha_pool_step attempt={attempt:04d}/{attempts:04d} "
                    f"status=accepted unique={len(unique)} "
                    f"target_candidates={target_candidates} reward={breakdown.reward:.6f} "
                    f"rank_ic={breakdown.rank_ic:.6f} expression={expression}",
                    flush=False,
                )
                if len(unique) >= target_candidates:
                    break
            sys.stdout.flush()
        if reward_executor is not None:
            reward_executor.shutdown(wait=True)
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
