from __future__ import annotations

from pathlib import Path

import torch

from .minute_grammar import MINUTE_ACTION_TOKENS, MinuteGrammarState, MinuteVocabulary
from .minute_reward import MinuteRewardEvaluator
from .model import GFlowNetPolicy, PolicyConfig
from .trainer import GFlowNetTrainer, TrainerConfig


class MinuteGFlowNetTrainer(GFlowNetTrainer):
    def __init__(
        self,
        model: GFlowNetPolicy,
        reward_evaluator: MinuteRewardEvaluator,
        config: TrainerConfig,
        device: str | torch.device | None = None,
    ) -> None:
        if tuple(model.vocabulary.action_tokens) != MINUTE_ACTION_TOKENS:
            raise ValueError("MinuteGFlowNetTrainer requires MinuteVocabulary")
        super().__init__(model, reward_evaluator, config, device)

    def _make_state(self) -> MinuteGrammarState:
        return MinuteGrammarState(max_depth=self.config.max_depth, max_nodes=self.config.max_nodes)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        reward_evaluator: MinuteRewardEvaluator,
        device: str | torch.device | None = None,
    ) -> "MinuteGFlowNetTrainer":
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
        if tuple(payload["action_tokens"]) != MINUTE_ACTION_TOKENS:
            raise ValueError("Checkpoint is not compatible with the minute grammar")
        model = GFlowNetPolicy(PolicyConfig(**payload["policy_config"]), MinuteVocabulary())
        trainer = cls(model, reward_evaluator, TrainerConfig(**payload["trainer_config"]), target_device)
        trainer.model.load_state_dict(payload["model_state"])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        trainer.log_z.data.copy_(payload["log_z"].to(target_device))
        trainer.history = list(payload.get("history", []))
        trainer.trajectory_history = list(payload.get("trajectory_history", []))
        return trainer
