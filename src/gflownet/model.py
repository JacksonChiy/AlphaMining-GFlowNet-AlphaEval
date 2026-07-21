from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .grammar import ACTION_TOKENS, Vocabulary


@dataclass
class PolicyConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 32

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class GFlowNetPolicy(nn.Module):
    """Transformer encoder over the action history plus three state statistics."""

    def __init__(self, config: PolicyConfig, vocabulary: Vocabulary | None = None) -> None:
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary or Vocabulary()
        self.embedding = nn.Embedding(len(self.vocabulary.tokens), config.hidden_dim, padding_idx=self.vocabulary.pad_id)
        self.position = nn.Embedding(config.max_sequence_length + 1, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
        self.state_projection = nn.Sequential(
            nn.Linear(3, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, len(ACTION_TOKENS)),
        )

    def forward(self, token_ids: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, length = token_ids.shape
        if length > self.config.max_sequence_length:
            raise ValueError(f"Sequence length {length} exceeds configured maximum")
        positions = torch.arange(length, device=token_ids.device).expand(batch, -1)
        encoded = self.embedding(token_ids) + self.position(positions)
        padding_mask = token_ids.eq(self.vocabulary.pad_id)
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        last_index = (~padding_mask).sum(dim=1).sub(1).clamp_min(0)
        pooled = encoded[torch.arange(batch, device=token_ids.device), last_index]
        return self.head(pooled + self.state_projection(state_features))

