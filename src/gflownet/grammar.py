from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from src.expression.tree import (
    BINARY_OPS,
    CS_OPS,
    FEATURES,
    TS_UNARY_OPS,
    UNARY_OPS,
    WINDOWS,
    Expression,
    expression_from_tokens,
)

WINDOW_TOKENS = tuple(f"W{window}" for window in WINDOWS)
ACTION_TOKENS = FEATURES + UNARY_OPS + BINARY_OPS + TS_UNARY_OPS + CS_OPS + WINDOW_TOKENS


class Vocabulary:
    def __init__(self) -> None:
        self.special = ("<PAD>", "<BOS>")
        self.tokens = self.special + ACTION_TOKENS
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = dict(enumerate(self.tokens))

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def action_id(self, token: str) -> int:
        return ACTION_TOKENS.index(token)


@dataclass(frozen=True)
class GrammarState:
    """A partial prefix-expression state with explicit pending grammar symbols."""

    tokens: tuple[str, ...] = ()
    pending: tuple[tuple[str, int], ...] = (("expr", 1),)
    max_depth: int = 5
    max_nodes: int = 15
    operator_count: int = 0
    feature_count: int = 0
    max_depth_seen: int = 1

    @property
    def terminal(self) -> bool:
        return not self.pending

    @property
    def node_count(self) -> int:
        return self.operator_count + self.feature_count

    @property
    def expression(self) -> Expression | None:
        return expression_from_tokens(self.tokens) if self.terminal else None

    def valid_actions(self) -> tuple[str, ...]:
        if self.terminal:
            return ()
        symbol, depth = self.pending[-1]
        if symbol == "window":
            return WINDOW_TOKENS
        remaining_expr = sum(item[0] == "expr" for item in self.pending)
        must_close = self.node_count + remaining_expr >= self.max_nodes
        if depth >= self.max_depth or must_close:
            return FEATURES
        return FEATURES + UNARY_OPS + BINARY_OPS + TS_UNARY_OPS + CS_OPS

    def action_mask(self) -> list[bool]:
        valid = set(self.valid_actions())
        return [token in valid for token in ACTION_TOKENS]

    def step(self, action: str) -> "GrammarState":
        if action not in self.valid_actions():
            raise ValueError(f"Invalid action {action!r}; expected one of {self.valid_actions()}")
        symbol, depth = self.pending[-1]
        pending = list(self.pending[:-1])
        operators, features = self.operator_count, self.feature_count
        max_seen = max(self.max_depth_seen, depth)
        if symbol == "window":
            pass
        elif action in FEATURES:
            features += 1
        elif action in UNARY_OPS or action in CS_OPS:
            operators += 1
            pending.append(("expr", depth + 1))
        elif action in BINARY_OPS:
            operators += 1
            # Stack is LIFO: right is pushed before left, yielding canonical prefix order.
            pending.extend((("expr", depth + 1), ("expr", depth + 1)))
        elif action in TS_UNARY_OPS:
            operators += 1
            pending.extend((("expr", depth + 1), ("window", depth)))
        else:
            raise AssertionError(action)
        return replace(
            self,
            tokens=self.tokens + (action,),
            pending=tuple(pending),
            operator_count=operators,
            feature_count=features,
            max_depth_seen=max_seen,
        )

    def handcrafted_features(self) -> tuple[float, float, float]:
        return (
            self.max_depth_seen / self.max_depth,
            self.operator_count / max(1, self.max_nodes),
            self.node_count / max(1, self.max_nodes),
        )

