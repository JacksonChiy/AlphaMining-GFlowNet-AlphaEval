from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from src.expression.minute import (
    MASK_BINARY_OPS,
    MASK_BINARY_WINDOW_OPS,
    MASK_UNARY_OPS,
    MASK_WINDOW_OPS,
    MINUTE_BINARY_OPS,
    MINUTE_FEATURES,
    MINUTE_UNARY_OPS,
    MINUTE_WINDOW_OPS,
    MINUTE_WINDOWS,
    REDUCE_BINARY_OPS,
    REDUCE_UNARY_OPS,
    MinuteExpression,
    minute_expression_from_tokens,
)
from src.expression.tree import BINARY_OPS, CS_OPS, TS_UNARY_OPS, UNARY_OPS


MINUTE_WINDOW_TOKENS = tuple(f"W{window}" for window in MINUTE_WINDOWS)
MINUTE_ACTION_TOKENS = (
    MINUTE_FEATURES
    + MINUTE_UNARY_OPS
    + MINUTE_WINDOW_OPS
    + MINUTE_BINARY_OPS
    + MASK_WINDOW_OPS
    + MASK_UNARY_OPS
    + MASK_BINARY_WINDOW_OPS
    + MASK_BINARY_OPS
    + REDUCE_UNARY_OPS
    + REDUCE_BINARY_OPS
    + UNARY_OPS
    + BINARY_OPS
    + TS_UNARY_OPS
    + CS_OPS
    + MINUTE_WINDOW_TOKENS
)


class MinuteVocabulary:
    def __init__(self) -> None:
        self.special = ("<PAD>", "<BOS>")
        self.action_tokens = MINUTE_ACTION_TOKENS
        self.tokens = self.special + self.action_tokens
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
        return self.action_tokens.index(token)


@dataclass(frozen=True)
class MinuteGrammarState:
    """Report prefix grammar: intraday blocks followed by optional daily operators."""

    tokens: tuple[str, ...] = ()
    pending: tuple[tuple[str, int], ...] = (("daily", 1),)
    max_depth: int = 6
    max_nodes: int = 18
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
    def expression(self) -> MinuteExpression | None:
        return minute_expression_from_tokens(self.tokens) if self.terminal else None

    def valid_actions(self) -> tuple[str, ...]:
        if self.terminal:
            return ()
        symbol, depth = self.pending[-1]
        if symbol == "window":
            return MINUTE_WINDOW_TOKENS
        if symbol == "daily":
            remaining = sum(item[0] == "daily" for item in self.pending)
            must_close = depth >= self.max_depth - 1 or self.node_count + remaining >= self.max_nodes
            if must_close:
                return REDUCE_UNARY_OPS + REDUCE_BINARY_OPS
            return (
                REDUCE_UNARY_OPS + REDUCE_BINARY_OPS
                + UNARY_OPS + BINARY_OPS + TS_UNARY_OPS + CS_OPS
            )
        if symbol == "block":
            return REDUCE_UNARY_OPS + REDUCE_BINARY_OPS
        remaining = sum(item[0] in {"minute", "source"} for item in self.pending)
        must_close = self.node_count + remaining >= self.max_nodes
        if symbol == "minute":
            if depth >= self.max_depth or must_close:
                return MINUTE_FEATURES
            return MINUTE_FEATURES + MINUTE_UNARY_OPS + MINUTE_WINDOW_OPS + MINUTE_BINARY_OPS
        if symbol == "source":
            if depth >= self.max_depth or must_close:
                return MINUTE_FEATURES
            return (
                MINUTE_FEATURES + MINUTE_UNARY_OPS + MINUTE_WINDOW_OPS + MINUTE_BINARY_OPS
                + MASK_WINDOW_OPS + MASK_UNARY_OPS + MASK_BINARY_WINDOW_OPS + MASK_BINARY_OPS
            )
        raise AssertionError(symbol)

    def action_mask(self) -> list[bool]:
        valid = set(self.valid_actions())
        return [token in valid for token in MINUTE_ACTION_TOKENS]

    def step(self, action: str) -> "MinuteGrammarState":
        if action not in self.valid_actions():
            raise ValueError(f"Invalid minute action {action!r}; expected one of {self.valid_actions()}")
        symbol, depth = self.pending[-1]
        pending = list(self.pending[:-1])
        operators, features = self.operator_count, self.feature_count
        max_seen = max(self.max_depth_seen, depth)
        if symbol == "window":
            pass
        elif action in UNARY_OPS:
            operators += 1
            pending.append(("daily", depth + 1))
        elif action in BINARY_OPS:
            operators += 1
            pending.extend((("daily", depth + 1), ("daily", depth + 1)))
        elif action in TS_UNARY_OPS:
            operators += 1
            pending.extend((("daily", depth + 1), ("window", depth)))
        elif action in CS_OPS:
            operators += 1
            pending.append(("daily", depth + 1))
        elif action in MINUTE_FEATURES:
            features += 1
        elif action in REDUCE_UNARY_OPS:
            operators += 1
            pending.append(("source", depth + 1))
        elif action in REDUCE_BINARY_OPS:
            operators += 1
            pending.extend((("minute", depth + 1), ("minute", depth + 1)))
        elif action in MINUTE_UNARY_OPS:
            operators += 1
            pending.append(("minute", depth + 1))
        elif action in MINUTE_WINDOW_OPS:
            operators += 1
            pending.extend((("minute", depth + 1), ("window", depth)))
        elif action in MINUTE_BINARY_OPS:
            operators += 1
            pending.extend((("minute", depth + 1), ("minute", depth + 1)))
        elif action in MASK_WINDOW_OPS:
            operators += 1
            pending.extend((("minute", depth + 1), ("window", depth)))
        elif action in MASK_UNARY_OPS:
            operators += 1
            pending.append(("minute", depth + 1))
        elif action in MASK_BINARY_WINDOW_OPS:
            operators += 1
            pending.extend((("minute", depth + 1), ("minute", depth + 1), ("window", depth)))
        elif action in MASK_BINARY_OPS:
            operators += 1
            pending.extend((("minute", depth + 1), ("minute", depth + 1)))
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
