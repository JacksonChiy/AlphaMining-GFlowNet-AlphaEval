from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, Sequence

import numpy as np
import pandas as pd

from src.operators.minute import (
    apply_mask_binary,
    apply_mask_unary,
    apply_mask_window,
    apply_minute_binary,
    apply_minute_unary,
    apply_minute_window,
    apply_reduce_binary,
    apply_reduce_unary,
    build_minute_features,
)


MINUTE_FEATURES = (
    "open", "high", "low", "close", "vol", "amount",
    "ret", "vwap", "hl_pct", "bar_pos", "amihud", "rv", "signed_vol", "signed_amt",
    "typical", "vwap_cum", "twap", "obv", "pvt", "logret", "oc_ret",
)
MINUTE_WINDOWS = (5, 10, 20, 40, 60)
MINUTE_UNARY_OPS = ("m_ret", "m_logret", "m_rank", "m_zscore", "m_abs", "m_sign", "m_log")
MINUTE_WINDOW_OPS = ("m_delay", "m_delta", "m_ma", "m_std")
MINUTE_BINARY_OPS = ("m_add", "m_sub", "m_mul", "m_div")
MASK_WINDOW_OPS = ("m_head", "m_tail", "m_mid", "m_top", "m_bot", "m_xtreme")
MASK_UNARY_OPS = ("m_above", "m_below", "m_inner", "m_outer")
MASK_BINARY_WINDOW_OPS = ("m_at_top", "m_at_bot")
MASK_BINARY_OPS = ("m_when_pos", "m_when_gt")
REDUCE_UNARY_OPS = (
    "r_mean", "r_std", "r_sum", "r_max", "r_min", "r_median", "r_first", "r_last",
    "r_skew", "r_kurt", "r_slope", "r_rsquare", "r_argmax",
)
REDUCE_BINARY_OPS = ("r_corr", "r_cov", "r_wmean")


_ARITY = {
    "feature": 0,
    "minute_unary": 1,
    "minute_window": 1,
    "minute_binary": 2,
    "mask_window": 1,
    "mask_unary": 1,
    "mask_binary_window": 2,
    "mask_binary": 2,
    "reduce_unary": 1,
    "reduce_binary": 2,
}
_WINDOW_KINDS = {"minute_window", "mask_window", "mask_binary_window"}


@dataclass(frozen=True)
class MinuteNode:
    kind: str
    name: str
    children: tuple["MinuteNode", ...] = field(default_factory=tuple)
    window: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ARITY:
            raise ValueError(f"Unknown minute node kind: {self.kind}")
        if len(self.children) != _ARITY[self.kind]:
            raise ValueError(f"{self.kind} requires {_ARITY[self.kind]} children")
        if self.kind in _WINDOW_KINDS and self.window not in MINUTE_WINDOWS:
            raise ValueError(f"Invalid minute window: {self.window}")
        if self.kind not in _WINDOW_KINDS and self.window is not None:
            raise ValueError(f"{self.kind} does not accept a window")

    def render(self) -> str:
        if self.kind == "feature":
            return self.name
        arguments = [child.render() for child in self.children]
        if self.kind in _WINDOW_KINDS:
            arguments.append(str(self.window))
        return f"{self.name}({','.join(arguments)})"

    def complexity(self) -> int:
        return 1 + sum(child.complexity() for child in self.children)

    def depth(self) -> int:
        return 1 if not self.children else 1 + max(child.depth() for child in self.children)

    def prefix_tokens(self) -> Iterator[str]:
        yield self.name
        if self.kind in _WINDOW_KINDS:
            yield f"W{self.window}"
        for child in self.children:
            yield from child.prefix_tokens()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "window": self.window,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MinuteNode":
        return cls(
            kind=value["kind"],
            name=value["name"],
            window=value.get("window"),
            children=tuple(cls.from_dict(child) for child in value.get("children", [])),
        )


@dataclass(frozen=True)
class MinuteExpression:
    """A minute expression whose root reduction produces one value per date and stock."""

    root: MinuteNode
    FEATURES: ClassVar[tuple[str, ...]] = MINUTE_FEATURES

    def __post_init__(self) -> None:
        if self.root.kind not in {"reduce_unary", "reduce_binary"}:
            raise ValueError("A minute expression must terminate with a reduction operator")

    @classmethod
    def generate(cls, max_depth: int = 5, seed: int | None = None) -> "MinuteExpression":
        return MinuteExpressionGenerator(max_depth=max_depth, seed=seed).generate()

    def execute(self, data: pd.DataFrame) -> pd.Series:
        prepared = build_minute_features(data)
        cache: dict[MinuteNode, pd.Series] = {}
        result = self._execute_node(self.root, prepared, cache)
        result.name = str(self)
        return result.replace([np.inf, -np.inf], np.nan).astype(float)

    def _execute_node(
        self,
        node: MinuteNode,
        data: pd.DataFrame,
        cache: dict[MinuteNode, pd.Series],
    ) -> pd.Series:
        cached = cache.get(node)
        if cached is not None:
            return cached
        if node.kind == "feature":
            result = data[node.name].astype(float)
        else:
            values = [self._execute_node(child, data, cache) for child in node.children]
            if node.kind == "minute_unary":
                result = apply_minute_unary(node.name, values[0], data)
            elif node.kind == "minute_window":
                result = apply_minute_window(node.name, values[0], data, int(node.window))
            elif node.kind == "minute_binary":
                result = apply_minute_binary(node.name, values[0], values[1])
            elif node.kind == "mask_window":
                result = apply_mask_window(node.name, values[0], data, int(node.window))
            elif node.kind == "mask_unary":
                result = apply_mask_unary(node.name, values[0], data)
            elif node.kind == "mask_binary_window":
                result = apply_mask_binary(node.name, values[0], values[1], data, int(node.window))
            elif node.kind == "mask_binary":
                result = apply_mask_binary(node.name, values[0], values[1], data)
            elif node.kind == "reduce_unary":
                result = apply_reduce_unary(node.name, values[0], data)
            elif node.kind == "reduce_binary":
                result = apply_reduce_binary(node.name, values[0], values[1], data)
            else:
                raise AssertionError(node.kind)
        cache[node] = result
        return result

    def complexity(self) -> int:
        return self.root.complexity()

    def depth(self) -> int:
        return self.root.depth()

    def to_dict(self) -> dict[str, Any]:
        return self.root.to_dict()

    def to_tokens(self) -> list[str]:
        return list(self.root.prefix_tokens())

    def __str__(self) -> str:
        return self.root.render()


class MinuteExpressionGenerator:
    def __init__(self, max_depth: int = 5, seed: int | None = None) -> None:
        if max_depth < 2:
            raise ValueError("max_depth must be at least two for a reduction and a feature")
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def generate(self) -> MinuteExpression:
        if self.rng.random() < 0.8:
            child = self._source(2)
            root = MinuteNode("reduce_unary", self.rng.choice(REDUCE_UNARY_OPS), (child,))
        else:
            root = MinuteNode(
                "reduce_binary",
                self.rng.choice(REDUCE_BINARY_OPS),
                (self._minute(2), self._minute(2)),
            )
        return MinuteExpression(root)

    def _source(self, depth: int) -> MinuteNode:
        if depth < self.max_depth and self.rng.random() < 0.30:
            kind = self.rng.choice(("mask_window", "mask_unary", "mask_binary_window", "mask_binary"))
            if kind == "mask_window":
                return MinuteNode(kind, self.rng.choice(MASK_WINDOW_OPS), (self._minute(depth + 1),), self.rng.choice(MINUTE_WINDOWS))
            if kind == "mask_unary":
                return MinuteNode(kind, self.rng.choice(MASK_UNARY_OPS), (self._minute(depth + 1),))
            children = (self._minute(depth + 1), self._minute(depth + 1))
            if kind == "mask_binary_window":
                return MinuteNode(kind, self.rng.choice(MASK_BINARY_WINDOW_OPS), children, self.rng.choice(MINUTE_WINDOWS))
            return MinuteNode(kind, self.rng.choice(MASK_BINARY_OPS), children)
        return self._minute(depth)

    def _minute(self, depth: int) -> MinuteNode:
        if depth >= self.max_depth or self.rng.random() < 0.32:
            return MinuteNode("feature", self.rng.choice(MINUTE_FEATURES))
        kind = self.rng.choice(("minute_unary", "minute_window", "minute_binary"))
        if kind == "minute_unary":
            return MinuteNode(kind, self.rng.choice(MINUTE_UNARY_OPS), (self._minute(depth + 1),))
        if kind == "minute_window":
            return MinuteNode(kind, self.rng.choice(MINUTE_WINDOW_OPS), (self._minute(depth + 1),), self.rng.choice(MINUTE_WINDOWS))
        return MinuteNode(kind, self.rng.choice(MINUTE_BINARY_OPS), (self._minute(depth + 1), self._minute(depth + 1)))


def minute_expression_from_tokens(tokens: Sequence[str]) -> MinuteExpression:
    index = 0

    def parse(expected: str) -> MinuteNode:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("Incomplete minute prefix expression")
        token = tokens[index]
        index += 1
        if expected in {"minute", "source"} and token in MINUTE_FEATURES:
            return MinuteNode("feature", token)
        if expected in {"minute", "source"} and token in MINUTE_UNARY_OPS:
            return MinuteNode("minute_unary", token, (parse("minute"),))
        if expected in {"minute", "source"} and token in MINUTE_WINDOW_OPS:
            window = parse_window(token)
            return MinuteNode("minute_window", token, (parse("minute"),), window)
        if expected in {"minute", "source"} and token in MINUTE_BINARY_OPS:
            return MinuteNode("minute_binary", token, (parse("minute"), parse("minute")))
        if expected == "source" and token in MASK_UNARY_OPS:
            return MinuteNode("mask_unary", token, (parse("minute"),))
        if expected == "source" and token in MASK_WINDOW_OPS:
            window = parse_window(token)
            return MinuteNode("mask_window", token, (parse("minute"),), window)
        if expected == "source" and token in MASK_BINARY_OPS:
            return MinuteNode("mask_binary", token, (parse("minute"), parse("minute")))
        if expected == "source" and token in MASK_BINARY_WINDOW_OPS:
            window = parse_window(token)
            return MinuteNode("mask_binary_window", token, (parse("minute"), parse("minute")), window)
        if expected == "block" and token in REDUCE_UNARY_OPS:
            return MinuteNode("reduce_unary", token, (parse("source"),))
        if expected == "block" and token in REDUCE_BINARY_OPS:
            return MinuteNode("reduce_binary", token, (parse("minute"), parse("minute")))
        raise ValueError(f"Token {token!r} is invalid for minute grammar symbol {expected!r}")

    def parse_window(operator: str) -> int:
        nonlocal index
        if index >= len(tokens) or not tokens[index].startswith("W"):
            raise ValueError(f"{operator} must be followed by a window token")
        window = int(tokens[index][1:])
        index += 1
        if window not in MINUTE_WINDOWS:
            raise ValueError(f"Invalid minute window: {window}")
        return window

    root = parse("block")
    if index != len(tokens):
        raise ValueError(f"Unused tokens after minute expression: {tokens[index:]}")
    return MinuteExpression(root)
