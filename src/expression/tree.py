from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator, Sequence

import pandas as pd

from src.operators import apply_binary, apply_cross_sectional, apply_time_series, apply_unary


FEATURES = ("open", "high", "low", "close", "volume", "vwap")
WINDOWS = (5, 10, 20, 40, 60)
UNARY_OPS = ("log", "abs", "neg", "sqrt", "tanh")
BINARY_OPS = ("add", "sub", "mul", "div")
TS_UNARY_OPS = (
    "ts_mean", "ts_std", "ts_rank", "ts_delay", "ts_delta",
    "ts_sum", "ts_max", "ts_min", "ts_zscore",
)
CS_OPS = ("cs_rank", "cs_zscore", "cs_demean")


@dataclass(frozen=True)
class Node:
    kind: str
    name: str
    children: tuple["Node", ...] = field(default_factory=tuple)
    window: int | None = None

    def __post_init__(self) -> None:
        arity = {"feature": 0, "unary": 1, "binary": 2, "ts": 1, "cs": 1}
        if self.kind not in arity:
            raise ValueError(f"Unknown node kind: {self.kind}")
        if len(self.children) != arity[self.kind]:
            raise ValueError(f"{self.kind} requires {arity[self.kind]} children")
        if self.kind == "ts" and self.window not in WINDOWS:
            raise ValueError(f"Invalid time-series window: {self.window}")

    def render(self) -> str:
        if self.kind == "feature":
            return self.name
        if self.kind == "binary":
            return f"{self.name}({self.children[0].render()},{self.children[1].render()})"
        if self.kind == "ts":
            return f"{self.name}({self.children[0].render()},{self.window})"
        return f"{self.name}({self.children[0].render()})"

    def complexity(self) -> int:
        return 1 + sum(child.complexity() for child in self.children)

    def depth(self) -> int:
        return 1 if not self.children else 1 + max(child.depth() for child in self.children)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "window": self.window,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Node":
        return cls(
            kind=value["kind"],
            name=value["name"],
            window=value.get("window"),
            children=tuple(cls.from_dict(child) for child in value.get("children", [])),
        )

    def prefix_tokens(self) -> Iterator[str]:
        yield self.name if self.kind != "ts" else self.name
        if self.kind == "ts":
            yield f"W{self.window}"
        for child in self.children:
            yield from child.prefix_tokens()


@dataclass(frozen=True)
class Expression:
    root: Node
    FEATURES: ClassVar[tuple[str, ...]] = FEATURES

    @classmethod
    def generate(cls, max_depth: int = 4, seed: int | None = None) -> "Expression":
        return ExpressionGenerator(max_depth=max_depth, seed=seed).generate()

    def execute(self, data: pd.DataFrame) -> pd.Series:
        required = {"date", "code", *FEATURES}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Expression data is missing columns: {missing}")
        if not data[["code", "date"]].equals(
            data.sort_values(["code", "date"], kind="stable")[["code", "date"]]
        ):
            ordered = data.sort_values(["code", "date"], kind="stable")
            values = self._execute_node(self.root, ordered)
            return values.reindex(data.index)
        return self._execute_node(self.root, data)

    def _execute_node(self, node: Node, data: pd.DataFrame) -> pd.Series:
        if node.kind == "feature":
            return data[node.name].astype(float)
        values = [self._execute_node(child, data) for child in node.children]
        if node.kind == "unary":
            return apply_unary(node.name, values[0])
        if node.kind == "binary":
            return apply_binary(node.name, values[0], values[1])
        if node.kind == "ts":
            return apply_time_series(node.name, values[0], data["code"], int(node.window))
        if node.kind == "cs":
            return apply_cross_sectional(node.name, values[0], data["date"])
        raise AssertionError(node.kind)

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


class ExpressionGenerator:
    def __init__(self, max_depth: int = 4, seed: int | None = None) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self.max_depth = max_depth
        self.rng = random.Random(seed)

    def generate(self) -> Expression:
        return Expression(self._node(depth=1))

    def _node(self, depth: int) -> Node:
        if depth >= self.max_depth or self.rng.random() < 0.28:
            return Node("feature", self.rng.choice(FEATURES))
        kind = self.rng.choices(("unary", "binary", "ts", "cs"), weights=(2, 3, 4, 2), k=1)[0]
        if kind == "unary":
            return Node(kind, self.rng.choice(UNARY_OPS), (self._node(depth + 1),))
        if kind == "binary":
            return Node(kind, self.rng.choice(BINARY_OPS), (self._node(depth + 1), self._node(depth + 1)))
        if kind == "ts":
            return Node(kind, self.rng.choice(TS_UNARY_OPS), (self._node(depth + 1),), self.rng.choice(WINDOWS))
        return Node(kind, self.rng.choice(CS_OPS), (self._node(depth + 1),))


def expression_from_tokens(tokens: Sequence[str]) -> Expression:
    index = 0

    def parse() -> Node:
        nonlocal index
        if index >= len(tokens):
            raise ValueError("Incomplete prefix expression")
        token = tokens[index]
        index += 1
        if token in FEATURES:
            return Node("feature", token)
        if token in UNARY_OPS:
            return Node("unary", token, (parse(),))
        if token in BINARY_OPS:
            return Node("binary", token, (parse(), parse()))
        if token in CS_OPS:
            return Node("cs", token, (parse(),))
        if token in TS_UNARY_OPS:
            if index >= len(tokens) or not tokens[index].startswith("W"):
                raise ValueError(f"{token} must be followed by a window token")
            window = int(tokens[index][1:])
            index += 1
            return Node("ts", token, (parse(),), window)
        raise ValueError(f"Unknown expression token: {token}")

    root = parse()
    if index != len(tokens):
        raise ValueError(f"Unused tokens after expression: {tokens[index:]}")
    return Expression(root)

