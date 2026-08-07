from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import numpy as np

from src.expression.minute import (
    MASK_BINARY_OPS,
    MASK_BINARY_WINDOW_OPS,
    MASK_UNARY_OPS,
    MASK_WINDOW_OPS,
    MINUTE_BINARY_OPS,
    MINUTE_FEATURES,
    MINUTE_UNARY_OPS,
    MINUTE_WINDOW_OPS,
    REDUCE_BINARY_OPS,
    REDUCE_UNARY_OPS,
    MinuteNode,
)
from src.operators.minute import EPS


NUMPY_MINUTE_EXECUTOR_VERSION = 1
_STORED_FEATURES = set(MINUTE_FEATURES).difference({"logret", "oc_ret"})


class UnsupportedNumpyNode(ValueError):
    """Raised only when a grammar node has no NumPy implementation."""


_SUPPORTED_NAMES = {
    "feature": set(MINUTE_FEATURES),
    "minute_unary": set(MINUTE_UNARY_OPS),
    "minute_window": set(MINUTE_WINDOW_OPS),
    "minute_binary": set(MINUTE_BINARY_OPS),
    "mask_window": set(MASK_WINDOW_OPS),
    "mask_unary": set(MASK_UNARY_OPS),
    "mask_binary_window": set(MASK_BINARY_WINDOW_OPS),
    "mask_binary": set(MASK_BINARY_OPS),
    "reduce_unary": set(REDUCE_UNARY_OPS),
    "reduce_binary": set(REDUCE_BINARY_OPS),
}


def validate_numpy_nodes(nodes: Sequence[MinuteNode]) -> None:
    """Fail before worker launch so an unsupported grammar can use Pandas fallback."""
    def visit(node: MinuteNode) -> None:
        if node.kind not in _SUPPORTED_NAMES or node.name not in _SUPPORTED_NAMES[node.kind]:
            raise UnsupportedNumpyNode(f"Unsupported NumPy node: {node.kind}/{node.name}")
        for child in node.children:
            visit(child)

    for node in nodes:
        if node.kind not in {"reduce_unary", "reduce_binary"}:
            raise UnsupportedNumpyNode("NumPy block roots must be reduce nodes")
        visit(node)


def required_memmap_channels(nodes: Sequence[MinuteNode]) -> tuple[str, ...]:
    """Return the minimal stored channel set needed by a group of blocks."""
    validate_numpy_nodes(nodes)
    features: set[str] = set()

    def visit(node: MinuteNode) -> None:
        if node.kind == "feature":
            if node.name in _STORED_FEATURES:
                features.add(node.name)
            elif node.name == "logret":
                features.add("close")
            elif node.name == "oc_ret":
                features.update(("open", "close"))
            else:
                raise UnsupportedNumpyNode(f"Unsupported minute feature: {node.name}")
            return
        for child in node.children:
            visit(child)

    for node in nodes:
        visit(node)
    return tuple(sorted(features))


def _safe_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.divide(
        left,
        right,
        out=np.full(np.broadcast_shapes(left.shape, right.shape), np.nan, dtype=np.float64),
        where=np.abs(right) > EPS,
    )


class NumpyMinuteBlockExecutor:
    """Execute report minute blocks directly on dense (day, minute, stock) arrays."""

    def __init__(self, mask: np.ndarray, channels: Mapping[str, np.ndarray]) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.channels = channels
        self.cache: dict[MinuteNode, np.ndarray] = {}
        if self.mask.ndim != 3:
            raise ValueError("Minute mask must have shape (day, minute, stock)")

    def execute(self, nodes: Sequence[MinuteNode]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for node in nodes:
            if node.kind not in {"reduce_unary", "reduce_binary"}:
                raise UnsupportedNumpyNode("NumPy block roots must be reduce nodes")
            output[node.render()] = np.asarray(self._node(node), dtype=np.float32)
        return output

    def _node(self, node: MinuteNode) -> np.ndarray:
        cached = self.cache.get(node)
        if cached is not None:
            return cached
        if node.kind == "feature":
            result = self._feature(node.name)
        else:
            values = [self._node(child) for child in node.children]
            if node.kind == "minute_unary":
                result = self._minute_unary(node.name, values[0])
            elif node.kind == "minute_window":
                result = self._minute_window(node.name, values[0], int(node.window))
            elif node.kind == "minute_binary":
                result = self._minute_binary(node.name, values[0], values[1])
            elif node.kind == "mask_window":
                result = self._mask_window(node.name, values[0], int(node.window))
            elif node.kind == "mask_unary":
                result = self._mask_unary(node.name, values[0])
            elif node.kind in {"mask_binary", "mask_binary_window"}:
                result = self._mask_binary(node.name, values[0], values[1], node.window)
            elif node.kind == "reduce_unary":
                result = self._reduce_unary(node.name, values[0])
            elif node.kind == "reduce_binary":
                result = self._reduce_binary(node.name, values[0], values[1])
            else:
                raise UnsupportedNumpyNode(f"Unsupported NumPy node kind: {node.kind}")
        result = np.asarray(result, dtype=np.float64)
        self.cache[node] = result
        return result

    def _feature(self, name: str) -> np.ndarray:
        if name in self.channels:
            values = np.asarray(self.channels[name], dtype=np.float64)
        elif name == "logret":
            close = self._feature("close")
            ratio = _safe_div(close, self._shift(close, 1))
            values = np.log(np.where(ratio > 0, ratio, np.nan))
        elif name == "oc_ret":
            values = _safe_div(self._feature("close"), self._feature("open")) - 1.0
        else:
            raise UnsupportedNumpyNode(f"Missing minute channel: {name}")
        if values.shape != self.mask.shape:
            raise ValueError(f"Channel {name} has shape {values.shape}, expected {self.mask.shape}")
        return np.where(self.mask, values, np.nan)

    def _groups(self, values: np.ndarray) -> np.ndarray:
        return values.transpose(0, 2, 1).reshape(-1, values.shape[1])

    def _restore(self, groups: np.ndarray) -> np.ndarray:
        days, minutes, stocks = self.mask.shape
        return groups.reshape(days, stocks, minutes).transpose(0, 2, 1)

    def _shift(self, values: np.ndarray, periods: int) -> np.ndarray:
        groups, masks = self._groups(values), self._groups(self.mask)
        output = np.full_like(groups, np.nan, dtype=np.float64)
        complete = masks.all(axis=1)
        if periods < groups.shape[1] and complete.any():
            output[complete, periods:] = groups[complete, :-periods]
        for index in np.flatnonzero(~complete):
            positions = np.flatnonzero(masks[index])
            if len(positions) > periods:
                output[index, positions[periods:]] = groups[index, positions[:-periods]]
        return self._restore(output)

    def _minute_unary(self, name: str, values: np.ndarray) -> np.ndarray:
        if name in {"m_ret", "m_logret"}:
            ratio = _safe_div(values, self._shift(values, 1))
            return ratio - 1.0 if name == "m_ret" else np.log(np.where(ratio > 0, ratio, np.nan))
        if name == "m_abs":
            return np.abs(values)
        if name == "m_sign":
            return np.sign(values)
        if name == "m_log":
            return np.log(np.abs(values) + EPS)
        groups = self._groups(values)
        output = np.full_like(groups, np.nan, dtype=np.float64)
        for index, row in enumerate(groups):
            valid = np.isfinite(row)
            clean = row[valid]
            if name == "m_rank":
                output[index, valid] = self._rank(clean, average=True) / len(clean) if len(clean) else np.nan
            elif name == "m_zscore" and len(clean):
                std = clean.std(ddof=1) if len(clean) > 1 else np.nan
                if std > EPS:
                    output[index, valid] = (clean - clean.mean()) / std
            elif name not in {"m_rank", "m_zscore"}:
                raise UnsupportedNumpyNode(f"Unsupported minute unary: {name}")
        return self._restore(output)

    @staticmethod
    def _rank(values: np.ndarray, average: bool) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=np.float64)
        if not average:
            ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
            return ranks
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return ranks

    def _minute_window(self, name: str, values: np.ndarray, window: int) -> np.ndarray:
        if name == "m_delay":
            return self._shift(values, window)
        if name == "m_delta":
            return values - self._shift(values, window)
        if name not in {"m_ma", "m_std"}:
            raise UnsupportedNumpyNode(f"Unsupported minute window: {name}")
        groups, masks = self._groups(values), self._groups(self.mask)
        output = np.full_like(groups, np.nan, dtype=np.float64)
        minimum = max(2, window // 2)
        for index, row in enumerate(groups):
            positions = np.flatnonzero(masks[index])
            compact = row[positions]
            finite = np.isfinite(compact)
            sums = np.r_[0.0, np.cumsum(np.where(finite, compact, 0.0))]
            sums2 = np.r_[0.0, np.cumsum(np.where(finite, compact * compact, 0.0))]
            counts = np.r_[0, np.cumsum(finite)]
            right = np.arange(1, len(compact) + 1)
            left = np.maximum(0, right - window)
            count = counts[right] - counts[left]
            total = sums[right] - sums[left]
            valid_window = count >= minimum
            result = np.full(len(compact), np.nan)
            if name == "m_ma":
                result[valid_window] = total[valid_window] / count[valid_window]
            else:
                variance = np.full(len(compact), np.nan)
                variance[valid_window] = (
                    (sums2[right] - sums2[left])[valid_window]
                    - total[valid_window] ** 2 / count[valid_window]
                ) / (count[valid_window] - 1)
                result[valid_window] = np.sqrt(np.maximum(variance[valid_window], 0.0))
            output[index, positions] = result
        return self._restore(output)

    @staticmethod
    def _minute_binary(name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if name == "m_add": return left + right
        if name == "m_sub": return left - right
        if name == "m_mul": return left * right
        if name == "m_div": return _safe_div(left, right)
        raise UnsupportedNumpyNode(f"Unsupported minute binary: {name}")

    def _position_layout(self) -> tuple[np.ndarray, np.ndarray]:
        groups = self._groups(self.mask)
        ordinal = np.cumsum(groups, axis=1) - 1
        sizes = groups.sum(axis=1)[:, None]
        return groups, (ordinal, sizes)

    def _rank_mask(self, values: np.ndarray, window: int, largest: bool) -> np.ndarray:
        groups = self._groups(values)
        selected = np.zeros_like(groups, dtype=bool)
        for index, row in enumerate(groups):
            valid = np.isfinite(row)
            clean = row[valid]
            if len(clean):
                ranks = self._rank(-clean if largest else clean, average=False)
                selected[index, np.flatnonzero(valid)] = ranks <= window
        return self._restore(selected)

    def _mask_window(self, name: str, values: np.ndarray, window: int) -> np.ndarray:
        groups, (ordinal, sizes) = self._position_layout()
        if name == "m_head": selected = groups & (ordinal < window)
        elif name == "m_tail": selected = groups & (ordinal >= np.maximum(sizes - window, 0))
        elif name == "m_mid":
            start = np.maximum(sizes - window, 0) // 2
            selected = groups & (ordinal >= start) & (ordinal < start + window)
        elif name in {"m_top", "m_bot"}:
            return np.where(self._rank_mask(values, window, name == "m_top"), values, np.nan)
        elif name == "m_xtreme":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(self._groups(values), axis=1)[:, None]
            deviation = self._restore(np.abs(self._groups(values) - median))
            return np.where(self._rank_mask(deviation, window, True), values, np.nan)
        else:
            raise UnsupportedNumpyNode(f"Unsupported mask window: {name}")
        return np.where(self._restore(selected), values, np.nan)

    def _mask_unary(self, name: str, values: np.ndarray) -> np.ndarray:
        groups = self._groups(values)
        if name in {"m_above", "m_below"}:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(groups, axis=1)[:, None]
            selected = groups > median if name == "m_above" else groups < median
        elif name in {"m_inner", "m_outer"}:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q1 = np.nanquantile(groups, 0.25, axis=1)[:, None]
                q3 = np.nanquantile(groups, 0.75, axis=1)[:, None]
            selected = (groups >= q1) & (groups <= q3)
            if name == "m_outer": selected = (groups < q1) | (groups > q3)
        else:
            raise UnsupportedNumpyNode(f"Unsupported mask unary: {name}")
        return self._restore(np.where(selected, groups, np.nan))

    def _mask_binary(
        self, name: str, values: np.ndarray, condition: np.ndarray, window: int | None
    ) -> np.ndarray:
        if name in {"m_at_top", "m_at_bot"}:
            if window is None: raise UnsupportedNumpyNode(f"{name} requires a window")
            selected = self._rank_mask(condition, int(window), name == "m_at_top")
        elif name == "m_when_pos":
            selected = condition > 0
        elif name == "m_when_gt":
            groups = self._groups(condition)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median = np.nanmedian(groups, axis=1)[:, None]
            selected = self._restore(groups > median)
        else:
            raise UnsupportedNumpyNode(f"Unsupported binary mask: {name}")
        return np.where(selected, values, np.nan)

    def _reduce_unary(self, name: str, values: np.ndarray) -> np.ndarray:
        groups = self._groups(values)
        source_groups = self._groups(self.mask)
        count = np.isfinite(groups).sum(axis=1)
        has_source = source_groups.any(axis=1)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            if name == "r_mean": output = np.nanmean(groups, axis=1)
            elif name == "r_std": output = np.nanstd(groups, axis=1, ddof=1)
            elif name == "r_sum": output = np.nansum(groups, axis=1); output[~has_source] = np.nan
            elif name == "r_max": output = np.nanmax(groups, axis=1)
            elif name == "r_min": output = np.nanmin(groups, axis=1)
            elif name == "r_median": output = np.nanmedian(groups, axis=1)
            elif name in {"r_first", "r_last"}:
                finite = np.isfinite(groups)
                position = finite.argmax(axis=1) if name == "r_first" else groups.shape[1] - 1 - finite[:, ::-1].argmax(axis=1)
                output = groups[np.arange(len(groups)), position]
                output[count == 0] = np.nan
            elif name in {"r_skew", "r_kurt", "r_slope", "r_rsquare", "r_argmax"}:
                output = self._complex_reduce(name, groups, source_groups)
            else:
                raise UnsupportedNumpyNode(f"Unsupported unary reduction: {name}")
        days, _, stocks = self.mask.shape
        return output.reshape(days, stocks)

    @staticmethod
    def _complex_reduce(name: str, groups: np.ndarray, source: np.ndarray) -> np.ndarray:
        output = np.full(len(groups), np.nan)
        for index, row in enumerate(groups):
            positions = np.flatnonzero(np.isfinite(row))
            clean = row[positions]
            n = len(clean)
            if not n: continue
            if name == "r_argmax":
                source_ordinal = np.cumsum(source[index]) - 1
                output[index] = source_ordinal[positions[np.argmax(clean)]] / max(1, int(source[index].sum()) - 1)
            elif name == "r_skew" and n >= 3:
                centered = clean - clean.mean(); m2 = np.mean(centered ** 2)
                output[index] = 0.0 if m2 <= EPS else np.sqrt(n * (n - 1)) / (n - 2) * np.mean(centered ** 3) / m2 ** 1.5
            elif name == "r_kurt" and n >= 4:
                centered = clean - clean.mean(); m2 = np.mean(centered ** 2)
                if m2 <= EPS: output[index] = 0.0
                else:
                    g2 = np.mean(centered ** 4) / (m2 * m2) - 3.0
                    output[index] = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)
            elif name in {"r_slope", "r_rsquare"} and n >= 2:
                x = np.arange(n, dtype=float); xc = x - x.mean(); yc = clean - clean.mean()
                denominator = float(xc @ xc)
                slope = float(xc @ yc / denominator) if denominator > EPS else np.nan
                if name == "r_slope": output[index] = slope
                else:
                    total = float(yc @ yc)
                    residual = float(np.square(clean - (clean.mean() + slope * xc)).sum())
                    output[index] = 1.0 - residual / total if total > EPS else np.nan
        return output

    def _reduce_binary(self, name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        xgroups, ygroups = self._groups(left), self._groups(right)
        output = np.full(len(xgroups), np.nan)
        for index, (x, y) in enumerate(zip(xgroups, ygroups)):
            valid = np.isfinite(x) & np.isfinite(y); x, y = x[valid], y[valid]
            if name == "r_wmean":
                denominator = float(y.sum())
                if abs(denominator) > EPS: output[index] = float(x @ y / denominator)
                continue
            if len(x) < 2: continue
            xc, yc = x - x.mean(), y - y.mean(); cross = float(xc @ yc)
            if name == "r_cov": output[index] = cross / (len(x) - 1)
            elif name == "r_corr":
                denominator = float(np.sqrt((xc @ xc) * (yc @ yc)))
                if denominator > EPS: output[index] = cross / denominator
            else: raise UnsupportedNumpyNode(f"Unsupported binary reduction: {name}")
        days, _, stocks = self.mask.shape
        return output.reshape(days, stocks)
