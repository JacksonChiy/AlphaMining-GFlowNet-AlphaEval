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
        for index in np.flatnonzero(masks.any(axis=1) & ~complete):
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
        if name == "m_zscore":
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(groups, axis=1, keepdims=True)
                std = np.nanstd(groups, axis=1, ddof=1, keepdims=True)
            output = np.divide(
                groups - mean,
                std,
                out=np.full_like(groups, np.nan, dtype=np.float64),
                where=std > EPS,
            )
            return self._restore(output)
        if name != "m_rank":
            raise UnsupportedNumpyNode(f"Unsupported minute unary: {name}")
        output = np.full_like(groups, np.nan, dtype=np.float64)
        finite = np.isfinite(groups)
        active = finite.any(axis=1)
        active_index = np.flatnonzero(active)
        active_groups = groups[active]
        active_finite = finite[active]
        count = active_finite.sum(axis=1)
        order = np.argsort(
            np.where(active_finite, active_groups, np.inf), axis=1, kind="stable"
        )
        sorted_values = np.take_along_axis(active_groups, order, axis=1)
        sorted_finite = np.take_along_axis(active_finite, order, axis=1)
        has_ties = np.any(
            (np.diff(sorted_values, axis=1) == 0)
            & sorted_finite[:, 1:] & sorted_finite[:, :-1],
            axis=1,
        )
        fast = ~has_ties
        if fast.any():
            fast_order = order[fast]
            ranks = np.empty_like(fast_order, dtype=np.float64)
            np.put_along_axis(
                ranks,
                fast_order,
                np.broadcast_to(
                    np.arange(1, groups.shape[1] + 1, dtype=np.float64),
                    fast_order.shape,
                ),
                axis=1,
            )
            fast_output = ranks / count[fast, None]
            fast_output[~active_finite[fast]] = np.nan
            output[active_index[fast]] = fast_output
        for active_row in np.flatnonzero(has_ties):
            index = active_index[active_row]
            row = active_groups[active_row]
            valid = np.isfinite(row)
            clean = row[valid]
            output[index, valid] = self._rank(clean, average=True) / len(clean)
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
        complete = masks.all(axis=1)
        if complete.any():
            output[complete] = self._rolling_dense(
                groups[complete], window, minimum, standard_deviation=name == "m_std"
            )
        for index in np.flatnonzero(masks.any(axis=1) & ~complete):
            row = groups[index]
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
    def _rolling_dense(
        groups: np.ndarray,
        window: int,
        minimum: int,
        standard_deviation: bool,
    ) -> np.ndarray:
        """Vectorized Pandas-compatible rolling path for complete minute grids."""
        finite = np.isfinite(groups)
        clean = np.where(finite, groups, 0.0)
        zeros = np.zeros((len(groups), 1), dtype=np.float64)
        sums = np.concatenate((zeros, np.cumsum(clean, axis=1)), axis=1)
        counts = np.concatenate(
            (np.zeros((len(groups), 1), dtype=np.int32), np.cumsum(finite, axis=1)),
            axis=1,
        )
        right = np.arange(1, groups.shape[1] + 1)
        left = np.maximum(0, right - window)
        count = counts[:, right] - counts[:, left]
        total = sums[:, right] - sums[:, left]
        valid = count >= minimum
        output = np.full_like(groups, np.nan, dtype=np.float64)
        if not standard_deviation:
            np.divide(total, count, out=output, where=valid)
            return output
        sums2 = np.concatenate(
            (zeros, np.cumsum(clean * clean, axis=1)), axis=1
        )
        square_total = sums2[:, right] - sums2[:, left]
        variance = np.divide(
            square_total - np.divide(
                total * total,
                count,
                out=np.zeros_like(total),
                where=count > 0,
            ),
            count - 1,
            out=np.full_like(total, np.nan),
            where=valid,
        )
        output[valid] = np.sqrt(np.maximum(variance[valid], 0.0))
        return output

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
        finite = np.isfinite(groups)
        active = finite.any(axis=1)
        selected = np.zeros_like(groups, dtype=bool)
        if not active.any():
            return self._restore(selected)
        active_groups = groups[active]
        active_finite = finite[active]
        key = -active_groups if largest else active_groups
        order = np.argsort(
            np.where(active_finite, key, np.inf), axis=1, kind="stable"
        )
        ranks = np.empty_like(order, dtype=np.int32)
        np.put_along_axis(
            ranks,
            order,
            np.broadcast_to(
                np.arange(1, groups.shape[1] + 1, dtype=np.int32), order.shape
            ),
            axis=1,
        )
        selected[active] = active_finite & (ranks <= window)
        return self._restore(selected)

    @staticmethod
    def _row_quantile(groups: np.ndarray, quantile: float) -> np.ndarray:
        output = np.full((len(groups), 1), np.nan, dtype=np.float64)
        active = np.isfinite(groups).any(axis=1)
        if active.any():
            output[active, 0] = np.nanquantile(groups[active], quantile, axis=1)
        return output

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
            median = self._row_quantile(self._groups(values), 0.5)
            deviation = self._restore(np.abs(self._groups(values) - median))
            return np.where(self._rank_mask(deviation, window, True), values, np.nan)
        else:
            raise UnsupportedNumpyNode(f"Unsupported mask window: {name}")
        return np.where(self._restore(selected), values, np.nan)

    def _mask_unary(self, name: str, values: np.ndarray) -> np.ndarray:
        groups = self._groups(values)
        if name in {"m_above", "m_below"}:
            median = self._row_quantile(groups, 0.5)
            selected = groups > median if name == "m_above" else groups < median
        elif name in {"m_inner", "m_outer"}:
            q1 = self._row_quantile(groups, 0.25)
            q3 = self._row_quantile(groups, 0.75)
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
            median = self._row_quantile(groups, 0.5)
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
        finite = np.isfinite(groups)
        count = finite.sum(axis=1).astype(np.float64)
        clean = np.where(finite, groups, 0.0)
        output = np.full(len(groups), np.nan)
        if name == "r_argmax":
            position = np.argmax(np.where(finite, groups, -np.inf), axis=1)
            ordinal = np.cumsum(source, axis=1) - 1
            numerator = ordinal[np.arange(len(groups)), position]
            denominator = np.maximum(1, source.sum(axis=1) - 1)
            output[count > 0] = numerator[count > 0] / denominator[count > 0]
            return output
        total = clean.sum(axis=1)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        centered = np.where(finite, groups - mean[:, None], 0.0)
        centered2 = centered * centered
        second_sum = centered2.sum(axis=1)
        m2 = np.divide(second_sum, count, out=np.zeros_like(total), where=count > 0)
        if name == "r_skew":
            third_sum = (centered2 * centered).sum(axis=1)
            m3 = np.divide(third_sum, count, out=np.zeros_like(total), where=count > 0)
            eligible = count >= 3
            constant = eligible & (m2 <= EPS)
            output[constant] = 0.0
            variable = eligible & (m2 > EPS)
            output[variable] = (
                np.sqrt(count[variable] * (count[variable] - 1)) /
                (count[variable] - 2) * m3[variable] / m2[variable] ** 1.5
            )
            return output
        if name == "r_kurt":
            fourth_sum = (centered2 * centered2).sum(axis=1)
            m4 = np.divide(fourth_sum, count, out=np.zeros_like(total), where=count > 0)
            eligible = count >= 4
            constant = eligible & (m2 <= EPS)
            output[constant] = 0.0
            variable = eligible & (m2 > EPS)
            g2 = m4[variable] / (m2[variable] ** 2) - 3.0
            n = count[variable]
            output[variable] = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0)
            return output
        if name in {"r_slope", "r_rsquare"}:
            x = np.cumsum(finite, axis=1) - 1
            x = np.where(finite, x, 0.0)
            x_centered = np.where(finite, x - (count[:, None] - 1) / 2.0, 0.0)
            cross = (x_centered * centered).sum(axis=1)
            ssx = (x_centered * x_centered).sum(axis=1)
            slope = np.divide(cross, ssx, out=np.full_like(cross, np.nan), where=ssx > EPS)
            eligible = count >= 2
            if name == "r_slope":
                output[eligible] = slope[eligible]
            else:
                ssy = second_sum
                valid = eligible & (ssy > EPS) & (ssx > EPS)
                output[valid] = cross[valid] ** 2 / (ssx[valid] * ssy[valid])
            return output
        raise UnsupportedNumpyNode(f"Unsupported complex reduction: {name}")

    def _reduce_binary(self, name: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        xgroups, ygroups = self._groups(left), self._groups(right)
        valid = np.isfinite(xgroups) & np.isfinite(ygroups)
        count = valid.sum(axis=1).astype(np.float64)
        x = np.where(valid, xgroups, 0.0)
        y = np.where(valid, ygroups, 0.0)
        sum_x, sum_y = x.sum(axis=1), y.sum(axis=1)
        product_sum = (x * y).sum(axis=1)
        output = np.full(len(xgroups), np.nan)
        if name == "r_wmean":
            np.divide(product_sum, sum_y, out=output, where=np.abs(sum_y) > EPS)
        elif name in {"r_cov", "r_corr"}:
            mean_x = np.divide(
                sum_x, count, out=np.zeros_like(sum_x), where=count > 0
            )
            mean_y = np.divide(
                sum_y, count, out=np.zeros_like(sum_y), where=count > 0
            )
            x_centered = np.where(valid, xgroups - mean_x[:, None], 0.0)
            y_centered = np.where(valid, ygroups - mean_y[:, None], 0.0)
            cross = (x_centered * y_centered).sum(axis=1)
            eligible = count >= 2
            if name == "r_cov":
                np.divide(cross, count - 1, out=output, where=eligible)
            else:
                ssx = (x_centered * x_centered).sum(axis=1)
                ssy = (y_centered * y_centered).sum(axis=1)
                denominator = np.sqrt(np.maximum(ssx * ssy, 0.0))
                np.divide(
                    cross,
                    denominator,
                    out=output,
                    where=eligible & (denominator > EPS),
                )
        else:
            raise UnsupportedNumpyNode(f"Unsupported binary reduction: {name}")
        days, _, stocks = self.mask.shape
        return output.reshape(days, stocks)
