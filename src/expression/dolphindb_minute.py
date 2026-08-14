from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .minute import MinuteNode


class DolphinDBPushdownUnsupported(ValueError):
    pass


@dataclass(frozen=True)
class CompiledMinuteBlocks:
    script: str
    aliases: dict[str, str]


class DolphinDBMinuteCompiler:
    """Compile report-style intraday blocks into one partition-pruned DDB query."""

    _RAW = {"open", "high", "low", "close", "amount"}
    _REDUCE_UNARY = {
        "r_mean": "avg", "r_std": "std", "r_sum": "sum", "r_max": "max",
        "r_min": "min", "r_median": "med", "r_first": "firstNot", "r_last": "lastNot",
    }
    _UNSUPPORTED_REDUCERS = {"r_skew", "r_kurt", "r_slope", "r_rsquare", "r_argmax"}

    def __init__(self, table_expression: str) -> None:
        self.table_expression = table_expression

    def supports(self, node: MinuteNode) -> bool:
        try:
            self._block_parts(node)
            return True
        except DolphinDBPushdownUnsupported:
            return False

    def compile(
        self,
        nodes: Sequence[MinuteNode],
        start: pd.Timestamp,
        end: pd.Timestamp,
        *,
        time_filter_sql: str | None = None,
    ) -> CompiledMinuteBlocks:
        aliases: dict[str, str] = {}
        vector_columns: list[str] = []
        reductions: list[str] = []
        for index, node in enumerate(nodes):
            rendered = node.render()
            vectors, reducer = self._block_parts(node)
            output_alias = "ddb_" + hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:16]
            vector_aliases = []
            for child_index, vector in enumerate(vectors):
                vector_alias = f"am_vec_{index:03d}_{child_index}"
                vector_columns.append(f"{vector} as {vector_alias}")
                vector_aliases.append(vector_alias)
            reductions.append(f"{reducer(vector_aliases)} as {output_alias}")
            aliases[rendered] = output_alias
        if not aliases:
            raise ValueError("At least one minute block is required for DolphinDB pushdown")
        columns = "date, sym, time, open, high, low, close, volume, amount, tradeCount"
        start_literal, end_literal = start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
        time_filter = f", ({time_filter_sql})" if time_filter_sql else ""
        script = (
            "// ALPHAMINING_MINUTE_PUSHDOWN_V1\n"
            f"am_source = select {columns} from {self.table_expression} "
            f"where date >= {start_literal}, date <= {end_literal}{time_filter} "
            "order by date, sym, time;\n"
            "am_vectors = select date, sym, time, " + ", ".join(vector_columns) +
            " from am_source context by date, sym csort time;\n"
            "select " + ", ".join(reductions) +
            " from am_vectors group by date, sym order by date, sym"
        )
        return CompiledMinuteBlocks(script=script, aliases=aliases)

    def _block_parts(self, node: MinuteNode):
        if node.kind == "reduce_unary":
            if node.name in self._UNSUPPORTED_REDUCERS:
                raise DolphinDBPushdownUnsupported(node.name)
            function = self._REDUCE_UNARY.get(node.name)
            if function is None:
                raise DolphinDBPushdownUnsupported(node.name)
            vector = self._vector(node.children[0])
            return [vector], lambda columns: f"{function}({columns[0]})"
        if node.kind == "reduce_binary":
            left, right = (self._vector(child) for child in node.children)
            if node.name == "r_corr":
                return [left, right], lambda columns: f"corr({columns[0]}, {columns[1]})"
            if node.name == "r_cov":
                return [left, right], lambda columns: f"covar({columns[0]}, {columns[1]})"
            if node.name == "r_wmean":
                return [left, right], lambda columns: f"wavg({columns[0]}, {columns[1]})"
            raise DolphinDBPushdownUnsupported(node.name)
        raise DolphinDBPushdownUnsupported(f"not a reduce block: {node.kind}")

    def _vector(self, node: MinuteNode) -> str:
        if node.kind == "feature":
            return self._feature(node.name)
        children = [self._vector(child) for child in node.children]
        if node.kind == "minute_unary":
            child = children[0]
            ratio = self._safe_div(child, f"move({child}, 1)")
            mapping = {
                "m_ret": ratio + " - 1.0",
                "m_logret": f"iif(({ratio}) > 0, log({ratio}), double(NULL))",
                "m_rank": f"rank(X={child}, tiesMethod='average', percent=true)",
                "m_zscore": self._safe_div(f"({child} - avg({child}))", f"std({child})"),
                "m_abs": f"abs({child})", "m_sign": f"sign({child})",
                "m_log": f"log(abs({child}) + 1e-12)",
            }
            return mapping[node.name]
        if node.kind == "minute_window":
            child, window = children[0], int(node.window or 0)
            minimum = max(2, window // 2)
            mapping = {
                "m_delay": f"move({child}, {window})",
                "m_delta": f"({child} - move({child}, {window}))",
                "m_ma": f"mavg({child}, {window}, {minimum})",
                "m_std": f"mstd({child}, {window}, {minimum})",
            }
            return mapping[node.name]
        if node.kind == "minute_binary":
            left, right = children
            mapping = {
                "m_add": f"({left} + {right})", "m_sub": f"({left} - {right})",
                "m_mul": f"({left} * {right})", "m_div": self._safe_div(left, right),
            }
            return mapping[node.name]
        if node.kind == "mask_window":
            child, window = children[0], int(node.window or 0)
            if node.name in {"m_head", "m_tail", "m_mid"}:
                raise DolphinDBPushdownUnsupported(node.name)
            if node.name == "m_top":
                condition = f"rank(X={child}, ascending=false, tiesMethod='first') < {window}"
            elif node.name == "m_bot":
                condition = f"rank(X={child}, ascending=true, tiesMethod='first') < {window}"
            elif node.name == "m_xtreme":
                distance = f"abs({child} - med({child}))"
                condition = f"rank(X={distance}, ascending=false, tiesMethod='first') < {window}"
            else:
                raise DolphinDBPushdownUnsupported(node.name)
            return self._mask(child, condition)
        if node.kind == "mask_unary":
            child = children[0]
            if node.name == "m_above":
                condition = f"{child} > med({child})"
            elif node.name == "m_below":
                condition = f"{child} < med({child})"
            elif node.name == "m_inner":
                condition = f"{child} >= percentile({child}, 25) and {child} <= percentile({child}, 75)"
            elif node.name == "m_outer":
                condition = f"{child} < percentile({child}, 25) or {child} > percentile({child}, 75)"
            else:
                raise DolphinDBPushdownUnsupported(node.name)
            return self._mask(child, condition)
        if node.kind in {"mask_binary", "mask_binary_window"}:
            child, condition_value = children
            if node.name == "m_when_pos":
                condition = f"{condition_value} > 0"
            elif node.name == "m_when_gt":
                condition = f"{condition_value} > med({condition_value})"
            elif node.name in {"m_at_top", "m_at_bot"}:
                ascending = "false" if node.name == "m_at_top" else "true"
                condition = (
                    f"rank(X={condition_value}, ascending={ascending}, tiesMethod='first') "
                    f"< {int(node.window or 0)}"
                )
            else:
                raise DolphinDBPushdownUnsupported(node.name)
            return self._mask(child, condition)
        raise DolphinDBPushdownUnsupported(node.kind)

    def _feature(self, name: str) -> str:
        if name in self._RAW:
            return name
        if name == "vol":
            return "volume"
        ret = self._safe_div("close", "move(close, 1)") + " - 1.0"
        features = {
            "ret": ret,
            "vwap": self._safe_div("amount", "volume"),
            "hl_pct": self._safe_div("(high - low)", "abs(close)"),
            "bar_pos": self._safe_div("(close - low)", "(high - low)"),
            "amihud": self._safe_div(f"abs({ret})", "abs(amount)"),
            "rv": f"pow(({ret}), 2)",
            "signed_vol": f"sign({ret}) * volume",
            "signed_amt": f"sign({ret}) * amount",
            "typical": "(high + low + close) / 3.0",
            "vwap_cum": self._safe_div("cumsum(amount)", "cumsum(volume)"),
            "twap": "cumavg(close)",
            "obv": f"cumsum(sign({ret}) * volume)",
            "pvt": f"cumsum(({ret}) * volume)",
            "logret": (
                f"iif(({self._safe_div('close', 'move(close, 1)')}) > 0, "
                f"log({self._safe_div('close', 'move(close, 1)')}), double(NULL))"
            ),
            "oc_ret": self._safe_div("close", "open") + " - 1.0",
        }
        try:
            return features[name]
        except KeyError as error:
            raise DolphinDBPushdownUnsupported(name) from error

    @staticmethod
    def _safe_div(left: str, right: str) -> str:
        return f"iif(abs({right}) > 1e-12, ({left}) / ({right}), double(NULL))"

    @staticmethod
    def _mask(value: str, condition: str) -> str:
        return f"iif(({condition}), {value}, double(NULL))"
