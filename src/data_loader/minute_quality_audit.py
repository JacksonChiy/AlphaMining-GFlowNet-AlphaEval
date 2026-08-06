from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dolphindb_minute import DolphinDBMinuteLoader


DEFAULT_MINUTE_SESSIONS = (
    ("09:31:00", "11:30:00"),
    ("13:01:00", "15:00:00"),
)
DEFAULT_MINUTE_EXTRA_TIMES = ("09:25:00",)


def _time_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        value = value.time()
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M:%S")
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Cannot normalize minute time value: {value!r}")
    return parsed.strftime("%H:%M:%S")


def build_expected_minute_grid(
    sessions: Sequence[Sequence[str]],
    extra_times: Sequence[str] = (),
) -> tuple[str, ...]:
    values: list[str] = [_time_key(value) for value in extra_times]
    for item in sessions:
        if len(item) != 2:
            raise ValueError("Each minute session must contain exactly [start, end]")
        start = pd.Timestamp(f"2000-01-01 {item[0]}")
        end = pd.Timestamp(f"2000-01-01 {item[1]}")
        if start > end:
            raise ValueError(f"Minute session start is later than end: {item}")
        values.extend(value.strftime("%H:%M:%S") for value in pd.date_range(start, end, freq="min"))
    if len(values) != len(set(values)):
        raise ValueError("Configured minute sessions overlap")
    return tuple(sorted(values))


@dataclass(frozen=True)
class MinuteQualityAuditConfig:
    output_dir: Path
    expected_minutes: int = 241
    sessions: tuple[tuple[str, str], ...] = DEFAULT_MINUTE_SESSIONS
    extra_times: tuple[str, ...] = DEFAULT_MINUTE_EXTRA_TIMES
    chunk_days: int = 20
    scope: str = "grid"

    def __post_init__(self) -> None:
        if self.expected_minutes < 1 or self.chunk_days < 1:
            raise ValueError("Audit expected_minutes and chunk_days must be positive")
        if self.scope not in {"grid", "full"}:
            raise ValueError("Audit scope must be 'grid' or 'full'")
        grid = build_expected_minute_grid(self.sessions, self.extra_times)
        if len(grid) != self.expected_minutes:
            raise ValueError(
                f"Configured sessions produce {len(grid)} minutes, "
                f"but expected_minutes={self.expected_minutes}"
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        output_dir: str | Path | None = None,
        scope: str | None = None,
    ) -> "MinuteQualityAuditConfig":
        raw_sessions = values.get("minute_sessions", DEFAULT_MINUTE_SESSIONS)
        sessions = tuple((str(item[0]), str(item[1])) for item in raw_sessions)
        extra_times = tuple(
            str(value) for value in values.get(
                "minute_extra_times", DEFAULT_MINUTE_EXTRA_TIMES
            )
        )
        return cls(
            output_dir=Path(output_dir or values.get(
                "quality_audit_dir", "results/minute_cpu_ddb/data_quality"
            )),
            expected_minutes=int(values.get("expected_minutes", 241)),
            sessions=sessions,
            extra_times=extra_times,
            chunk_days=int(values.get("quality_audit_chunk_days", 20)),
            scope=str(scope or values.get("quality_audit_scope", "grid")).lower(),
        )


class DolphinDBMinuteQualityAuditor:
    """Read-only, partition-bounded quality audit for a minute-bar table."""

    def __init__(
        self,
        loader: DolphinDBMinuteLoader,
        config: MinuteQualityAuditConfig,
    ) -> None:
        self.loader = loader
        self.config = config
        self.expected_grid = build_expected_minute_grid(
            config.sessions, config.extra_times
        )

    def run(self) -> Path:
        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=True)
        field_audit = self.loader.audit()
        self._write_json(output / "field_audit.json", field_audit.to_dict())
        dates = self.loader.load_trade_dates(
            self.loader.config.start_date, self.loader.config.end_date
        )
        time_parts: list[pd.DataFrame] = []
        value_parts: list[pd.DataFrame] = []
        duplicate_parts: list[pd.DataFrame] = []
        symbol_parts: list[pd.DataFrame] = []
        chunks = [
            dates[index:index + self.config.chunk_days]
            for index in range(0, len(dates), self.config.chunk_days)
        ]
        for index, chunk in enumerate(chunks, start=1):
            start, end = pd.Timestamp(chunk[0]), pd.Timestamp(chunk[-1])
            time_parts.append(self._query(self._time_distribution_sql(start, end)))
            value_parts.append(self._query(self._value_issue_sql(start, end)))
            duplicate_parts.append(self._query(self._duplicate_sql(start, end)))
            if self.config.scope == "full":
                symbol_parts.append(self._query(self._symbol_sql(start, end)))
            print(
                f"[DDBQuality] progress={index}/{len(chunks)} "
                f"dates={start.date()}..{end.date()} scope={self.config.scope}",
                flush=True,
            )

        time_distribution = self._concat(time_parts)
        value_issues = self._concat(value_parts)
        duplicates = self._concat(duplicate_parts)
        symbol_counts = self._concat(symbol_parts)
        date_quality, time_presence = self._summarize_dates(
            dates, time_distribution, value_issues, duplicates
        )
        self._write_csv(output / "date_quality.csv", date_quality)
        self._write_csv(output / "time_presence.csv", time_presence)
        self._write_csv(output / "value_issues.csv", value_issues)
        self._write_csv(output / "duplicate_keys.csv", duplicates)
        if self.config.scope == "full":
            problem_symbols = self._summarize_symbols(symbol_counts, duplicates)
            self._write_csv(output / "problem_symbols.csv", problem_symbols)

        problem_dates = date_quality.loc[date_quality["status"] != "ok"]
        summary = {
            "status": "failed" if len(problem_dates) else "passed",
            "scope": self.config.scope,
            "source": {
                "database": self.loader.config.database,
                "table": self.loader.config.table,
                "start_date": str(pd.Timestamp(dates[0]).date()),
                "end_date": str(pd.Timestamp(dates[-1]).date()),
            },
            "expected": {
                "minutes": self.config.expected_minutes,
                "sessions": [list(item) for item in self.config.sessions],
                "extra_times": list(self.config.extra_times),
            },
            "counts": {
                "trade_dates": len(dates),
                "problem_dates": len(problem_dates),
                "duplicate_keys": len(duplicates),
                "observed_distinct_times": int(time_presence["time"].nunique())
                if not time_presence.empty else 0,
            },
            "outputs": {
                "date_quality": "date_quality.csv",
                "time_presence": "time_presence.csv",
                "value_issues": "value_issues.csv",
                "duplicate_keys": "duplicate_keys.csv",
                "problem_symbols": "problem_symbols.csv" if self.config.scope == "full" else None,
            },
        }
        summary_path = output / "summary.json"
        self._write_json(summary_path, summary)
        print(
            f"[DDBQuality] complete status={summary['status']} "
            f"problem_dates={len(problem_dates)}/{len(dates)} output={output}",
            flush=True,
        )
        return summary_path

    def _time_distribution_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        return (
            "select date, time, count(*) as rowCount from "
            f"{self.loader.table_expression} where date >= {start:%Y.%m.%d}, "
            f"date <= {end:%Y.%m.%d} group by date, time order by date, time "
            "// ALPHAMINING_QUALITY_TIME_V1"
        )

    def _value_issue_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        return (
            "select "
            "sum(iif(isNull(sym) or isNull(time), 1, 0)) as invalidKeyRows, "
            "sum(iif(isNull(open) or isNull(high) or isNull(low) or isNull(close) "
            "or isNull(volume) or isNull(amount), 1, 0)) as nullValueRows, "
            "sum(iif(open <= 0 or high <= 0 or low <= 0 or close <= 0, 1, 0)) "
            "as nonPositivePriceRows, "
            "sum(iif(high < low or high < open or high < close or low > open "
            "or low > close, 1, 0)) as invalidOhlcRows, "
            "sum(iif(volume < 0 or amount < 0 or tradeCount < 0, 1, 0)) "
            f"as negativeActivityRows from {self.loader.table_expression} "
            f"where date >= {start:%Y.%m.%d}, date <= {end:%Y.%m.%d} group by date "
            "order by date // ALPHAMINING_QUALITY_VALUES_V1"
        )

    def _duplicate_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        return (
            "select date, sym, time, count(*) as duplicateCount from "
            f"{self.loader.table_expression} where date >= {start:%Y.%m.%d}, "
            f"date <= {end:%Y.%m.%d} group by date, sym, time having count(*) > 1 "
            "order by date, sym, time // ALPHAMINING_QUALITY_DUPLICATES_V1"
        )

    def _symbol_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        return (
            "select count(*) as rowCount, min(time) as firstTime, max(time) as lastTime "
            f"from {self.loader.table_expression} where date >= {start:%Y.%m.%d}, "
            f"date <= {end:%Y.%m.%d} group by date, sym order by date, sym "
            "// ALPHAMINING_QUALITY_SYMBOL_V1"
        )

    def _summarize_dates(
        self,
        dates: Sequence[pd.Timestamp],
        distribution: pd.DataFrame,
        value_issues: pd.DataFrame,
        duplicates: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        expected = set(self.expected_grid)
        if distribution.empty or not {"date", "time"}.issubset(distribution.columns):
            raise ValueError("Time-distribution query returned no date/time columns")
        normalized = distribution.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        normalized = normalized.loc[normalized["time"].notna()].copy()
        normalized["time"] = normalized["time"].map(_time_key)
        presence = normalized.groupby("time", observed=True).agg(
            present_dates=("date", "nunique"),
            total_rows=("rowCount", "sum"),
        ).reset_index().sort_values("time")
        presence["in_expected_session"] = presence["time"].isin(expected)

        values = self._index_by_date(value_issues)
        duplicate_counts = pd.Series(dtype=np.int64)
        if not duplicates.empty and "date" in duplicates:
            duplicate_frame = duplicates.copy()
            duplicate_frame["date"] = pd.to_datetime(duplicate_frame["date"]).dt.normalize()
            duplicate_frame["extraRows"] = (
                pd.to_numeric(duplicate_frame["duplicateCount"], errors="coerce").fillna(0) - 1
            ).clip(lower=0)
            duplicate_counts = duplicate_frame.groupby("date")["extraRows"].sum()

        rows: list[dict[str, Any]] = []
        for date in pd.DatetimeIndex(dates).normalize():
            observed = set(normalized.loc[normalized["date"] == date, "time"])
            extra = sorted(observed - expected)
            missing = sorted(expected - observed)
            issue_counts = {
                name: int(values.at[date, name])
                if date in values.index and name in values.columns and pd.notna(values.at[date, name])
                else 0
                for name in (
                    "invalidKeyRows", "nullValueRows", "nonPositivePriceRows",
                    "invalidOhlcRows", "negativeActivityRows",
                )
            }
            duplicate_rows = int(duplicate_counts.get(date, 0))
            status = "ok"
            if missing or extra or duplicate_rows or any(issue_counts.values()):
                status = "problem"
            rows.append({
                "date": str(date.date()),
                "status": status,
                "observed_minutes": len(observed),
                "expected_minutes": self.config.expected_minutes,
                "missing_minutes": len(missing),
                "extra_minutes": len(extra),
                "missing_times": "|".join(missing),
                "extra_times": "|".join(extra),
                "duplicate_extra_rows": duplicate_rows,
                **issue_counts,
            })
        return pd.DataFrame(rows), presence

    def _summarize_symbols(
        self, symbol_counts: pd.DataFrame, duplicates: pd.DataFrame
    ) -> pd.DataFrame:
        if symbol_counts.empty:
            return pd.DataFrame(columns=["date", "sym", "rowCount", "issue"])
        result = symbol_counts.copy()
        result["rowCount"] = pd.to_numeric(result["rowCount"], errors="coerce")
        result["issue"] = np.where(
            result["rowCount"] < self.config.expected_minutes,
            "missing_rows",
            np.where(result["rowCount"] > self.config.expected_minutes, "extra_or_duplicate_rows", "ok"),
        )
        result = result.loc[result["issue"] != "ok"].copy()
        if not duplicates.empty:
            duplicate_keys = duplicates[["date", "sym"]].drop_duplicates().assign(hasDuplicate=True)
            result = result.merge(duplicate_keys, on=["date", "sym"], how="left")
            result["hasDuplicate"] = result["hasDuplicate"].fillna(False)
        return result

    @staticmethod
    def _index_by_date(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "date" not in frame:
            return pd.DataFrame()
        result = frame.copy()
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        return result.drop_duplicates("date", keep="last").set_index("date")

    def _query(self, script: str) -> pd.DataFrame:
        return pd.DataFrame(self.loader.session.run(script))

    @staticmethod
    def _concat(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [part for part in parts if not part.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

    @staticmethod
    def _write_csv(path: Path, frame: pd.DataFrame) -> None:
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
