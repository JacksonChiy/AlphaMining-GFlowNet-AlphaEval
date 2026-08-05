from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


SOURCE_COLUMNS = (
    "sym", "date", "time", "open", "high", "low", "close",
    "volume", "amount", "tradeCount",
)
CANONICAL_COLUMNS = (
    "date", "datetime", "code", "open", "high", "low", "close",
    "vol", "amount", "trade_count",
)
NUMERIC_SOURCE_COLUMNS = ("open", "high", "low", "close", "volume", "amount", "tradeCount")


class SessionLike(Protocol):
    def run(self, script: str) -> Any: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DolphinDBConnectionConfig:
    host: str
    port: int
    user: str
    password: str
    keep_alive_seconds: int = 60
    reconnect_attempts: int = 3

    @classmethod
    def from_environment(cls, values: Mapping[str, Any]) -> "DolphinDBConnectionConfig":
        host_env = str(values.get("host_env", "DDB_HOST"))
        port_env = str(values.get("port_env", "DDB_PORT"))
        user_env = str(values.get("user_env", "DDB_USER"))
        password_env = str(values.get("password_env", "DDB_PASSWORD"))
        required = {name: os.environ.get(name) for name in (host_env, port_env, user_env, password_env)}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Missing DolphinDB connection environment variables: " + ", ".join(missing)
            )
        return cls(
            host=str(required[host_env]),
            port=int(str(required[port_env])),
            user=str(required[user_env]),
            password=str(required[password_env]),
            keep_alive_seconds=int(values.get("keep_alive_seconds", 60)),
            reconnect_attempts=int(values.get("reconnect_attempts", 3)),
        )


@dataclass(frozen=True)
class MinuteDolphinDBConfig:
    database: str
    table: str
    start_date: str
    end_date: str
    cache_dir: Path
    daily_file: Path
    chunk_days: int = 20
    prices_are_adjusted: bool = False
    force_refresh: bool = False
    load_mode: str = "cache"
    audit_chunk_days: int = 120
    daily_aggregate_chunk_days: int = 120
    trade_days_table: str = "TradeDays"
    trade_days_date_column: str | None = None
    pushdown_enabled: bool = True
    pushdown_fallback: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"dfs://[A-Za-z0-9_./-]+", self.database):
            raise ValueError("DolphinDB database must be an explicit dfs:// path")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.table):
            raise ValueError("DolphinDB table contains unsupported characters")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.trade_days_table):
            raise ValueError("DolphinDB TradeDays table contains unsupported characters")
        if self.trade_days_date_column is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.trade_days_date_column
        ):
            raise ValueError("DolphinDB TradeDays date column contains unsupported characters")
        if min(self.chunk_days, self.audit_chunk_days, self.daily_aggregate_chunk_days) < 1:
            raise ValueError("DolphinDB chunk day settings must be positive")
        if self.load_mode not in {"cache", "stream"}:
            raise ValueError("DolphinDB load_mode must be 'cache' or 'stream'")
        start, end = pd.Timestamp(self.start_date), pd.Timestamp(self.end_date)
        if start > end:
            raise ValueError("DolphinDB start_date must not be later than end_date")

    @classmethod
    def from_mapping(
        cls,
        dataset: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> "MinuteDolphinDBConfig":
        database = values.get("database") or os.environ.get(
            str(values.get("database_env", "DDB_DATABASE")), ""
        )
        table = values.get("table") or os.environ.get(
            str(values.get("table_env", "DDB_TABLE")), ""
        )
        return cls(
            database=str(database),
            table=str(table),
            start_date=str(values.get("start_date", dataset.get("mining_start_date", ""))),
            end_date=str(values.get("end_date", dataset.get("out_of_sample_end_date", ""))),
            cache_dir=Path(str(values.get("cache_dir", "data/minute_ddb_cache"))),
            daily_file=Path(str(dataset.get("daily_file", "data/daily_price_ddb.pkl"))),
            chunk_days=int(values.get("chunk_days", 20)),
            prices_are_adjusted=bool(values.get("prices_are_adjusted", False)),
            force_refresh=bool(values.get("force_refresh", False)),
            load_mode=str(values.get("load_mode", "cache")).lower(),
            audit_chunk_days=int(values.get("audit_chunk_days", 120)),
            daily_aggregate_chunk_days=int(values.get("daily_aggregate_chunk_days", 120)),
            trade_days_table=str(values.get("trade_days_table", "TradeDays")),
            trade_days_date_column=(
                str(values["trade_days_date_column"])
                if values.get("trade_days_date_column")
                else None
            ),
            pushdown_enabled=bool(values.get("pushdown_enabled", True)),
            pushdown_fallback=bool(values.get("pushdown_fallback", True)),
        )


@dataclass
class DolphinDBFieldAudit:
    database: str
    table: str
    required_fields: list[dict[str, Any]]
    source_schema: list[dict[str, str]]
    source_min_date: str | None
    source_max_date: str | None
    source_rows: int
    requested_start_date: str
    requested_end_date: str
    prices_are_adjusted: bool
    data_sql: str
    trade_days_table: str
    trade_days_date_column: str
    requested_trade_days: int

    @property
    def passed(self) -> bool:
        return self.prices_are_adjusted and all(
            field["status"] in {"direct", "derived"} for field in self.required_fields
        )

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return {
            "passed": self.passed,
            "data": [{
                "dbName": self.database,
                "tbName": self.table,
                "baseTb": True,
                "requiredFields": self.required_fields,
                "joinFields": [],
            }],
            "timeRange": {
                "start": self.requested_start_date,
                "end": self.requested_end_date,
                "sourceStart": self.source_min_date,
                "sourceEnd": self.source_max_date,
            },
            "joinPlan": {"steps": []},
            "tradeCalendar": {
                "table": self.trade_days_table,
                "dateColumn": self.trade_days_date_column,
                "requestedTradeDays": self.requested_trade_days,
            },
            "output": {
                "frequency": "1min",
                "timeColumn": "datetime(date + time)",
                "groupFields": ["sym"],
            },
            "pricesAreAdjusted": self.prices_are_adjusted,
            "dataSql": self.data_sql,
            "sourceAudit": raw,
        }


class DolphinDBMinuteLoader:
    def __init__(self, config: MinuteDolphinDBConfig, session: SessionLike) -> None:
        self.config = config
        self.session = session
        self._trade_date_column: str | None = None
        self._trade_dates_cache: dict[tuple[str, str], tuple[pd.Timestamp, ...]] = {}

    @property
    def table_expression(self) -> str:
        return f'loadTable("{self.config.database}", "{self.config.table}")'

    @property
    def trade_days_expression(self) -> str:
        return f'loadTable("{self.config.database}", "{self.config.trade_days_table}")'

    def resolve_trade_date_column(self) -> str:
        """Resolve the calendar date column from schema, never by an unchecked guess."""
        if self._trade_date_column is not None:
            return self._trade_date_column
        schema = self._schema_frame(
            self.session.run(f"schema({self.trade_days_expression}).colDefs")
        )
        configured = self.config.trade_days_date_column
        if configured is not None:
            matches = schema.loc[schema["name"].astype(str) == configured]
            if matches.empty:
                raise ValueError(
                    f"TradeDays date column '{configured}' does not exist; "
                    f"available columns={schema['name'].astype(str).tolist()}"
                )
            type_name = str(matches.iloc[0]["typeString"]).upper().removeprefix("DT_")
            if type_name != "DATE":
                raise ValueError(
                    f"TradeDays column '{configured}' must be DATE, actual={type_name}"
                )
            self._trade_date_column = configured
            return configured
        date_columns = schema.loc[
            schema["typeString"].astype(str).str.upper().str.removeprefix("DT_") == "DATE",
            "name",
        ].astype(str).tolist()
        if len(date_columns) != 1:
            raise ValueError(
                "TradeDays must have exactly one DATE column when "
                "trade_days_date_column is omitted; "
                f"DATE columns={date_columns}, all columns={schema['name'].astype(str).tolist()}"
            )
        self._trade_date_column = date_columns[0]
        print(
            f"[DDB] TradeDays date column resolved from schema: {self._trade_date_column}",
            flush=True,
        )
        return self._trade_date_column

    def load_trade_dates(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> tuple[pd.Timestamp, ...]:
        start = pd.Timestamp(start_date or self.config.start_date).normalize()
        end = pd.Timestamp(end_date or self.config.end_date).normalize()
        key = (str(start.date()), str(end.date()))
        cached = self._trade_dates_cache.get(key)
        if cached is not None:
            return cached
        column = self.resolve_trade_date_column()
        script = (
            f"select distinct {column} as tradeDate from {self.trade_days_expression} "
            f"where {column} >= {start:%Y.%m.%d}, {column} <= {end:%Y.%m.%d} "
            "order by tradeDate"
        )
        result = pd.DataFrame(self.session.run(script))
        if "tradeDate" not in result:
            raise ValueError("TradeDays query did not return the required tradeDate column")
        dates = tuple(
            pd.DatetimeIndex(pd.to_datetime(result["tradeDate"], errors="coerce"))
            .dropna().normalize().unique().sort_values()
        )
        if not dates:
            raise ValueError(f"TradeDays contains no dates in {start.date()}..{end.date()}")
        self._trade_dates_cache[key] = dates
        print(
            f"[DDB] trade_days_loaded count={len(dates):,} "
            f"start={dates[0].date()} end={dates[-1].date()}",
            flush=True,
        )
        return dates

    def audit(self) -> DolphinDBFieldAudit:
        schema = self.session.run(f"schema({self.table_expression}).colDefs")
        schema_frame = self._schema_frame(schema)
        available = set(schema_frame["name"].astype(str))
        source_types = {
            str(row["name"]): str(row["typeString"]).upper().removeprefix("DT_")
            for _, row in schema_frame.iterrows()
        }
        expected_types = {
            "sym": {"SYMBOL", "STRING"},
            "date": {"DATE"},
            "time": {"MINUTE", "SECOND", "TIME", "NANOTIME", "DATETIME", "TIMESTAMP"},
            "open": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
            "high": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
            "low": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
            "close": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
            "volume": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
            "amount": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
            "tradeCount": {"CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"},
        }
        numeric_columns = set(NUMERIC_SOURCE_COLUMNS)
        mappings = {
            "securityId": (["sym"], "direct", "sym maps to canonical code"),
            "tradeDate": (["date"], "direct", "date is the trading date"),
            "tradeTime": (["date", "time"], "derived", "datetime is date plus time"),
            "open": (["open"], "direct", "minute open"),
            "high": (["high"], "direct", "minute high"),
            "low": (["low"], "direct", "minute low"),
            "close": (["close"], "direct", "minute close"),
            "volume": (["volume"], "direct", "volume maps to canonical vol"),
            "amount": (["amount"], "direct", "minute turnover amount"),
            "tradeCount": (["tradeCount"], "direct", "retained as optional audit field"),
        }
        required_fields: list[dict[str, Any]] = []
        for name, (columns, status, reason) in mappings.items():
            missing = [column for column in columns if column not in available]
            incompatible = [
                f"{column}:{source_types.get(column, 'UNKNOWN')}"
                for column in columns
                if column in available
                and not (
                    (
                        column in numeric_columns
                        and source_types.get(column, "").startswith("DECIMAL")
                    )
                    or source_types.get(column) in expected_types[column]
                )
            ]
            required_fields.append({
                "name": name,
                "sourceColumns": columns,
                "status": "missing" if missing or incompatible else status,
                "reason": (
                    f"missing source columns: {missing}"
                    if missing
                    else f"incompatible source types: {incompatible}"
                    if incompatible
                    else reason
                ),
            })
        first = self._collect_requested_range_stats()
        trade_dates = self.load_trade_dates(self.config.start_date, self.config.end_date)
        audit = DolphinDBFieldAudit(
            database=self.config.database,
            table=self.config.table,
            required_fields=required_fields,
            source_schema=schema_frame[["name", "typeString"]].astype(str).to_dict("records"),
            source_min_date=self._date_string(first.get("minDate")),
            source_max_date=self._date_string(first.get("maxDate")),
            source_rows=int(first.get("rows", 0)),
            requested_start_date=str(pd.Timestamp(self.config.start_date).date()),
            requested_end_date=str(pd.Timestamp(self.config.end_date).date()),
            prices_are_adjusted=self.config.prices_are_adjusted,
            data_sql=self.build_data_sql(
                pd.Timestamp(self.config.start_date), pd.Timestamp(self.config.end_date)
            ),
            trade_days_table=self.config.trade_days_table,
            trade_days_date_column=self.resolve_trade_date_column(),
            requested_trade_days=len(trade_dates),
        )
        fields_passed = all(
            item["status"] in {"direct", "derived"} for item in required_fields
        )
        if not fields_passed:
            missing = [item["name"] for item in required_fields if item["status"] == "missing"]
            raise ValueError(f"DolphinDB field audit failed; missing fields: {missing}")
        self._validate_date_coverage(audit)
        return audit

    def extract(self) -> tuple[Path, Path]:
        cache_dir = self.config.cache_dir
        manifest_path = cache_dir / "manifest.json"
        if manifest_path.exists() and not self.config.force_refresh:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == self._fingerprint() and manifest.get("complete"):
                print(
                    f"[DDB] cache_reused rows={manifest.get('rows', 0):,} "
                    f"partitions={len(manifest.get('files', []))} path={cache_dir}",
                    flush=True,
                )
                if not self.config.daily_file.exists():
                    build_daily_from_minute_cache(cache_dir, self.config.daily_file)
                return cache_dir, self.config.daily_file

        audit = self.audit()
        cache_dir.mkdir(parents=True, exist_ok=True)
        audit_path = cache_dir / "field_audit.json"
        audit_path.write_text(
            json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not self.config.prices_are_adjusted:
            raise ValueError(
                "Field audit completed, but the source has no adjustment factor. Confirm that OHLC "
                "prices are already adjusted, then set "
                "dataset.dolphindb.prices_are_adjusted=true."
            )
        files: list[str] = []
        total_rows = 0
        for chunk_index, (start, end) in enumerate(self._date_chunks(), start=1):
            script = self.build_data_sql(start, end)
            raw = self.session.run(script)
            frame = normalize_dolphindb_minutes(pd.DataFrame(raw))
            output = cache_dir / f"minute_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
            temporary = output.with_suffix(".pkl.tmp")
            frame.to_pickle(temporary)
            temporary.replace(output)
            files.append(output.name)
            total_rows += len(frame)
            print(
                f"[DDB] chunk_complete index={chunk_index:03d} "
                f"start={start.date()} end={end.date()} rows={len(frame):,} "
                f"total_rows={total_rows:,} file={output}",
                flush=True,
            )
        if total_rows == 0:
            raise ValueError("DolphinDB extraction returned zero rows")
        manifest = {
            "complete": True,
            "fingerprint": self._fingerprint(),
            "database": self.config.database,
            "table": self.config.table,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "rows": total_rows,
            "files": files,
            "field_audit": audit_path.name,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        build_daily_from_minute_cache(cache_dir, self.config.daily_file)
        return cache_dir, self.config.daily_file

    def iter_frames(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ):
        """Yield normalized DDB chunks directly from memory without local minute files."""
        trade_dates = self.load_trade_dates(start_date, end_date)
        days = self.config.chunk_days
        chunks = [
            (trade_dates[index], trade_dates[min(index + days - 1, len(trade_dates) - 1)])
            for index in range(0, len(trade_dates), days)
        ]
        for chunk_index, (start, end) in enumerate(chunks, start=1):
            raw = self.session.run(self.build_data_sql(start, end))
            frame = normalize_dolphindb_minutes(pd.DataFrame(raw))
            print(
                f"[DDBStream] chunk={chunk_index:03d}/{len(chunks):03d} "
                f"start={start.date()} end={end.date()} minute_rows={len(frame):,}",
                flush=True,
            )
            if not frame.empty:
                yield start, end, frame

    def execute_minute_blocks(
        self,
        nodes,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.Series]:
        """Execute supported minute blocks in DDB and return daily indexed series only."""
        from src.expression.dolphindb_minute import DolphinDBMinuteCompiler

        compiler = DolphinDBMinuteCompiler(self.table_expression)
        compiled = compiler.compile(nodes, start, end)
        frame = pd.DataFrame(self.session.run(compiled.script))
        if frame.empty:
            return {key: pd.Series(dtype=float) for key in compiled.aliases}
        required = {"date", "sym", *compiled.aliases.values()}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"DolphinDB pushdown result is missing columns: {missing}")
        index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(frame["date"], errors="coerce").dt.normalize(),
                frame["sym"].astype(str),
            ],
            names=["date", "code"],
        )
        output: dict[str, pd.Series] = {}
        for rendered, alias in compiled.aliases.items():
            values = pd.to_numeric(frame[alias], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            output[rendered] = pd.Series(values.to_numpy(dtype=float), index=index)
        return output

    def iter_minute_blocks(
        self,
        nodes,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ):
        """Push a block batch to DDB one trade-date chunk at a time."""
        trade_dates = self.load_trade_dates(start_date, end_date)
        days = self.config.chunk_days
        chunks = [
            (trade_dates[index], trade_dates[min(index + days - 1, len(trade_dates) - 1)])
            for index in range(0, len(trade_dates), days)
        ]
        for chunk_index, (start, end) in enumerate(chunks, start=1):
            values = self.execute_minute_blocks(nodes, start, end)
            rows = max((len(value) for value in values.values()), default=0)
            print(
                f"[DDBPushdown] chunk={chunk_index:03d}/{len(chunks):03d} "
                f"start={start.date()} end={end.date()} blocks={len(nodes):03d} "
                f"daily_rows={rows:,}",
                flush=True,
            )
            yield start, end, values

    def build_daily_in_memory(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Let DolphinDB aggregate daily OHLCV so raw minutes never cross the network here."""
        daily_parts: list[pd.DataFrame] = []
        chunks = list(self._date_chunks(
            start_date, end_date, self.config.daily_aggregate_chunk_days
        ))
        for chunk_index, (start, end) in enumerate(chunks, start=1):
            raw = pd.DataFrame(self.session.run(self.build_daily_sql(start, end)))
            if raw.empty:
                continue
            required = {
                "date", "sym", "open", "high", "low", "close",
                "volume", "amount", "trade_count",
            }
            missing = sorted(required.difference(raw.columns))
            if missing:
                raise ValueError(f"DolphinDB daily aggregate is missing columns: {missing}")
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
            raw["code"] = raw["sym"].astype("string").str.strip()
            for column in (
                "open", "high", "low", "close", "volume", "amount", "trade_count"
            ):
                raw[column] = pd.to_numeric(raw[column], errors="coerce")
            daily_parts.append(raw[[
                "date", "code", "open", "high", "low", "close",
                "volume", "amount", "trade_count",
            ]])
            if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % 25 == 0:
                print(
                    f"[DDBStream] daily_aggregate_progress "
                    f"chunks={chunk_index}/{len(chunks)} rows={sum(map(len, daily_parts)):,}",
                    flush=True,
                )
        if not daily_parts:
            raise ValueError("DolphinDB streaming query returned no minute rows")
        daily = pd.concat(daily_parts, ignore_index=True)
        daily = daily.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
        daily["vwap"] = daily["amount"].div(daily["volume"].where(daily["volume"] > 0))
        print(
            f"[DDBStream] daily_ready rows={len(daily):,} "
            f"dates={daily['date'].nunique():,} stocks={daily['code'].nunique():,}",
            flush=True,
        )
        return daily

    def build_daily_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        start_literal, end_literal = start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
        return (
            "dailySource = select date, sym, time, open, high, low, close, volume, "
            f"amount, tradeCount from {self.table_expression} "
            f"where date >= {start_literal}, date <= {end_literal} "
            "order by date, sym, time; "
            "select first(open) as open, max(high) as high, min(low) as low, "
            "last(close) as close, sum(volume) as volume, sum(amount) as amount, "
            "sum(tradeCount) as trade_count "
            "from dailySource "
            "group by date, sym order by date, sym"
        )

    def build_data_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        start_literal, end_literal = start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
        columns = ", ".join(SOURCE_COLUMNS)
        return (
            f"select {columns} from {self.table_expression} "
            f"where date >= {start_literal}, date <= {end_literal} "
            "order by date, sym, time"
        )

    def build_stats_sql(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        """Build a partition-pruned audit query for one bounded date chunk."""
        start_literal, end_literal = start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
        return (
            "select min(date) as minDate, max(date) as maxDate, count(*) as rows "
            f"from {self.table_expression} "
            f"where date >= {start_literal}, date <= {end_literal}"
        )

    def _collect_requested_range_stats(self) -> dict[str, Any]:
        """Aggregate audit statistics without ever scanning all table partitions."""
        min_dates: list[pd.Timestamp] = []
        max_dates: list[pd.Timestamp] = []
        total_rows = 0
        chunks = list(self._date_chunks(
            chunk_days=self.config.audit_chunk_days
        ))
        for chunk_index, (start, end) in enumerate(chunks, start=1):
            result = pd.DataFrame(self.session.run(self.build_stats_sql(start, end)))
            if not result.empty:
                row = result.iloc[0]
                rows = int(row.get("rows", 0) or 0)
                total_rows += rows
                min_date = row.get("minDate")
                max_date = row.get("maxDate")
                if rows > 0 and min_date is not None and not pd.isna(min_date):
                    min_dates.append(pd.Timestamp(min_date))
                if rows > 0 and max_date is not None and not pd.isna(max_date):
                    max_dates.append(pd.Timestamp(max_date))
            if chunk_index == 1 or chunk_index == len(chunks) or chunk_index % 25 == 0:
                print(
                    f"[DDB] audit_progress chunks={chunk_index}/{len(chunks)} "
                    f"rows={total_rows:,}",
                    flush=True,
                )
        return {
            "minDate": min(min_dates) if min_dates else None,
            "maxDate": max(max_dates) if max_dates else None,
            "rows": total_rows,
        }

    def _date_chunks(
        self,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        chunk_days: int | None = None,
    ):
        current = pd.Timestamp(start_date or self.config.start_date).normalize()
        final = pd.Timestamp(end_date or self.config.end_date).normalize()
        if current > final:
            raise ValueError("DolphinDB stream start_date must not be later than end_date")
        while current <= final:
            days = int(chunk_days or self.config.chunk_days)
            end = min(current + pd.Timedelta(days=days - 1), final)
            yield current, end
            current = end + pd.Timedelta(days=1)

    def _fingerprint(self) -> str:
        payload = {
            "database": self.config.database,
            "table": self.config.table,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "columns": SOURCE_COLUMNS,
            "prices_are_adjusted": self.config.prices_are_adjusted,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _schema_frame(schema: Any) -> pd.DataFrame:
        frame = pd.DataFrame(schema).copy()
        if "name" not in frame:
            raise ValueError(f"DolphinDB schema result has no name column: {list(frame.columns)}")
        if "typeString" not in frame:
            if "typeInt" in frame:
                frame["typeString"] = frame["typeInt"].astype(str)
            else:
                frame["typeString"] = "UNKNOWN"
        return frame

    @staticmethod
    def _date_string(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(pd.Timestamp(value).date())

    def _validate_date_coverage(self, audit: DolphinDBFieldAudit) -> None:
        if audit.source_min_date is None or audit.source_max_date is None:
            raise ValueError("DolphinDB source table contains no valid date range")
        # Calendar boundaries commonly fall on weekends or exchange holidays.
        boundary_tolerance = pd.Timedelta(days=7)
        if pd.Timestamp(audit.source_min_date) > (
            pd.Timestamp(self.config.start_date) + boundary_tolerance
        ):
            raise ValueError(
                "DolphinDB source starts later than requested extraction start: "
                f"requested_start={self.config.start_date}, "
                f"actual_first_trade_date={audit.source_min_date}. "
                "Update dataset.dolphindb.start_date and dataset.mining_start_date "
                "to a covered date."
            )
        if pd.Timestamp(audit.source_max_date) < (
            pd.Timestamp(self.config.end_date) - boundary_tolerance
        ):
            raise ValueError(
                "DolphinDB source ends earlier than requested extraction end: "
                f"requested_end={self.config.end_date}, "
                f"actual_last_trade_date={audit.source_max_date}. "
                "Set dataset.dolphindb.end_date and "
                "dataset.out_of_sample_end_date to actual_last_trade_date (or earlier)."
            )


def create_dolphindb_session(values: Mapping[str, Any]) -> SessionLike:
    try:
        import dolphindb as ddb
    except ImportError as exc:
        raise ImportError(
            "DolphinDB Python API is not installed. Run: pip install -r requirements-ddb.txt"
        ) from exc
    connection = DolphinDBConnectionConfig.from_environment(values)
    session = ddb.Session()
    connected = session.connect(
        connection.host,
        connection.port,
        connection.user,
        connection.password,
        keepAliveTime=connection.keep_alive_seconds,
        reconnect=True,
        tryReconnectNums=connection.reconnect_attempts,
    )
    if connected is False:
        session.close()
        raise ConnectionError("DolphinDB Session.connect returned False")
    return session


def normalize_dolphindb_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"DolphinDB minute chunk is missing columns: {missing}")
    work = frame.loc[:, SOURCE_COLUMNS].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["datetime"] = _combine_date_time(work["date"], work["time"])
    work["code"] = work["sym"].astype("string").str.strip()
    for column in NUMERIC_SOURCE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.rename(columns={"volume": "vol", "tradeCount": "trade_count"})
    invalid_key = work["date"].isna() | work["datetime"].isna() | work["code"].isna() | work["code"].eq("")
    invalid_price = (
        (work["open"] <= 0) | (work["high"] <= 0) | (work["low"] <= 0)
        | (work["close"] <= 0) | (work["low"] > work["high"])
    )
    work = work.loc[~(invalid_key | invalid_price)].copy()
    work["vol"] = work["vol"].clip(lower=0)
    work["amount"] = work["amount"].clip(lower=0)
    work["trade_count"] = work["trade_count"].clip(lower=0)
    work = work.loc[:, CANONICAL_COLUMNS]
    work = work.sort_values(["date", "code", "datetime"], kind="stable")
    work = work.drop_duplicates(["date", "code", "datetime"], keep="last")
    return work.reset_index(drop=True)


def _combine_date_time(dates: pd.Series, times: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(times.dtype):
        offsets = pd.to_timedelta(times, errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(times.dtype):
        parsed = pd.to_datetime(times, errors="coerce")
        offsets = parsed - parsed.dt.normalize()
    else:
        text = times.map(lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value))
        offsets = pd.to_timedelta(text, errors="coerce")
    return dates + offsets


def load_minute_cache(
    cache_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"DolphinDB minute cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("DolphinDB minute cache is incomplete")
    frames = []
    for filename in manifest.get("files", []):
        frame = pd.read_pickle(cache_dir / filename)
        dates = pd.to_datetime(frame["date"])
        mask = dates.notna()
        if start_date is not None:
            mask &= dates >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= dates <= pd.Timestamp(end_date)
        selected = frame.loc[mask]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise ValueError("No cached minute rows match the requested date range")
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["date", "code", "datetime"], kind="stable").reset_index(drop=True)


def build_daily_from_minute_cache(cache_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    daily_parts = []
    for filename in manifest.get("files", []):
        minute = pd.read_pickle(cache_dir / filename)
        grouped = minute.groupby(["date", "code"], observed=True, sort=True)
        daily_parts.append(grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("vol", "sum"),
            amount=("amount", "sum"),
            trade_count=("trade_count", "sum"),
        ).reset_index())
    if not daily_parts:
        raise ValueError("DolphinDB minute cache contains no partition files")
    daily = pd.concat(daily_parts, ignore_index=True)
    daily = daily.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    daily["vwap"] = daily["amount"].div(daily["volume"].where(daily["volume"] > 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_pickle(output_path)
    print(
        f"[DDB] daily_cache_complete rows={len(daily):,} dates={daily['date'].nunique():,} "
        f"stocks={daily['code'].nunique():,} path={output_path}",
        flush=True,
    )
    return daily


def prepare_dolphindb_minute_data(dataset: Mapping[str, Any]) -> tuple[Path, Path]:
    values = dataset.get("dolphindb")
    if not isinstance(values, Mapping):
        raise ValueError("dataset.dolphindb mapping is required for DolphinDB source")
    config = MinuteDolphinDBConfig.from_mapping(dataset, values)
    session = create_dolphindb_session(values)
    try:
        return DolphinDBMinuteLoader(config, session).extract()
    finally:
        session.close()
