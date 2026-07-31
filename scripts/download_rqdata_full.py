"""一次性下载三指数研究所需的 RQData 基础数据与风险模型数据。

认证只从 RQDATAC2_CONF/RQDATAC_CONF 环境变量读取。程序按年份和股票
批次保存 Parquet 分片；已经完成的分片默认跳过，因而支持安全断点续传。
RQData 和 RQAlphaPlus 均不使用代理。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import rqdatac


STYLE_FACTORS = [
    "size",
    "beta",
    "momentum",
    "liquidity",
    "residual_volatility",
    "non_linear_size",
    "book_to_price",
    "earnings_yield",
    "growth",
    "leverage",
]
MARKET_CAP_FIELDS = ["market_cap_3", "a_share_market_val_in_circulation"]
SHARE_FIELDS = ["total", "circulation_a", "total_a", "free_circulation"]
TURNOVER_FIELDS = ["today", "week", "month", "year", "current_year"]


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def retry(call: Callable[[], object], label: str, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(30, 2**attempt)
            log(f"重试 {label}：第 {attempt}/{attempts} 次失败，{type(exc).__name__}: {exc}")
            time.sleep(delay)


def normalize_frame(value: object) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        value = value.rename(value.name or "value").to_frame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"预期 DataFrame/Series，实际为 {type(value).__name__}")
    return value.reset_index()


def normalize_wide(value: object, value_name: str) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.rename(value_name).reset_index()
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"预期 DataFrame/Series，实际为 {type(value).__name__}")
    index_names = [name or "date" for name in value.index.names]
    if len(index_names) != 1:
        return value.reset_index()
    result = value.rename_axis(index=index_names[0], columns="order_book_id").stack(dropna=False)
    return result.rename(value_name).reset_index()


def load_universe(path: Path, start: str, end: str):
    frame = pd.read_csv(path, usecols=["date", "code"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if frame.empty:
        raise ValueError("指定区间内没有指数成分股数据")
    dates = sorted(frame["date"].drop_duplicates())
    by_year = {
        int(year): sorted(group["code"].dropna().astype(str).unique())
        for year, group in frame.groupby(frame["date"].dt.year)
    }
    return frame, dates, by_year


def download_batched_dataset(
    name: str,
    output: Path,
    dates: list[pd.Timestamp],
    stocks_by_year: dict[int, list[str]],
    batch_size: int,
    query: Callable[[list[str], str, str], object],
    normalize: Callable[[object], pd.DataFrame] = normalize_frame,
) -> dict:
    total = done = skipped = rows = 0
    for year, stocks in stocks_by_year.items():
        year_dates = [date for date in dates if date.year == year]
        if not year_dates:
            continue
        start, end = str(year_dates[0].date()), str(year_dates[-1].date())
        batches = list(chunks(stocks, batch_size))
        total += len(batches)
        for number, stock_batch in enumerate(batches):
            path = output / name / f"year={year}" / f"batch-{number:04d}.parquet"
            if path.exists() and path.stat().st_size > 0:
                skipped += 1
                continue
            label = f"{name}/{year}/{number + 1}/{len(batches)}"
            log(f"下载 {label}，股票数={len(stock_batch)}，区间={start}~{end}")
            value = retry(lambda: query(stock_batch, start, end), label)
            frame = normalize(value)
            atomic_parquet(frame, path)
            done += 1
            rows += len(frame)
            log(f"完成 {label}，行数={len(frame):,}，文件={path}")
    return {"total_chunks": total, "downloaded_chunks": done, "skipped_chunks": skipped, "new_rows": rows}


def download_covariance(
    output: Path, dates: list[pd.Timestamp], workers: int
) -> dict:
    root = output / "factor_covariance"
    pending = []
    skipped = 0
    for date in dates:
        path = root / f"year={date.year}" / f"{date.date()}.parquet"
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
        else:
            pending.append((date, path))

    def fetch(item):
        date, path = item
        value = retry(
            lambda: rqdatac.get_factor_covariance(
                date=str(date.date()),
                horizon="daily",
                model="v2trd",
                industry_mapping="citics_2019",
            ),
            f"factor_covariance/{date.date()}",
        )
        frame = value.rename_axis(index="factor_1", columns="factor_2").stack(dropna=False)
        frame = frame.rename("covariance").reset_index()
        frame.insert(0, "date", date)
        atomic_parquet(frame, path)
        return date, len(frame)

    done = rows = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in pending}
        for future in as_completed(futures):
            date, count = future.result()
            done += 1
            rows += count
            log(f"协方差进度 {done}/{len(pending)}：{date.date()}，行数={count:,}")
    return {"total_chunks": len(dates), "downloaded_chunks": done, "skipped_chunks": skipped, "new_rows": rows}


def download_industry_snapshots(
    output: Path, universe: pd.DataFrame, dates: list[pd.Timestamp], workers: int
) -> dict:
    # 月末快照既能追踪行业变更，又避免按日重复下载相同分类。
    monthly_dates = (
        pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).max().tolist()
    )
    pending = []
    skipped = 0
    for date in monthly_dates:
        path = output / "industry" / f"year={date.year}" / f"{date.date()}.parquet"
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        active = sorted(universe.loc[universe["date"] == date, "code"].unique())
        pending.append((date, active, path))

    def fetch(item):
        date, stocks, path = item
        value = retry(
            lambda: rqdatac.get_instrument_industry(
                stocks, source="citics_2019", level=0, date=str(date.date()), market="cn"
            ),
            f"industry/{date.date()}",
        )
        frame = normalize_frame(value)
        frame.insert(0, "date", date)
        atomic_parquet(frame, path)
        return date, len(frame)

    done = rows = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in pending}
        for future in as_completed(futures):
            date, count = future.result()
            done += 1
            rows += count
            log(f"行业快照进度 {done}/{len(pending)}：{date.date()}，行数={count:,}")
    return {"total_chunks": len(monthly_dates), "downloaded_chunks": done, "skipped_chunks": skipped, "new_rows": rows}


def validate_download(output: Path) -> dict:
    result = {}
    for dataset in sorted(path for path in output.iterdir() if path.is_dir() and path.name != "_logs"):
        files = sorted(dataset.rglob("*.parquet"))
        row_count = 0
        missing = 0
        for path in files:
            frame = pd.read_parquet(path)
            row_count += len(frame)
            missing += int(frame.isna().sum().sum())
        result[dataset.name] = {
            "files": len(files),
            "rows": row_count,
            "missing_values": missing,
            "size_mb": round(sum(path.stat().st_size for path in files) / 1024**2, 2),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", default="data/index_components.csv.gz")
    parser.add_argument("--output", default="data/rqdata")
    parser.add_argument("--start-date", default="2020-01-02")
    parser.add_argument("--end-date", default="2026-07-27")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "factor_exposure",
            "specific_risk",
            "factor_covariance",
            "specific_return",
            "market_cap",
            "shares",
            "turnover_rate",
            "descriptor_exposure",
            "industry",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "RQDATAC_PROXY"]:
        os.environ.pop(key, None)
    if not (os.environ.get("RQDATAC2_CONF") or os.environ.get("RQDATAC_CONF")):
        raise RuntimeError("未找到 RQDATAC2_CONF 或 RQDATAC_CONF 环境变量")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    universe, dates, stocks_by_year = load_universe(
        Path(args.components), args.start_date, args.end_date
    )
    log(
        f"股票并集={universe['code'].nunique():,}，交易日={len(dates):,}，"
        f"区间={dates[0].date()}~{dates[-1].date()}"
    )
    rqdatac.init()
    log(f"RQData连接成功，版本={rqdatac.__version__}，代理=禁用")

    queries = {
        "factor_exposure": (
            lambda stocks, start, end: rqdatac.get_factor_exposure(
                stocks, start_date=start, end_date=end, factors=None,
                industry_mapping="citics_2019", model="v2trd", market="cn"
            ), normalize_frame
        ),
        "specific_risk": (
            lambda stocks, start, end: rqdatac.get_specific_risk(
                stocks, start, end, horizon="daily", model="v2trd",
                industry_mapping="citics_2019"
            ), lambda value: normalize_wide(value, "specific_risk")
        ),
        "specific_return": (
            lambda stocks, start, end: rqdatac.get_specific_return(
                stocks, start, end, model="v2trd", industry_mapping="citics_2019"
            ), lambda value: normalize_wide(value, "specific_return")
        ),
        "market_cap": (
            lambda stocks, start, end: rqdatac.get_factor(
                stocks, MARKET_CAP_FIELDS, start_date=start, end_date=end,
                expect_df=True, market="cn"
            ), normalize_frame
        ),
        "shares": (
            lambda stocks, start, end: rqdatac.get_shares(
                stocks, start_date=start, end_date=end, fields=SHARE_FIELDS,
                expect_df=True, market="cn"
            ), normalize_frame
        ),
        "turnover_rate": (
            lambda stocks, start, end: rqdatac.get_turnover_rate(
                stocks, start_date=start, end_date=end, fields=TURNOVER_FIELDS,
                expect_df=True, market="cn"
            ), normalize_frame
        ),
        "descriptor_exposure": (
            lambda stocks, start, end: rqdatac.get_descriptor_exposure(
                stocks, start, end, descriptors=None, model="v2trd",
                industry_mapping="citics_2019", market="cn"
            ), normalize_frame
        ),
    }

    manifest = {
        "created_at": datetime.now().isoformat(),
        "rqdatac_version": rqdatac.__version__,
        "model": "v2trd",
        "industry_mapping": "citics_2019",
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
        "trading_dates": len(dates),
        "unique_stocks": int(universe["code"].nunique()),
        "proxy": "disabled",
        "datasets": {},
    }
    try:
        for name in args.datasets:
            if name in queries:
                query, normalizer = queries[name]
                manifest["datasets"][name] = download_batched_dataset(
                    name, output, dates, stocks_by_year, args.batch_size, query, normalizer
                )
            elif name == "factor_covariance":
                manifest["datasets"][name] = download_covariance(output, dates, args.workers)
            elif name == "industry":
                manifest["datasets"][name] = download_industry_snapshots(
                    output, universe, dates, args.workers
                )
            else:
                raise ValueError(f"未知数据集：{name}")
            (output / "download_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except Exception as exc:
        manifest["status"] = "incomplete"
        manifest["failed_at"] = datetime.now().isoformat()
        manifest["failure"] = {
            "dataset": name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["validation"] = validate_download(output)
        (output / "download_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise

    manifest["validation"] = validate_download(output)
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().isoformat()
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"全部完成，清单={output / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
