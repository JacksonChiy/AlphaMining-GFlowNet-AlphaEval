from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dolphindb_minute import (
    DolphinDBMinuteLoader,
    MinuteDolphinDBConfig,
    create_dolphindb_session,
)
from src.data_loader.minute_memmap import (
    MinuteMemMapConfig,
    build_ddb_minute_time_filter,
)
from src.expression.minute import minute_expression_from_tokens
from src.gflownet.minute_factor_pool import (
    save_minute_alpha_pool_from_dolphindb_stream,
)


KEYS = ["date", "code"]


def _read_factor_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str)
    if frame["date"].isna().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"Invalid date/code keys in factor matrix: {path}")
    return frame


def _load_saved_pool(metadata_path: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    metadata = pd.read_csv(metadata_path)
    if metadata.empty or not {"factor", "tokens"}.issubset(metadata.columns):
        raise ValueError("Alpha Pool must contain non-empty factor and tokens columns")
    if metadata["factor"].duplicated().any():
        raise ValueError("Alpha Pool contains duplicate factor names")
    pool: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        raw_tokens = row["tokens"]
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Invalid saved tokens for factor {row['factor']}")
        pool.append({
            "expression": minute_expression_from_tokens(tokens),
            "tokens": tokens,
            "coverage": 1.0,
            "valid_date_coverage": 1.0,
        })
    return metadata, pool


def merge_repaired_factor_range(
    existing: pd.DataFrame,
    repaired: pd.DataFrame,
    factor_names: list[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    old_columns = [column for column in existing.columns if column not in KEYS]
    if old_columns != factor_names:
        raise ValueError(
            "Existing factor columns do not match Alpha Pool order: "
            f"matrix={len(old_columns)} pool={len(factor_names)}"
        )
    missing = sorted(set(KEYS + factor_names).difference(repaired.columns))
    if missing:
        raise ValueError(f"Repaired factor range is missing columns: {missing}")
    repair_dates = pd.to_datetime(repaired["date"]).dt.normalize()
    replacement = repaired.loc[repair_dates.between(start, end), KEYS + factor_names].copy()
    if replacement.empty:
        raise ValueError("Repaired factor range contains no requested dates")
    existing_dates = pd.to_datetime(existing["date"]).dt.normalize()
    retained = existing.loc[~existing_dates.between(start, end), KEYS + factor_names]
    combined = pd.concat([retained, replacement], ignore_index=True)
    combined = combined.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if combined.duplicated(KEYS).any():
        raise ValueError("Merged factor matrix contains duplicate date/code rows")
    return combined


def validate_repaired_range(
    frame: pd.DataFrame,
    factor_names: list[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    minimum_coverage: float,
) -> dict[str, float | int | str]:
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    selected = frame.loc[dates.between(start, end), KEYS + factor_names].copy()
    if selected.empty:
        raise ValueError("No rows are available for repaired-range validation")
    numeric = selected[factor_names].apply(pd.to_numeric, errors="coerce")
    finite_ratio = float(np.isfinite(numeric.to_numpy(dtype=float)).mean())
    active = selected.assign(**{
        name: pd.to_numeric(selected[name], errors="coerce") for name in factor_names
    }).groupby("date", observed=True)[factor_names].std(ddof=0).gt(1e-12).sum(axis=1)
    zero_dates = active.index[active.eq(0)]
    if len(zero_dates):
        examples = ",".join(str(pd.Timestamp(value).date()) for value in zero_dates[:10])
        raise ValueError(
            f"Repaired factors still have no cross-sectional variation on "
            f"{len(zero_dates)} dates (first={examples})"
        )
    if finite_ratio < minimum_coverage:
        raise ValueError(
            f"Repaired factor finite ratio {finite_ratio:.2%} is below "
            f"reward.min_coverage={minimum_coverage:.2%}"
        )
    return {
        "start_date": str(selected["date"].min().date()),
        "end_date": str(selected["date"].max().date()),
        "rows": int(len(selected)),
        "dates": int(selected["date"].nunique()),
        "stocks": int(selected["code"].nunique()),
        "factors": int(len(factor_names)),
        "factor_finite_ratio": finite_ratio,
        "active_factors_min": int(active.min()),
        "active_factors_median": float(active.median()),
        "zero_active_factor_dates": int(active.eq(0).sum()),
    }


def _write_matrix(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        frame.to_pickle(temporary)
    elif path.name.endswith(".csv.gz"):
        frame.to_csv(temporary, index=False, compression="gzip")
    elif path.suffix.lower() == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")
    temporary.replace(path)


def _repaired_path(path: Path) -> Path:
    if path.name.endswith(".csv.gz"):
        return path.with_name(path.name[:-7] + ".repaired.csv.gz")
    return path.with_name(path.stem + ".repaired" + path.suffix)


def _replace_outputs(
    pairs: tuple[tuple[Path, Path], ...],
) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for original, repaired_path in pairs:
        if not repaired_path.exists():
            raise FileNotFoundError(repaired_path)
        if original.exists():
            backup = original.with_name(original.name + f".backup_{stamp}")
            shutil.copy2(original, backup)
            print(f"[FactorRepair] backup={backup}", flush=True)
        shutil.copy2(repaired_path, original)
        print(f"[FactorRepair] replaced={original}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute a broken PPU OOS minute-factor range without GFlowNet training"
    )
    parser.add_argument("--config", default="configs/minute/ppu_ddb_ram.yaml")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--warmup-days", type=int, default=60,
        help="Trading-day warmup for daily time-series operators",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Back up and replace configured factor matrices after validation",
    )
    parser.add_argument(
        "--promote-existing", action="store_true",
        help="Validate and publish existing .repaired files without querying DDB",
    )
    args = parser.parse_args()
    if args.replace and args.promote_existing:
        raise ValueError("Use either --replace or --promote-existing, not both")
    if args.warmup_days < 0:
        raise ValueError("--warmup-days must be non-negative")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = config["dataset"]
    outputs = config["outputs"]
    metadata_path = Path(outputs["alpha_pool"])
    pickle_path = Path(outputs["factor_matrix_pickle"])
    csv_path = Path(outputs["factor_matrix"])
    daily_path = Path(outputs["daily_price"])
    end_date = args.end_date or dataset["out_of_sample_end_date"]
    repair_start = pd.Timestamp(args.start_date).normalize()
    repair_end = pd.Timestamp(end_date).normalize()
    if repair_end < repair_start:
        raise ValueError("--end-date must not precede --start-date")

    metadata, pool = _load_saved_pool(metadata_path)
    factor_names = metadata["factor"].astype(str).tolist()
    repaired_pickle = _repaired_path(pickle_path)
    repaired_csv = _repaired_path(csv_path)
    minimum_coverage = float(config["reward"]["min_coverage"])
    if args.promote_existing:
        repaired_existing = _read_factor_matrix(repaired_pickle)
        report = validate_repaired_range(
            repaired_existing,
            factor_names,
            repair_start,
            repair_end,
            minimum_coverage,
        )
        if not repaired_csv.exists():
            raise FileNotFoundError(repaired_csv)
        print(
            f"[FactorRepair] promote_validation="
            f"{json.dumps(report, ensure_ascii=False)}",
            flush=True,
        )
        _replace_outputs(((pickle_path, repaired_pickle), (csv_path, repaired_csv)))
        return
    existing = _read_factor_matrix(pickle_path if pickle_path.exists() else csv_path)
    daily = pd.read_pickle(daily_path)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["code"] = daily["code"].astype(str)

    ddb_values = dataset["dolphindb"]
    source = MinuteDolphinDBConfig.from_mapping(dataset, ddb_values)
    memory = MinuteMemMapConfig.from_mapping(dataset["memory"])
    time_filter_sql = build_ddb_minute_time_filter(
        memory.minute_sessions, memory.minute_extra_times
    )
    session = create_dolphindb_session(ddb_values)
    repair_root = Path(outputs.get("lightgbm_dir", "results/lightgbm")).parent / "factor_repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    try:
        loader = DolphinDBMinuteLoader(source, session)
        trade_dates = pd.DatetimeIndex(loader.load_trade_dates(
            source.start_date, repair_end
        )).normalize()
        repair_position = int(trade_dates.searchsorted(repair_start, side="left"))
        if repair_position >= len(trade_dates):
            raise ValueError("Repair start is later than available TradeDays")
        compute_position = max(0, repair_position - args.warmup_days)
        compute_start = pd.Timestamp(trade_dates[compute_position]).normalize()
        print(
            f"[FactorRepair] plan repair={repair_start.date()}..{repair_end.date()} "
            f"compute={compute_start.date()}..{repair_end.date()} "
            f"warmup_trade_days={repair_position - compute_position} factors={len(pool)}",
            flush=True,
        )
        _, repaired = save_minute_alpha_pool_from_dolphindb_stream(
            pool,
            loader,
            daily,
            start_date=str(compute_start.date()),
            end_date=str(repair_end.date()),
            metadata_path=repair_root / "alpha_pool_recomputed.csv",
            matrix_path=repair_root / "factor_range.pkl",
            min_coverage=0.0,
            time_filter_sql=time_filter_sql,
        )
    finally:
        session.close()

    generated = [column for column in repaired.columns if column not in KEYS]
    if len(generated) != len(factor_names):
        raise ValueError(
            f"Recomputed factor count {len(generated)} != Alpha Pool {len(factor_names)}"
        )
    repaired = repaired.rename(columns=dict(zip(generated, factor_names)))
    combined = merge_repaired_factor_range(
        existing, repaired, factor_names, repair_start, repair_end
    )
    report = validate_repaired_range(
        combined,
        factor_names,
        repair_start,
        repair_end,
        minimum_coverage,
    )
    _write_matrix(combined, repaired_pickle)
    _write_matrix(combined, repaired_csv)
    report.update({
        "source_alpha_pool": str(metadata_path),
        "source_daily_price": str(daily_path),
        "repaired_pickle": str(repaired_pickle),
        "repaired_csv": str(repaired_csv),
        "replace_requested": bool(args.replace),
    })
    report_path = repair_root / "repair_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FactorRepair] validation={json.dumps(report, ensure_ascii=False)}", flush=True)

    if args.replace:
        _replace_outputs(((pickle_path, repaired_pickle), (csv_path, repaired_csv)))
    else:
        print(
            "[FactorRepair] dry_replace=true repaired files were validated but originals "
            "were not changed; rerun the same command with --replace",
            flush=True,
        )


if __name__ == "__main__":
    main()
