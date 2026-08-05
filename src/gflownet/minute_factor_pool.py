from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.expression.minute import MinuteExpression, minute_expression_from_tokens
from src.expression.dolphindb_minute import DolphinDBMinuteCompiler
from src.operators.minute import build_minute_features
from src.data_loader.dolphindb_minute import DolphinDBMinuteLoader


def _daily_keys(daily_data: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [pd.to_datetime(daily_data["date"]).dt.normalize(), daily_data["code"].astype(str)],
        names=["date", "code"],
    )


def save_minute_alpha_pool(
    pool: list[dict[str, Any]],
    minute_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    metadata_path: str | Path = "results/minute_alpha_pool.csv",
    matrix_path: str | Path = "results/minute_alpha_factor_matrix.pkl",
    min_coverage: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = [
        item for item in pool
        if float(item.get("coverage", 0.0)) >= min_coverage
        and float(item.get("valid_date_coverage", 0.0)) >= min_coverage
    ]
    if not eligible:
        raise ValueError(f"No minute expression meets minimum coverage {min_coverage:.2%}")
    prepared = build_minute_features(minute_data)
    matrix = daily_data[["date", "code"]].copy()
    keys = _daily_keys(matrix)
    metadata_rows: list[dict[str, Any]] = []
    for index, item in enumerate(eligible, start=1):
        factor_name = f"minute_factor_{index:03d}"
        expression: MinuteExpression = item["expression"]
        values = expression.execute(prepared).reindex(keys)
        matrix[factor_name] = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy()
        metadata_rows.append({
            "factor": factor_name,
            "expression": str(expression),
            **{key: value for key, value in item.items() if key not in {"expression", "tokens"}},
            "tokens": json.dumps(item["tokens"], ensure_ascii=False),
        })
        print(
            f"[MinuteFactorPool] factor_complete index={index:03d}/{len(eligible):03d} "
            f"factor={factor_name} coverage={matrix[factor_name].notna().mean():.2%} "
            f"expression={expression}",
            flush=True,
        )
    metadata = pd.DataFrame(metadata_rows)
    metadata_path, matrix_path = Path(metadata_path), Path(matrix_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    matrix.to_pickle(matrix_path)
    return metadata, matrix


def execute_saved_minute_alpha_pool(
    minute_data: pd.DataFrame,
    daily_data: pd.DataFrame,
    metadata_path: str | Path = "results/minute_alpha_pool.csv",
    matrix_path: str | Path = "results/minute_alpha_factor_matrix.pkl",
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    if metadata.empty or not {"factor", "tokens"}.issubset(metadata.columns):
        raise ValueError("Minute alpha metadata must contain non-empty factor and tokens columns")
    prepared = build_minute_features(minute_data)
    matrix = daily_data[["date", "code"]].copy()
    keys = _daily_keys(matrix)
    for index, row in metadata.iterrows():
        tokens = json.loads(row["tokens"])
        expression = minute_expression_from_tokens(tokens)
        values = expression.execute(prepared).reindex(keys)
        matrix[str(row["factor"])] = pd.to_numeric(values, errors="coerce").to_numpy()
        print(
            f"[MinuteFactorPool] saved_factor_complete index={index + 1:03d}/{len(metadata):03d} "
            f"factor={row['factor']} expression={expression}",
            flush=True,
        )
    matrix_path = Path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_pickle(matrix_path)
    return matrix


def save_minute_alpha_pool_from_cache(
    pool: list[dict[str, Any]],
    cache_dir: str | Path,
    daily_data: pd.DataFrame,
    metadata_path: str | Path = "results/minute_alpha_pool.csv",
    matrix_path: str | Path = "results/minute_alpha_factor_matrix.pkl",
    min_coverage: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute intraday-only expressions one DDB date partition at a time."""
    eligible = [
        item for item in pool
        if float(item.get("coverage", 0.0)) >= min_coverage
        and float(item.get("valid_date_coverage", 0.0)) >= min_coverage
    ]
    if not eligible:
        raise ValueError(f"No minute expression meets minimum coverage {min_coverage:.2%}")
    metadata_rows = []
    factor_names = [f"minute_factor_{index:03d}" for index in range(1, len(eligible) + 1)]
    for factor_name, item in zip(factor_names, eligible):
        metadata_rows.append({
            "factor": factor_name,
            "expression": str(item["expression"]),
            **{key: value for key, value in item.items() if key not in {"expression", "tokens"}},
            "tokens": json.dumps(item["tokens"], ensure_ascii=False),
        })

    block_nodes: dict[str, Any] = {}
    for item in eligible:
        for node in item["expression"].block_nodes():
            block_nodes.setdefault(node.render(), node)
    block_parts: dict[str, list[pd.Series]] = {key: [] for key in block_nodes}
    executor_expression: MinuteExpression = eligible[0]["expression"]
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    for partition_index, filename in enumerate(manifest["files"], start=1):
        minute_part = pd.read_pickle(cache_dir / filename)
        computed = executor_expression.execute_blocks(list(block_nodes.values()), minute_part)
        for key, values in computed.items():
            block_parts[key].append(values)
        print(
            f"[MinuteFactorPool] cache_partition_complete "
            f"index={partition_index:03d}/{len(manifest['files']):03d} "
            f"minute_rows={len(minute_part):,} blocks={len(block_nodes):03d} file={filename}",
            flush=True,
        )
    blocks: dict[str, pd.Series] = {}
    for key, parts in block_parts.items():
        values = pd.concat(parts).sort_index()
        blocks[key] = values[~values.index.duplicated(keep="last")]
    matrix = daily_data[["date", "code"]].copy()
    keys = _daily_keys(matrix)
    for factor_name, item in zip(factor_names, eligible):
        expression: MinuteExpression = item["expression"]
        required = {node.render(): blocks[node.render()] for node in expression.block_nodes()}
        values = expression.execute_from_blocks(required).reindex(keys)
        matrix[factor_name] = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy()
    metadata = pd.DataFrame(metadata_rows)
    metadata_path, matrix_path = Path(metadata_path), Path(matrix_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    matrix.to_pickle(matrix_path)
    return metadata, matrix


def save_minute_alpha_pool_from_dolphindb_stream(
    pool: list[dict[str, Any]],
    loader: DolphinDBMinuteLoader,
    daily_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    metadata_path: str | Path = "results/minute_cpu_ddb/alpha_pool.csv",
    matrix_path: str | Path = "results/minute_cpu_ddb/alpha_factor_matrix.csv.gz",
    min_coverage: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute selected report factors in one DDB scan; no raw minute file is written."""
    eligible = [
        item for item in pool
        if float(item.get("coverage", 0.0)) >= min_coverage
        and float(item.get("valid_date_coverage", 0.0)) >= min_coverage
    ]
    if not eligible:
        raise ValueError(f"No minute expression meets minimum coverage {min_coverage:.2%}")
    factor_names = [f"minute_factor_{index:03d}" for index in range(1, len(eligible) + 1)]
    block_nodes: dict[str, Any] = {}
    for item in eligible:
        for node in item["expression"].block_nodes():
            block_nodes.setdefault(node.render(), node)
    block_parts: dict[str, list[pd.Series]] = {key: [] for key in block_nodes}
    executor_expression: MinuteExpression = eligible[0]["expression"]
    nodes = list(block_nodes.values())
    compiler = DolphinDBMinuteCompiler(loader.table_expression)
    supported = (
        [node for node in nodes if compiler.supports(node)]
        if loader.config.pushdown_enabled else []
    )
    fallback = [node for node in nodes if node not in supported]
    print(
        f"[MinuteFactorPool] execution_plan pushdown_blocks={len(supported):03d} "
        f"numpy_fallback_blocks={len(fallback):03d} coarse_screen=false",
        flush=True,
    )
    if supported:
        try:
            for _, _, computed in loader.iter_minute_blocks(supported, start_date, end_date):
                for key, values in computed.items():
                    block_parts[key].append(values)
        except Exception as error:
            if not loader.config.pushdown_fallback:
                raise
            print(
                f"[MinuteFactorPool] pushdown_failed fallback_to_numpy=true "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            for node in supported:
                block_parts[node.render()].clear()
            fallback = nodes
    chunks = 0
    if fallback:
        for chunks, (_, _, minute) in enumerate(
            loader.iter_frames(start_date, end_date), start=1
        ):
            computed = executor_expression.execute_blocks(fallback, minute)
            for key, values in computed.items():
                block_parts[key].append(values)
            print(
                f"[MinuteFactorPool] numpy_stream_chunk_complete index={chunks:03d} "
                f"blocks={len(fallback):03d} factors={len(eligible):03d}",
                flush=True,
            )
    if any(not parts for parts in block_parts.values()):
        missing = [key for key, parts in block_parts.items() if not parts]
        raise ValueError(f"No values were produced for minute blocks: {missing}")
    blocks: dict[str, pd.Series] = {}
    for key, parts in block_parts.items():
        if not parts:
            continue
        values = pd.concat(parts).sort_index()
        blocks[key] = values[~values.index.duplicated(keep="last")]
    matrix = daily_data[["date", "code"]].copy()
    dates = pd.to_datetime(matrix["date"]).dt.normalize()
    matrix = matrix.loc[
        dates.between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()
    keys = _daily_keys(matrix)
    metadata_rows: list[dict[str, Any]] = []
    for index, (factor_name, item) in enumerate(zip(factor_names, eligible), start=1):
        expression: MinuteExpression = item["expression"]
        required = {node.render(): blocks[node.render()] for node in expression.block_nodes()}
        values = expression.execute_from_blocks(required).reindex(keys)
        matrix[factor_name] = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy()
        metadata_rows.append({
            "factor": factor_name,
            "expression": str(expression),
            **{key: value for key, value in item.items() if key not in {"expression", "tokens"}},
            "tokens": json.dumps(item["tokens"], ensure_ascii=False),
        })
        print(
            f"[MinuteFactorPool] factor_complete index={index:03d}/{len(eligible):03d} "
            f"factor={factor_name} coverage={matrix[factor_name].notna().mean():.2%}",
            flush=True,
        )
    metadata = pd.DataFrame(metadata_rows)
    metadata_path, matrix_path = Path(metadata_path), Path(matrix_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    if matrix_path.name.endswith(".csv.gz"):
        matrix.to_csv(matrix_path, index=False, compression="gzip")
    elif matrix_path.suffix.lower() == ".csv":
        matrix.to_csv(matrix_path, index=False)
    elif matrix_path.suffix.lower() in {".parquet", ".pq"}:
        matrix.to_parquet(matrix_path, index=False)
    elif matrix_path.suffix.lower() in {".pkl", ".pickle"}:
        matrix.to_pickle(matrix_path)
    else:
        raise ValueError("factor matrix must use .csv, .csv.gz, .parquet or .pkl")
    return metadata, matrix
