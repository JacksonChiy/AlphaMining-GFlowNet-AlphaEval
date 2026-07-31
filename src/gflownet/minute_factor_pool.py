from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.expression.minute import MinuteExpression, minute_expression_from_tokens
from src.operators.minute import build_minute_features


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
