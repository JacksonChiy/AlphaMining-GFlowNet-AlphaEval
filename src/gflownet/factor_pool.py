from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.expression import SubexpressionCache, expression_from_tokens
from src.utils import slice_date_range


def execute_saved_alpha_pool(
    data: pd.DataFrame,
    metadata_path: str | Path = "results/alpha_pool.csv",
    matrix_path: str | Path = "results/alpha_factor_matrix.pkl",
    oos_matrix_path: str | Path = "results/alpha_factor_matrix_oos.pkl",
    oos_start_date: str | None = None,
    oos_end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rebuild expressions from saved prefix tokens and execute them on complete history."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Alpha pool metadata not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required = {"factor", "tokens"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Alpha pool metadata is missing columns: {missing}")
    if metadata.empty:
        raise ValueError("Alpha pool metadata is empty")

    matrix = data[["date", "code"]].copy()
    ordered = data.sort_values(["code", "date"], kind="stable")
    expression_cache = SubexpressionCache(ordered)
    print(
        f"[FactorPool] execution_start factors={len(metadata)} rows={len(data):,} "
        f"start={data['date'].min()} end={data['date'].max()}",
        flush=True,
    )
    for index, row in metadata.reset_index(drop=True).iterrows():
        raw_tokens = row["tokens"]
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Invalid token sequence for factor {row['factor']}")
        expression = expression_from_tokens(tokens)
        values = pd.to_numeric(
            expression.execute(ordered, cache=expression_cache).reindex(data.index),
            errors="coerce",
        ).replace(
            [np.inf, -np.inf], np.nan
        )
        matrix[str(row["factor"])] = values.to_numpy()
        print(
            f"[FactorPool] factor_complete index={index + 1:03d}/{len(metadata):03d} "
            f"factor={row['factor']} coverage={values.notna().mean():.2%} "
            f"expression={expression}",
            flush=True,
        )

    matrix_path = Path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_pickle(matrix_path)
    oos_matrix: pd.DataFrame | None = None
    if oos_start_date is not None or oos_end_date is not None:
        oos_matrix = slice_date_range(
            matrix,
            oos_start_date,
            oos_end_date,
            label="out-of-sample factor matrix",
        )
        oos_matrix_path = Path(oos_matrix_path)
        oos_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        oos_matrix.to_pickle(oos_matrix_path)
    cache = expression_cache.stats()
    print(
        f"[FactorPool] execution_complete full_rows={len(matrix):,} "
        f"oos_rows={len(oos_matrix) if oos_matrix is not None else 0:,} "
        f"cache_hits={cache['hits']} cache_misses={cache['misses']} "
        f"cache_hit_rate={cache['hit_rate']:.2%} "
        f"cache_memory_mb={cache['memory_mb']:.1f}",
        flush=True,
    )
    return matrix, oos_matrix
