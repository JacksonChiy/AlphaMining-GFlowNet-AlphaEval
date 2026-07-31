"""Benchmark-aware portfolio construction for index enhancement.

The optimizer consumes only persisted prediction and point-in-time index-weight
files.  It uses a diagonal benchmark-relative risk proxy so it can run inside a
local RQAlphaPlus strategy without an external risk-model service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .universe import normalize_order_book_id


@dataclass(frozen=True)
class PortfolioOptimizerConfig:
    max_names: int = 200
    alpha_top_n: int = 100
    cash_buffer: float = 0.98
    alpha_strength: float = 0.05
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.01
    max_active_weight: float = 0.01
    max_stock_weight: float = 0.10
    max_rebalance_turnover: float = 0.20
    estimated_buy_cost: float = 0.0018
    estimated_sell_cost: float = 0.0023
    risk_weight_floor: float = 0.001
    max_iterations: int = 200
    tolerance: float = 1e-9


def load_weight_history_for_strategy(
    path: str | Path,
    index_key: str,
    start_date=None,
    end_date=None,
    chunksize: int = 500_000,
) -> dict:
    """Load one index from the local long-form weight file into date mappings."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Index weight file not found: {source.resolve()}")
    frames = []
    columns = ["date", "index_key", "code", "benchmark_weight"]
    for chunk in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        selected = chunk.loc[chunk["index_key"].astype(str).eq(str(index_key))].copy()
        if selected.empty:
            continue
        selected["date"] = pd.to_datetime(selected["date"]).dt.normalize()
        if start_date is not None:
            selected = selected.loc[selected["date"] >= pd.Timestamp(start_date).normalize()]
        if end_date is not None:
            selected = selected.loc[selected["date"] <= pd.Timestamp(end_date).normalize()]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise ValueError(f"No index weights found for index_key={index_key!r}")
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].map(normalize_order_book_id)
    frame["benchmark_weight"] = pd.to_numeric(frame["benchmark_weight"], errors="coerce")
    frame = frame.dropna(subset=["benchmark_weight"])
    if frame.duplicated(["date", "code"]).any():
        raise ValueError(f"Duplicate index weights for index_key={index_key!r}")
    if (frame["benchmark_weight"] < 0).any():
        raise ValueError("Benchmark weights must be non-negative")
    totals = frame.groupby("date", observed=True)["benchmark_weight"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Benchmark weights contain a zero-sum date")
    frame["benchmark_weight"] = frame["benchmark_weight"] / totals
    return {
        timestamp.date(): group.set_index("code")["benchmark_weight"].sort_values(ascending=False)
        for timestamp, group in frame.groupby("date", observed=True, sort=True)
    }


def latest_weight_date(weight_dates: list, signal_date):
    """Return the latest point-in-time weight date no later than the signal date."""
    eligible = [date for date in weight_dates if date <= signal_date]
    return eligible[-1] if eligible else None


def _bounded_simplex_projection(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
) -> np.ndarray:
    """Euclidean projection onto box bounds plus a fixed-sum constraint."""
    if lower.sum() > total + 1e-10 or upper.sum() < total - 1e-10:
        raise ValueError("Portfolio bounds are infeasible for the requested total weight")
    lo = float(np.min(values - upper)) - 1.0
    hi = float(np.max(values - lower)) + 1.0
    for _ in range(100):
        midpoint = (lo + hi) / 2
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > total:
            lo = midpoint
        else:
            hi = midpoint
    result = np.clip(values - (lo + hi) / 2, lower, upper)
    residual = total - result.sum()
    if abs(residual) > 1e-10:
        capacity = upper - result if residual > 0 else result - lower
        valid = capacity > 1e-12
        if valid.any():
            result[valid] += residual * capacity[valid] / capacity[valid].sum()
    return result


def _candidate_codes(
    ranked: pd.DataFrame,
    benchmark: pd.Series,
    current_weights: dict[str, float],
    config: PortfolioOptimizerConfig,
) -> list[str]:
    ranked_codes = ranked["code"].astype(str).tolist()
    benchmark_codes = benchmark.index.astype(str).tolist()
    available = set(ranked_codes) | set(current_weights)
    candidates: list[str] = []

    def extend(values):
        for code in values:
            if code in available and code not in candidates:
                candidates.append(code)

    extend(sorted(current_weights, key=current_weights.get, reverse=True))
    extend(ranked_codes[: config.alpha_top_n])
    extend(benchmark_codes)
    limit = max(config.max_names, len(current_weights), config.alpha_top_n)
    return candidates[:limit]


def optimize_benchmark_portfolio(
    ranked: pd.DataFrame,
    benchmark_weights: pd.Series,
    current_weights: dict[str, float] | None = None,
    config: PortfolioOptimizerConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Optimize benchmark-relative weights with risk, turnover and cost penalties."""
    config = config or PortfolioOptimizerConfig()
    current_weights = {
        normalize_order_book_id(code): max(0.0, float(weight))
        for code, weight in (current_weights or {}).items()
        if np.isfinite(weight) and weight > 0
    }
    required = {"code", "smoothed_rank_score"}
    missing = required.difference(ranked.columns)
    if missing:
        raise ValueError(f"Ranked score frame missing columns: {sorted(missing)}")
    frame = ranked[["code", "smoothed_rank_score"]].copy()
    frame["code"] = frame["code"].map(normalize_order_book_id)
    frame["smoothed_rank_score"] = pd.to_numeric(
        frame["smoothed_rank_score"], errors="coerce"
    )
    frame = frame.dropna().drop_duplicates("code", keep="first")
    benchmark = benchmark_weights.copy()
    benchmark.index = benchmark.index.map(normalize_order_book_id)
    benchmark = pd.to_numeric(benchmark, errors="coerce").dropna()
    benchmark = benchmark[benchmark >= 0]
    if benchmark.empty or benchmark.sum() <= 0:
        raise ValueError("Benchmark weight cross-section is empty")
    benchmark = benchmark / benchmark.sum()

    codes = _candidate_codes(frame, benchmark, current_weights, config)
    if not codes:
        raise ValueError("No eligible portfolio candidates")
    table = pd.DataFrame({"code": codes})
    table = table.merge(frame, on="code", how="left")
    table["smoothed_rank_score"] = table["smoothed_rank_score"].fillna(0.0)
    table["benchmark_weight_raw"] = table["code"].map(benchmark).fillna(0.0)
    raw_coverage = float(table["benchmark_weight_raw"].sum())
    if raw_coverage <= 0:
        raise ValueError("Selected candidates have zero benchmark-weight coverage")

    benchmark_target = (
        table["benchmark_weight_raw"].to_numpy(dtype=float)
        / raw_coverage
        * config.cash_buffer
    )
    alpha = table["smoothed_rank_score"].to_numpy(dtype=float)
    alpha = alpha - np.average(alpha, weights=np.maximum(benchmark_target, 1e-12))
    alpha_std = float(np.std(alpha))
    if alpha_std > 1e-12:
        alpha = alpha / alpha_std

    previous = np.array([current_weights.get(code, 0.0) for code in codes], dtype=float)
    current_coverage = float(previous.sum())
    if current_coverage < config.cash_buffer:
        previous += (config.cash_buffer - current_coverage) * benchmark_target / benchmark_target.sum()
    elif current_coverage > config.cash_buffer:
        previous *= config.cash_buffer / current_coverage

    lower = np.maximum(0.0, benchmark_target - config.max_active_weight)
    stock_cap = np.maximum(config.max_stock_weight, benchmark_target)
    upper = np.minimum(stock_cap, benchmark_target + config.max_active_weight)
    initial = _bounded_simplex_projection(previous, lower, upper, config.cash_buffer)
    minimum_turnover = float(np.abs(initial - previous).sum())
    turnover_budget = max(config.max_rebalance_turnover, minimum_turnover + 1e-8)
    # One basis point squared keeps the absolute-value approximation smooth at
    # zero without materially changing realistic portfolio turnover costs.
    smooth_eps = 1e-8

    def smooth_positive(values):
        root = np.sqrt(values * values + smooth_eps)
        return (values + root) / 2, (-values + root) / 2

    def objective(weights):
        active = weights - benchmark_target
        delta = weights - previous
        buy, sell = smooth_positive(delta)
        alpha_value = -config.alpha_strength * float(np.dot(alpha, weights))
        risk_value = 0.5 * config.risk_aversion * float(
            np.sum(active * active / np.maximum(benchmark_target, config.risk_weight_floor))
        )
        turnover_value = config.turnover_penalty * float(
            np.sum(np.sqrt(delta * delta + smooth_eps))
        )
        cost_value = float(
            config.estimated_buy_cost * buy.sum()
            + config.estimated_sell_cost * sell.sum()
        )
        return alpha_value + risk_value + turnover_value + cost_value

    def objective_gradient(weights):
        active = weights - benchmark_target
        delta = weights - previous
        root = np.sqrt(delta * delta + smooth_eps)
        buy_gradient = 0.5 * (1.0 + delta / root)
        sell_gradient = 0.5 * (-1.0 + delta / root)
        return (
            -config.alpha_strength * alpha
            + config.risk_aversion
            * active
            / np.maximum(benchmark_target, config.risk_weight_floor)
            + config.turnover_penalty * delta / root
            + config.estimated_buy_cost * buy_gradient
            + config.estimated_sell_cost * sell_gradient
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - config.cash_buffer),
            "jac": lambda weights: np.ones_like(weights),
        },
    ]
    result = minimize(
        objective,
        initial,
        jac=objective_gradient,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={
            "maxiter": config.max_iterations,
            "ftol": config.tolerance,
            "disp": False,
        },
    )
    solver_success = bool(result.success and np.isfinite(result.x).all())
    weights = result.x if solver_success else initial
    solver_method = "slsqp"
    fallback_iterations = 0
    if not solver_success:
        # Index reconstitutions can make many previous holdings hit a new lower
        # or upper bound at once.  A projected-gradient fallback is deterministic
        # and optimizes the same convex objective instead of returning the
        # benchmark projection unchanged when SLSQP reaches its iteration limit.
        solver_method = "projected_gradient"
        current = initial.copy()
        current_value = objective(current)
        for iteration in range(1, 301):
            gradient = objective_gradient(current)
            step = 0.05
            accepted = False
            for _ in range(30):
                candidate = _bounded_simplex_projection(
                    current - step * gradient,
                    lower,
                    upper,
                    config.cash_buffer,
                )
                candidate_value = objective(candidate)
                if candidate_value < current_value - 1e-12:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            fallback_iterations = iteration
            movement = float(np.max(np.abs(candidate - current)))
            current, current_value = candidate, candidate_value
            if movement < config.tolerance:
                break
        if current_value < objective(initial) - 1e-10:
            weights = current
            solver_success = True
    weights = _bounded_simplex_projection(weights, lower, upper, config.cash_buffer)
    # The L1 turnover constraint is non-smooth and made high-dimensional SLSQP
    # runs hit their iteration limit.  Both endpoints below already satisfy the
    # box and full-investment constraints, so a line search preserves feasibility
    # while enforcing the exact turnover budget deterministically.
    if np.abs(weights - previous).sum() > turnover_budget + 1e-10:
        low, high = 0.0, 1.0
        for _ in range(60):
            midpoint = (low + high) / 2
            candidate = initial + midpoint * (weights - initial)
            if np.abs(candidate - previous).sum() <= turnover_budget:
                low = midpoint
            else:
                high = midpoint
        weights = initial + low * (weights - initial)
    table["benchmark_weight"] = benchmark_target
    table["target_weight"] = weights
    table["active_weight"] = weights - benchmark_target
    table["alpha_score"] = alpha
    table = table.loc[table["target_weight"] > 1e-8].sort_values(
        "target_weight", ascending=False
    ).reset_index(drop=True)
    delta = weights - previous
    diagnostics = {
        "success": solver_success,
        "status": int(result.status),
        "message": (
            str(result.message)
            if solver_method == "slsqp"
            else f"{result.message}; projected-gradient fallback"
        ),
        "solver_method": solver_method,
        "fallback_iterations": fallback_iterations,
        "objective": float(objective(weights)),
        "candidate_count": int(len(table)),
        "benchmark_weight_coverage": raw_coverage,
        "gross_target_turnover": float(np.abs(delta).sum()),
        "max_active_weight": float(np.abs(weights - benchmark_target).max()),
        "estimated_explicit_cost": float(
            config.estimated_buy_cost * np.maximum(delta, 0).sum()
            + config.estimated_sell_cost * np.maximum(-delta, 0).sum()
        ),
    }
    return table, diagnostics
