"""基于本地 RQData 风险暴露的指数增强组合归因。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STYLE_FACTORS = [
    "beta",
    "book_to_price",
    "comovement",
    "dividend_yield",
    "earnings_quality",
    "earnings_variability",
    "earnings_yield",
    "growth",
    "industry_momentum",
    "investment_quality",
    "leverage",
    "liquidity",
    "longterm_reversal",
    "mid_cap",
    "momentum",
    "profitability",
    "residual_volatility",
    "seasonality",
    "sentiment",
    "shortterm_reversal",
    "size",
]


@dataclass(frozen=True)
class AttributionConfig:
    start_date: str = "2024-01-02"
    end_date: str = "2026-07-17"
    ridge_alpha: float = 1e-4
    return_winsor_lower: float = 0.01
    return_winsor_upper: float = 0.99
    minimum_cross_section: int = 200
    minimum_weight_coverage: float = 0.98


def _read_exposure_files(
    root: str | Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    root = Path(root)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    frames = []
    for path in sorted(root.rglob("*.parquet")):
        year_parts = [part for part in path.parts if part.startswith("year=")]
        if year_parts:
            year = int(year_parts[-1].split("=", 1)[1])
            if year < start.year or year > end.year:
                continue
        frame = pd.read_parquet(path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame = frame.loc[frame["date"].between(start, end)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError(f"没有找到区间内的风险暴露：{root}")
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["order_book_id"] = result["order_book_id"].astype(str)
    result = result.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    if result.duplicated(["date", "order_book_id"]).any():
        raise ValueError("风险暴露存在重复的 date/order_book_id")
    return result


def load_factor_exposures(
    root: str | Path,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    exposure = _read_exposure_files(root, start_date, end_date)
    styles = [factor for factor in STYLE_FACTORS if factor in exposure.columns]
    industries = [
        column
        for column in exposure.columns
        if column not in {"date", "order_book_id", *styles}
    ]
    if len(styles) != len(STYLE_FACTORS):
        missing = sorted(set(STYLE_FACTORS).difference(styles))
        raise ValueError(f"风险暴露缺少风格因子：{missing}")
    if not industries:
        raise ValueError("风险暴露中没有行业因子")
    factors = styles + industries
    exposure[factors] = exposure[factors].astype("float32")
    return exposure[["date", "order_book_id", *factors]], styles, industries


def load_price_returns(
    path: str | Path,
    start_date: str,
    end_date: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    warmup = start - pd.Timedelta(days=15)
    frames = []
    for chunk in pd.read_csv(
        path,
        usecols=["order_book_id", "date", "close"],
        chunksize=chunksize,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"]).dt.normalize()
        chunk = chunk.loc[chunk["date"].between(warmup, end)]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        raise ValueError("价格文件在指定区间内没有数据")
    price = pd.concat(frames, ignore_index=True)
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.sort_values(["order_book_id", "date"])
    price["return"] = price.groupby("order_book_id", observed=True)["close"].pct_change()
    price = price.loc[price["date"].between(start, end)]
    return price[["date", "order_book_id", "return"]].dropna()


def estimate_factor_returns(
    exposure: pd.DataFrame,
    returns: pd.DataFrame,
    factors: list[str],
    config: AttributionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lagged = exposure.sort_values(["order_book_id", "date"]).copy()
    lagged[factors] = lagged.groupby("order_book_id", observed=True)[factors].shift(1)
    regression = lagged.merge(
        returns,
        on=["date", "order_book_id"],
        how="inner",
        validate="one_to_one",
    )
    regression = regression.dropna(subset=["return", *factors])
    estimates = []
    diagnostics = []
    identity = np.eye(len(factors), dtype=float)
    for date, group in regression.groupby("date", observed=True, sort=True):
        if len(group) < max(config.minimum_cross_section, len(factors) * 2):
            continue
        y = group["return"].to_numpy(dtype=float)
        lower, upper = np.quantile(
            y, [config.return_winsor_lower, config.return_winsor_upper]
        )
        y = np.clip(y, lower, upper)
        x = group[factors].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x, y = x[valid], y[valid]
        if len(y) < max(config.minimum_cross_section, len(factors) * 2):
            continue
        lhs = x.T @ x + config.ridge_alpha * identity
        rhs = x.T @ y
        coefficients = np.linalg.solve(lhs, rhs)
        fitted = x @ coefficients
        residual = y - fitted
        denominator = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan
        estimates.append(pd.Series(coefficients, index=factors, name=date))
        diagnostics.append(
            {
                "date": date,
                "observations": len(y),
                "r_squared": r_squared,
                "residual_std": float(np.std(residual, ddof=1)),
                "cross_section_return_std": float(np.std(y, ddof=1)),
            }
        )
    if not estimates:
        raise ValueError("没有生成任何截面因子收益回归")
    factor_returns = pd.DataFrame(estimates)
    factor_returns.index.name = "date"
    return factor_returns.sort_index(), pd.DataFrame(diagnostics)


def load_benchmark_weights(
    path: str | Path,
    index_key: str,
    start_date: str,
    end_date: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    frames = []
    for chunk in pd.read_csv(
        path,
        usecols=["date", "index_key", "code", "benchmark_weight"],
        chunksize=chunksize,
    ):
        selected = chunk.loc[chunk["index_key"].astype(str).eq(index_key)].copy()
        if selected.empty:
            continue
        selected["date"] = pd.to_datetime(selected["date"]).dt.normalize()
        selected = selected.loc[selected["date"].between(start, end)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise ValueError(f"没有找到 {index_key} 的基准权重")
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"code": "order_book_id"})
    result["benchmark_weight"] = pd.to_numeric(
        result["benchmark_weight"], errors="coerce"
    )
    result = result.dropna(subset=["benchmark_weight"])
    totals = result.groupby("date", observed=True)["benchmark_weight"].transform("sum")
    result["benchmark_weight"] /= totals
    return result[["date", "order_book_id", "benchmark_weight"]]


def load_portfolio_weights(path: str | Path) -> pd.DataFrame:
    positions = pd.read_csv(path, usecols=["date", "order_book_id", "market_value"])
    positions["date"] = pd.to_datetime(positions["date"]).dt.normalize()
    positions["market_value"] = pd.to_numeric(positions["market_value"], errors="coerce")
    positions = positions.dropna(subset=["market_value"])
    positions = positions.loc[positions["market_value"] > 0]
    totals = positions.groupby("date", observed=True)["market_value"].transform("sum")
    positions["portfolio_weight"] = positions["market_value"] / totals
    return positions[["date", "order_book_id", "portfolio_weight"]]


def _weighted_factor_exposure(
    weights: pd.DataFrame,
    exposure: pd.DataFrame,
    weight_column: str,
    factors: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    merged = weights.merge(
        exposure,
        on=["date", "order_book_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    total = merged.groupby("date", observed=True)[weight_column].sum()
    covered = (
        merged.loc[merged["_merge"].eq("both")]
        .groupby("date", observed=True)[weight_column]
        .sum()
    )
    coverage = covered.div(total).rename("coverage")
    valid = merged.loc[merged["_merge"].eq("both")].copy()
    covered_weight = valid.groupby("date", observed=True)[weight_column].transform("sum")
    valid["normalized_weight"] = valid[weight_column] / covered_weight
    weighted = valid[factors].mul(valid["normalized_weight"], axis=0)
    weighted.insert(0, "date", valid["date"].to_numpy())
    result = weighted.groupby("date", observed=True)[factors].sum()
    return result, coverage


def compute_active_exposure(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    exposure: pd.DataFrame,
    factors: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    portfolio, portfolio_coverage = _weighted_factor_exposure(
        portfolio_weights, exposure, "portfolio_weight", factors
    )
    benchmark, benchmark_coverage = _weighted_factor_exposure(
        benchmark_weights, exposure, "benchmark_weight", factors
    )
    dates = portfolio.index.intersection(benchmark.index)
    active = portfolio.loc[dates] - benchmark.loc[dates]
    coverage = pd.concat(
        [
            portfolio_coverage.rename("portfolio_coverage"),
            benchmark_coverage.rename("benchmark_coverage"),
        ],
        axis=1,
    ).loc[dates]
    active.index.name = "date"
    return active, coverage


def load_realized_active_returns(path: str | Path) -> pd.Series:
    portfolio = pd.read_csv(
        path,
        usecols=["date", "unit_net_value", "benchmark_unit_net_value"],
    )
    portfolio["date"] = pd.to_datetime(portfolio["date"]).dt.normalize()
    portfolio = portfolio.set_index("date").sort_index()
    portfolio_return = portfolio["unit_net_value"].pct_change()
    benchmark_return = portfolio["benchmark_unit_net_value"].pct_change()
    return (portfolio_return - benchmark_return).rename("realized_active_return")


def attribute_index(
    index_key: str,
    backtest_dir: str | Path,
    benchmark_weight_path: str | Path,
    exposure: pd.DataFrame,
    factor_returns: pd.DataFrame,
    styles: list[str],
    industries: list[str],
    output_dir: str | Path,
    config: AttributionConfig,
) -> dict:
    factors = styles + industries
    backtest_dir = Path(backtest_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_weights = load_portfolio_weights(backtest_dir / "stock_positions.csv")
    benchmark_weights = load_benchmark_weights(
        benchmark_weight_path, index_key, config.start_date, config.end_date
    )
    active, coverage = compute_active_exposure(
        portfolio_weights, benchmark_weights, exposure, factors
    )
    active.to_csv(output_dir / "daily_active_exposure.csv")
    coverage.to_csv(output_dir / "exposure_coverage.csv")

    summary = pd.DataFrame(
        {
            "mean": active.mean(),
            "mean_abs": active.abs().mean(),
            "std": active.std(),
            "min": active.min(),
            "max": active.max(),
            "latest": active.iloc[-1],
            "positive_ratio": (active > 0).mean(),
        }
    )
    summary.index.name = "factor"
    summary["group"] = np.where(summary.index.isin(styles), "style", "industry")
    summary.to_csv(output_dir / "active_exposure_summary.csv")

    lagged_active = active.shift(1)
    common_dates = lagged_active.index.intersection(factor_returns.index)
    contribution = lagged_active.loc[common_dates, factors].mul(
        factor_returns.loc[common_dates, factors]
    )
    realized = load_realized_active_returns(backtest_dir / "portfolio.csv")
    contribution["style_contribution"] = contribution[styles].sum(axis=1)
    contribution["industry_contribution"] = contribution[industries].sum(axis=1)
    contribution["factor_contribution"] = (
        contribution["style_contribution"] + contribution["industry_contribution"]
    )
    contribution = contribution.join(realized, how="inner")
    contribution["residual_contribution"] = (
        contribution["realized_active_return"] - contribution["factor_contribution"]
    )
    contribution.index.name = "date"
    contribution.to_csv(output_dir / "daily_return_attribution.csv")

    factor_contribution = pd.DataFrame(
        {
            "cumulative_contribution": contribution[factors].sum(),
            "annualized_contribution": contribution[factors].mean() * 252,
            "daily_volatility": contribution[factors].std(),
        }
    )
    factor_contribution.index.name = "factor"
    factor_contribution["group"] = np.where(
        factor_contribution.index.isin(styles), "style", "industry"
    )
    factor_contribution.to_csv(output_dir / "factor_contribution_summary.csv")

    result = {
        "index_key": index_key,
        "start_date": str(contribution.index.min().date()),
        "end_date": str(contribution.index.max().date()),
        "attribution_days": int(len(contribution)),
        "mean_portfolio_exposure_coverage": float(coverage["portfolio_coverage"].mean()),
        "min_portfolio_exposure_coverage": float(coverage["portfolio_coverage"].min()),
        "mean_benchmark_exposure_coverage": float(coverage["benchmark_coverage"].mean()),
        "min_benchmark_exposure_coverage": float(coverage["benchmark_coverage"].min()),
        "cumulative_realized_active_return_additive": float(
            contribution["realized_active_return"].sum()
        ),
        "cumulative_style_contribution": float(contribution["style_contribution"].sum()),
        "cumulative_industry_contribution": float(contribution["industry_contribution"].sum()),
        "cumulative_factor_contribution": float(contribution["factor_contribution"].sum()),
        "cumulative_residual_contribution": float(contribution["residual_contribution"].sum()),
    }
    (output_dir / "attribution_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
