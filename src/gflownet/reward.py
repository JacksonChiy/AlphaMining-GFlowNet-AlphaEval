from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.expression import Expression


def make_forward_return(data: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Return from t+1 to t+horizon, used only as an evaluation/training label."""
    if horizon <= 1:
        raise ValueError("horizon must be greater than 1 for a t+1 to t+horizon label")
    ordered = data.sort_values(["code", "date"], kind="stable")
    grouped_close = ordered.groupby("code", observed=True)["close"]
    entry = grouped_close.shift(-1)
    exit_ = grouped_close.shift(-horizon)
    label = exit_ / entry - 1.0
    return label.reindex(data.index)


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    rank_ic: float
    long_ir: float
    annualized_long_excess: float
    risk_penalty: float
    coverage: float
    valid_date_coverage: float
    coverage_penalty: float
    industry_exposure: float
    size_exposure: float
    observations: int
    industry_penalty_applied: bool
    size_penalty_applied: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


class RewardEvaluator:
    def __init__(
        self,
        data: pd.DataFrame,
        horizon: int = 5,
        top_quantile: float = 0.10,
        risk_aversion: float = 1.0,
        min_cross_section: int = 20,
        min_coverage: float = 0.80,
        coverage_penalty_power: float = 2.0,
        reward_floor: float = 1e-8,
    ) -> None:
        if not 0.0 < min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        if coverage_penalty_power <= 0.0:
            raise ValueError("coverage_penalty_power must be positive")
        self.data = data.sort_values(["code", "date"], kind="stable").copy()
        self.horizon = horizon
        self.top_quantile = top_quantile
        self.risk_aversion = risk_aversion
        self.min_cross_section = min_cross_section
        self.min_coverage = min_coverage
        self.coverage_penalty_power = coverage_penalty_power
        self.reward_floor = reward_floor
        self.data["_target"] = make_forward_return(self.data, horizon)
        self.cache: dict[str, RewardBreakdown] = {}

    def evaluate(self, expression: Expression) -> RewardBreakdown:
        key = str(expression)
        if key in self.cache:
            return self.cache[key]
        try:
            factor = expression.execute(self.data)
            result = self.evaluate_factor(factor)
        except (FloatingPointError, ValueError, KeyError, OverflowError):
            result = self._empty_breakdown()
        self.cache[key] = result
        return result

    def evaluate_factor(self, factor: pd.Series) -> RewardBreakdown:
        work = self.data[["date", "code", "_target"]].copy()
        work["factor"] = pd.to_numeric(factor.reindex(work.index), errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan)

        eligible = work.dropna(subset=["_target"])
        eligible_counts = eligible.groupby("date", observed=True).size()
        eligible_dates = eligible_counts[eligible_counts >= self.min_cross_section].index
        eligible = eligible[eligible["date"].isin(eligible_dates)]
        if eligible.empty:
            return self._empty_breakdown()

        valid = eligible.dropna(subset=["factor"])
        coverage = float(len(valid) / len(eligible))
        valid_counts = valid.groupby("date", observed=True).size()
        valid_dates = valid_counts[valid_counts >= self.min_cross_section].index
        valid_date_coverage = float(len(valid_dates) / len(eligible_dates))
        effective_coverage = min(coverage, valid_date_coverage)
        coverage_penalty = float(
            min(1.0, (effective_coverage / self.min_coverage) ** self.coverage_penalty_power)
        )

        work = valid[valid["date"].isin(valid_dates)]
        if work.empty:
            return self._empty_breakdown(
                coverage=coverage,
                valid_date_coverage=valid_date_coverage,
                coverage_penalty=coverage_penalty,
            )

        factor_rank = work.groupby("date", observed=True)["factor"].rank(method="average")
        target_rank = work.groupby("date", observed=True)["_target"].rank(method="average")
        rank_ic_series = self._grouped_correlation(factor_rank, target_rank, work["date"])
        rank_ic = float(rank_ic_series.mean()) if len(rank_ic_series) else 0.0

        grouped = work.groupby("date", observed=True)
        cutoffs = grouped["factor"].quantile(1.0 - self.top_quantile)
        selected_mask = work["factor"] >= work["date"].map(cutoffs)
        selected_mean = work.loc[selected_mask].groupby("date", observed=True)["_target"].mean()
        market_mean = grouped["_target"].mean()
        excess = selected_mean.sub(market_mean).dropna()
        periods = 252.0 / (self.horizon - 1)
        excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        long_ir = float(excess.mean() / excess_std * math.sqrt(periods)) if excess_std > 0 else 0.0
        annualized = float(excess.mean() * periods) if len(excess) else 0.0

        aligned = self.data.loc[work.index].copy()
        aligned["factor"] = work["factor"]
        industry_exposure = self._industry_exposure(aligned)
        size_exposure = self._size_exposure(aligned)
        risk_penalty = math.exp(-self.risk_aversion * (industry_exposure + size_exposure))
        reward = (
            abs(rank_ic)
            * max(0.05, 1.0 + np.clip(long_ir, -0.95, 5.0))
            * risk_penalty
            * coverage_penalty
        )
        reward = float(max(self.reward_floor, reward))
        return RewardBreakdown(
            reward=reward,
            rank_ic=rank_ic,
            long_ir=long_ir,
            annualized_long_excess=annualized,
            risk_penalty=risk_penalty,
            coverage=coverage,
            valid_date_coverage=valid_date_coverage,
            coverage_penalty=coverage_penalty,
            industry_exposure=industry_exposure,
            size_exposure=size_exposure,
            observations=len(work),
            industry_penalty_applied="industry" in aligned,
            size_penalty_applied="market_cap" in aligned,
        )

    def _empty_breakdown(
        self,
        coverage: float = 0.0,
        valid_date_coverage: float = 0.0,
        coverage_penalty: float = 0.0,
    ) -> RewardBreakdown:
        return RewardBreakdown(
            reward=self.reward_floor,
            rank_ic=0.0,
            long_ir=0.0,
            annualized_long_excess=0.0,
            risk_penalty=1.0,
            coverage=coverage,
            valid_date_coverage=valid_date_coverage,
            coverage_penalty=coverage_penalty,
            industry_exposure=0.0,
            size_exposure=0.0,
            observations=0,
            industry_penalty_applied="industry" in self.data,
            size_penalty_applied="market_cap" in self.data,
        )

    @staticmethod
    def _grouped_correlation(left: pd.Series, right: pd.Series, dates: pd.Series) -> pd.Series:
        frame = pd.DataFrame({"date": dates, "left": left, "right": right}).dropna()
        grouped = frame.groupby("date", observed=True)
        left_centered = frame["left"] - grouped["left"].transform("mean")
        right_centered = frame["right"] - grouped["right"].transform("mean")
        numerator = (left_centered * right_centered).groupby(frame["date"], observed=True).sum()
        left_ss = left_centered.pow(2).groupby(frame["date"], observed=True).sum()
        right_ss = right_centered.pow(2).groupby(frame["date"], observed=True).sum()
        denominator = np.sqrt(left_ss * right_ss).replace(0.0, np.nan)
        return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _size_exposure(work: pd.DataFrame) -> float:
        if "market_cap" not in work:
            return 0.0
        valid = work.dropna(subset=["factor", "market_cap"]).copy()
        counts = valid.groupby("date", observed=True).size()
        valid = valid[valid["date"].isin(counts[counts >= 5].index)]
        if valid.empty:
            return 0.0
        factor_rank = valid.groupby("date", observed=True)["factor"].rank(method="average")
        log_size = np.log1p(valid["market_cap"].clip(lower=0.0))
        size_rank = log_size.groupby(valid["date"], observed=True).rank(method="average")
        correlations = RewardEvaluator._grouped_correlation(
            factor_rank, size_rank, valid["date"]
        )
        return float(correlations.abs().mean()) if len(correlations) else 0.0

    @staticmethod
    def _industry_exposure(work: pd.DataFrame) -> float:
        if "industry" not in work:
            return 0.0
        valid = work.dropna(subset=["factor", "industry"]).copy()
        grouped = valid.groupby("date", observed=True)
        counts = grouped.size()
        industry_counts = grouped["industry"].nunique()
        valid_dates = counts[(counts >= 8) & (industry_counts >= 2)].index
        valid = valid[valid["date"].isin(valid_dates)]
        if valid.empty:
            return 0.0
        daily_mean = valid.groupby("date", observed=True)["factor"].transform("mean")
        industry_mean = valid.groupby(
            ["date", "industry"], observed=True
        )["factor"].transform("mean")
        total_ss = (valid["factor"] - daily_mean).pow(2).groupby(
            valid["date"], observed=True
        ).sum()
        residual_ss = (valid["factor"] - industry_mean).pow(2).groupby(
            valid["date"], observed=True
        ).sum()
        r_squared = (1.0 - residual_ss.div(total_ss.replace(0.0, np.nan))).clip(0.0, 1.0)
        return float(r_squared.dropna().mean()) if r_squared.notna().any() else 0.0
