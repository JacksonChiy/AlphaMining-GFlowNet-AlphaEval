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
        reward_floor: float = 1e-8,
    ) -> None:
        self.data = data.sort_values(["code", "date"], kind="stable").copy()
        self.horizon = horizon
        self.top_quantile = top_quantile
        self.risk_aversion = risk_aversion
        self.min_cross_section = min_cross_section
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
            result = RewardBreakdown(
                self.reward_floor, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0,
                "industry" in self.data, "market_cap" in self.data,
            )
        self.cache[key] = result
        return result

    def evaluate_factor(self, factor: pd.Series) -> RewardBreakdown:
        work = self.data[["date", "code", "_target"]].copy()
        work["factor"] = pd.to_numeric(factor.reindex(work.index), errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor", "_target"])
        counts = work.groupby("date", observed=True).size()
        valid_dates = counts[counts >= self.min_cross_section].index
        work = work[work["date"].isin(valid_dates)]
        if work.empty:
            return RewardBreakdown(
                self.reward_floor, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0,
                "industry" in self.data, "market_cap" in self.data,
            )

        rank_ic_series = work.groupby("date", observed=True)[["factor", "_target"]].apply(
            lambda x: x["factor"].corr(x["_target"], method="spearman")
        ).dropna()
        rank_ic = float(rank_ic_series.mean()) if len(rank_ic_series) else 0.0

        def long_excess(group: pd.DataFrame) -> float:
            cutoff = group["factor"].quantile(1.0 - self.top_quantile)
            selected = group.loc[group["factor"] >= cutoff, "_target"]
            return float(selected.mean() - group["_target"].mean()) if len(selected) else np.nan

        excess = work.groupby("date", observed=True)[["factor", "_target"]].apply(long_excess).dropna()
        periods = 252.0 / (self.horizon - 1)
        excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        long_ir = float(excess.mean() / excess_std * math.sqrt(periods)) if excess_std > 0 else 0.0
        annualized = float(excess.mean() * periods) if len(excess) else 0.0

        aligned = self.data.loc[work.index].copy()
        aligned["factor"] = work["factor"]
        industry_exposure = self._industry_exposure(aligned)
        size_exposure = self._size_exposure(aligned)
        risk_penalty = math.exp(-self.risk_aversion * (industry_exposure + size_exposure))
        reward = abs(rank_ic) * max(0.05, 1.0 + np.clip(long_ir, -0.95, 5.0)) * risk_penalty
        reward = float(max(self.reward_floor, reward))
        return RewardBreakdown(
            reward=reward,
            rank_ic=rank_ic,
            long_ir=long_ir,
            annualized_long_excess=annualized,
            risk_penalty=risk_penalty,
            industry_exposure=industry_exposure,
            size_exposure=size_exposure,
            observations=len(work),
            industry_penalty_applied="industry" in aligned,
            size_penalty_applied="market_cap" in aligned,
        )

    @staticmethod
    def _size_exposure(work: pd.DataFrame) -> float:
        if "market_cap" not in work:
            return 0.0
        values: list[float] = []
        for _, group in work.dropna(subset=["market_cap"]).groupby("date", observed=True):
            if len(group) >= 5:
                corr = group["factor"].corr(np.log1p(group["market_cap"].clip(lower=0)), method="spearman")
                if pd.notna(corr):
                    values.append(abs(float(corr)))
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _industry_exposure(work: pd.DataFrame) -> float:
        if "industry" not in work:
            return 0.0
        values: list[float] = []
        for _, group in work.dropna(subset=["industry"]).groupby("date", observed=True):
            if group["industry"].nunique() < 2 or len(group) < 8:
                continue
            y = group["factor"].to_numpy(float)
            dummies = pd.get_dummies(group["industry"], drop_first=False, dtype=float).to_numpy()
            fitted = dummies @ np.linalg.lstsq(dummies, y, rcond=None)[0]
            total = np.square(y - y.mean()).sum()
            r2 = 1.0 - np.square(y - fitted).sum() / total if total > 0 else 0.0
            values.append(float(np.clip(r2, 0.0, 1.0)))
        return float(np.mean(values)) if values else 0.0
