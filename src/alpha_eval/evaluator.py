from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.gflownet.reward import make_forward_return


EPS = 1e-12


@dataclass
class AlphaEvalConfig:
    horizon: int = 5
    rolling_window: int = 60
    perturbation_std: float = 0.05
    top_quantile: float = 0.10
    min_cross_section: int = 20
    dpp_k: int = 50
    seed: int = 42


def _daily_corr(work: pd.DataFrame, method: str, min_cross_section: int) -> pd.Series:
    valid = work.dropna(subset=["factor", "target"])
    counts = valid.groupby("date", observed=True).size()
    valid = valid[valid["date"].isin(counts[counts >= min_cross_section].index)]
    return valid.groupby("date", observed=True)[["factor", "target"]].apply(
        lambda x: x["factor"].corr(x["target"], method=method)
    ).dropna()


def greedy_dpp_select(kernel: np.ndarray, k: int) -> list[int]:
    """Greedy MAP inference for a positive semidefinite DPP L-ensemble."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be square")
    n = kernel.shape[0]
    k = min(max(0, k), n)
    if k == 0:
        return []
    cis = np.zeros((k, n), dtype=float)
    residual = np.clip(np.diag(kernel).astype(float), 0.0, None)
    selected: list[int] = []
    for iteration in range(k):
        candidate = int(np.argmax(residual))
        if not np.isfinite(residual[candidate]) or residual[candidate] <= EPS:
            break
        selected.append(candidate)
        if iteration == k - 1:
            break
        previous = cis[:iteration, candidate] @ cis[:iteration] if iteration else 0.0
        cis[iteration] = (kernel[candidate] - previous) / math.sqrt(residual[candidate])
        residual = np.clip(residual - np.square(cis[iteration]), 0.0, None)
        residual[selected] = -np.inf
    return selected


class AlphaEval:
    """Deterministic mini AlphaEval: PPS, RRE, robustness, logic, and DPP diversity."""

    def __init__(self, price: pd.DataFrame, factors: pd.DataFrame, config: AlphaEvalConfig | None = None) -> None:
        self.config = config or AlphaEvalConfig()
        keys = ["date", "code"]
        if not set(keys).issubset(factors):
            raise ValueError("Factor matrix must contain date and code")
        base = price[keys + ["close"]].copy()
        base["target"] = make_forward_return(price, self.config.horizon).to_numpy()
        self.data = base.merge(factors, on=keys, how="inner", validate="one_to_one")
        self.factor_names = [column for column in factors.columns if column not in keys]
        if not self.factor_names:
            raise ValueError("No factor columns found")

    def evaluate(
        self,
        metadata: pd.DataFrame | None = None,
        output_path: str | Path | None = "results/alpha_eval_result.csv",
    ) -> pd.DataFrame:
        metadata_map = {}
        if metadata is not None and "factor" in metadata:
            metadata_map = metadata.set_index("factor").to_dict("index")
        rows = [self._evaluate_one(name, metadata_map.get(name, {})) for name in self.factor_names]
        result = pd.DataFrame(rows)
        result["pps_score"] = self._zscore(result["RankIC"].abs()) + self._zscore(result["ICIR"].abs())
        result["rre_score"] = self._zscore(result["rank_stability"]) - self._zscore(result["rank_turnover"])
        result["base_score"] = (
            0.35 * self._zscore(result["pps_score"])
            + 0.20 * self._zscore(result["temporal_stability"])
            + 0.15 * self._zscore(result["robustness"])
            + 0.15 * self._zscore(result["logic_score"])
            + 0.15 * self._zscore(result["rre_score"])
        )

        kernel, correlation = self._dpp_kernel(result)
        selected_indices = greedy_dpp_select(kernel, self.config.dpp_k)
        selected_factors = {result.iloc[index]["factor"] for index in selected_indices}
        result["dpp_selected"] = result["factor"].isin(selected_factors)
        result["diversity"] = [
            self._diversity_score(name, selected_factors, correlation) for name in result["factor"]
        ]
        result["score"] = result["base_score"] + 0.20 * self._zscore(result["diversity"])
        result = result.sort_values(["dpp_selected", "score"], ascending=[False, False]).reset_index(drop=True)
        ordered = [
            "factor", "IC", "RankIC", "ICIR", "Sharpe", "complexity", "score",
            "rolling_ic_mean", "rolling_ic_std", "temporal_stability", "robustness",
            "logic_score", "rank_stability", "rank_turnover", "diversity", "dpp_selected",
        ]
        result = result[ordered + [column for column in result.columns if column not in ordered]]
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_path, index=False)
        return result

    def _evaluate_one(self, name: str, metadata: dict[str, object]) -> dict[str, float | str]:
        work = self.data[["date", "code", "target", name]].rename(columns={name: "factor"})
        pearson = _daily_corr(work, "pearson", self.config.min_cross_section)
        rank_ic = _daily_corr(work, "spearman", self.config.min_cross_section)
        ic = float(pearson.mean()) if len(pearson) else 0.0
        ric = float(rank_ic.mean()) if len(rank_ic) else 0.0
        ric_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else 0.0
        periods = 252 / (self.config.horizon - 1)
        icir = ric / ric_std * math.sqrt(periods) if ric_std > 0 else 0.0
        rolling = rank_ic.rolling(
            self.config.rolling_window,
            min_periods=min(10, self.config.rolling_window),
        ).mean().dropna()
        rolling_mean = float(rolling.mean()) if len(rolling) else ric
        rolling_std = float(rolling.std(ddof=1)) if len(rolling) > 1 else 0.0
        sign_consistency = float((np.sign(rank_ic) == np.sign(ric)).mean()) if len(rank_ic) else 0.0
        temporal_stability = sign_consistency / (1.0 + rolling_std)
        sharpe = self._top_decile_sharpe(work)
        robustness = self._perturbation_robustness(work, ric, name)
        rank_stability, rank_turnover = self._rank_stability(work)
        complexity = float(metadata.get("complexity", 1.0))
        depth = float(metadata.get("depth", 1.0))
        logic_score = math.exp(-0.08 * max(0.0, complexity - 5.0) - 0.20 * max(0.0, depth - 3.0))
        return {
            "factor": name,
            "IC": ic,
            "RankIC": ric,
            "ICIR": float(icir),
            "Sharpe": sharpe,
            "complexity": complexity,
            "depth": depth,
            "rolling_ic_mean": rolling_mean,
            "rolling_ic_std": rolling_std,
            "temporal_stability": temporal_stability,
            "robustness": robustness,
            "logic_score": logic_score,
            "rank_stability": rank_stability,
            "rank_turnover": rank_turnover,
        }

    def _top_decile_sharpe(self, work: pd.DataFrame) -> float:
        def daily(group: pd.DataFrame) -> float:
            cutoff = group["factor"].quantile(1.0 - self.config.top_quantile)
            return float(group.loc[group["factor"] >= cutoff, "target"].mean() - group["target"].mean())

        returns = work.dropna().groupby("date", observed=True)[["factor", "target"]].apply(daily).dropna()
        std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        return float(returns.mean() / std * math.sqrt(252 / (self.config.horizon - 1))) if std > 0 else 0.0

    def _perturbation_robustness(self, work: pd.DataFrame, baseline: float, factor_name: str) -> float:
        seed = self.config.seed + sum(map(ord, factor_name))
        rng = np.random.default_rng(seed)
        perturbed = work.copy()
        scale = perturbed.groupby("date", observed=True)["factor"].transform("std").fillna(0.0)
        perturbed["factor"] += rng.normal(0.0, self.config.perturbation_std, len(perturbed)) * scale
        value = _daily_corr(perturbed, "spearman", self.config.min_cross_section).mean()
        if pd.isna(value):
            return 0.0
        denominator = max(abs(baseline), 0.01)
        return float(np.clip(1.0 - abs(float(value) - baseline) / denominator, 0.0, 1.0))

    @staticmethod
    def _rank_stability(work: pd.DataFrame) -> tuple[float, float]:
        ranked = work.assign(rank=work.groupby("date", observed=True)["factor"].rank(pct=True))
        pivot = ranked.pivot(index="date", columns="code", values="rank").sort_index()
        correlations, changes = [], []
        for previous, current in zip(pivot.iloc[:-1].to_numpy(), pivot.iloc[1:].to_numpy()):
            valid = np.isfinite(previous) & np.isfinite(current)
            if valid.sum() >= 3:
                correlations.append(pd.Series(previous[valid]).corr(pd.Series(current[valid]), method="spearman"))
                changes.append(float(np.abs(current[valid] - previous[valid]).mean()))
        stability = float(np.nanmean(correlations)) if correlations else 0.0
        turnover = float(np.nanmean(changes)) if changes else 1.0
        return stability, turnover

    def _dpp_kernel(self, result: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        values = self.data[self.factor_names].copy()
        grouped = values.groupby(self.data["date"], observed=True)
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0.0, np.nan)
        values = ((values - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        matrix = values.to_numpy(float)
        norms = np.linalg.norm(matrix, axis=0)
        matrix = matrix / np.where(norms > EPS, norms, 1.0)
        similarity = np.clip(matrix.T @ matrix, -1.0, 1.0)
        similarity = similarity * similarity
        quality_map = result.set_index("factor")["base_score"]
        quality = np.exp(np.clip(quality_map.reindex(self.factor_names).to_numpy(float), -4.0, 4.0))
        kernel = quality[:, None] * similarity * quality[None, :]
        kernel += np.eye(len(quality)) * 1e-8
        correlation = self.data[self.factor_names].corr(method="spearman").fillna(0.0)
        return kernel, correlation

    @staticmethod
    def _diversity_score(name: str, selected: set[str], correlation: pd.DataFrame) -> float:
        peers = [factor for factor in selected if factor != name and factor in correlation.columns]
        if not peers or name not in correlation:
            return 1.0
        return float(1.0 - correlation.loc[name, peers].abs().mean())

    @staticmethod
    def _zscore(values: Iterable[float]) -> pd.Series:
        series = pd.Series(values, dtype=float)
        std = series.std(ddof=0)
        return (series - series.mean()) / std if std > EPS else pd.Series(0.0, index=series.index)
