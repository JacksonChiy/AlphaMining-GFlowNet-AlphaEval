from __future__ import annotations

import math
import time
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
    dpp_max_rows: int = 500_000
    verbose: bool = True
    seed: int = 42


def _grouped_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return average ranks within integer groups without Python group callbacks."""
    frame = pd.DataFrame({"group": groups, "value": values})
    return frame.groupby("group", observed=True, sort=False)["value"].rank(method="average").to_numpy()


def _daily_corr_arrays(
    date_codes: np.ndarray,
    dates: pd.Index,
    factor: np.ndarray,
    target: np.ndarray,
    method: str,
    min_cross_section: int,
    target_ranks: np.ndarray | None = None,
) -> pd.Series:
    """Vectorized daily Pearson/Spearman correlation using ``bincount`` reductions."""
    valid = np.isfinite(factor) & np.isfinite(target) & (date_codes >= 0)
    if not valid.any():
        return pd.Series(dtype=float)
    groups = date_codes[valid]
    x = factor[valid].astype(np.float64, copy=False)
    y = target[valid].astype(np.float64, copy=False)
    if method == "spearman":
        x = _grouped_rank(x, groups)
        use_cached_target = False
        if target_ranks is not None:
            valid_count = np.bincount(groups, minlength=len(dates))
            target_groups = date_codes[np.isfinite(target) & (date_codes >= 0)]
            target_count = np.bincount(target_groups, minlength=len(dates))
            active = valid_count > 0
            use_cached_target = bool(np.all(valid_count[active] == target_count[active]))
        y = target_ranks[valid] if use_cached_target else _grouped_rank(y, groups)
    elif method != "pearson":
        raise ValueError(f"Unsupported correlation method: {method}")

    size = len(dates)
    count = np.bincount(groups, minlength=size).astype(np.float64)
    sum_x = np.bincount(groups, weights=x, minlength=size)
    sum_y = np.bincount(groups, weights=y, minlength=size)
    sum_x2 = np.bincount(groups, weights=x * x, minlength=size)
    sum_y2 = np.bincount(groups, weights=y * y, minlength=size)
    sum_xy = np.bincount(groups, weights=x * y, minlength=size)
    numerator = count * sum_xy - sum_x * sum_y
    denominator = np.sqrt(
        np.clip(count * sum_x2 - sum_x * sum_x, 0.0, None)
        * np.clip(count * sum_y2 - sum_y * sum_y, 0.0, None)
    )
    usable = (count >= min_cross_section) & (denominator > EPS)
    correlations = np.full(size, np.nan, dtype=np.float64)
    correlations[usable] = numerator[usable] / denominator[usable]
    return pd.Series(correlations, index=dates).dropna()


def _daily_corr(work: pd.DataFrame, method: str, min_cross_section: int) -> pd.Series:
    """Compatibility wrapper used by tests and external callers."""
    codes, dates = pd.factorize(work["date"], sort=True)
    return _daily_corr_arrays(
        codes,
        pd.Index(dates),
        work["factor"].to_numpy(float),
        work["target"].to_numpy(float),
        method,
        min_cross_section,
    )


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
        self.data = self.data.sort_values(keys).reset_index(drop=True)
        self.factor_names = [column for column in factors.columns if column not in keys]
        if not self.factor_names:
            raise ValueError("No factor columns found")
        codes, dates = pd.factorize(self.data["date"], sort=True)
        self._date_codes = codes.astype(np.int32, copy=False)
        self._dates = pd.Index(dates)
        self._target = self.data["target"].to_numpy(float)
        target_valid = np.isfinite(self._target)
        self._target_ranks = np.full(len(self._target), np.nan, dtype=np.float64)
        self._target_ranks[target_valid] = _grouped_rank(
            self._target[target_valid], self._date_codes[target_valid]
        )

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[AlphaEval] {message}", flush=True)

    def evaluate(
        self,
        metadata: pd.DataFrame | None = None,
        output_path: str | Path | None = "results/alpha_eval_result.csv",
    ) -> pd.DataFrame:
        metadata_map = {}
        if metadata is not None and "factor" in metadata:
            metadata_map = metadata.set_index("factor").to_dict("index")
        total_start = time.perf_counter()
        total = len(self.factor_names)
        self._log(
            f"开始评价：样本={len(self.data):,}，交易日={len(self._dates):,}，因子={total}"
        )
        rows = []
        for index, name in enumerate(self.factor_names, start=1):
            factor_start = time.perf_counter()
            self._log(f"因子 {index}/{total} {name}：开始")
            rows.append(self._evaluate_one(name, metadata_map.get(name, {}), index, total))
            elapsed = time.perf_counter() - total_start
            factor_elapsed = time.perf_counter() - factor_start
            eta = elapsed / index * (total - index)
            self._log(
                f"因子 {index}/{total} {name}：完成，耗时={factor_elapsed:.1f}s，"
                f"总进度={index / total:.1%}，预计剩余={eta / 60:.1f}min"
            )
        result = pd.DataFrame(rows)
        self._log("计算综合评价分数")
        result["pps_score"] = self._zscore(result["RankIC"].abs()) + self._zscore(result["ICIR"].abs())
        result["rre_score"] = self._zscore(result["rank_stability"]) - self._zscore(result["rank_turnover"])
        result["base_score"] = (
            0.35 * self._zscore(result["pps_score"])
            + 0.20 * self._zscore(result["temporal_stability"])
            + 0.15 * self._zscore(result["robustness"])
            + 0.15 * self._zscore(result["logic_score"])
            + 0.15 * self._zscore(result["rre_score"])
        )

        self._log("构建 DPP 多样性矩阵")
        dpp_start = time.perf_counter()
        kernel, correlation = self._dpp_kernel(result)
        self._log(f"DPP 矩阵完成，耗时={time.perf_counter() - dpp_start:.1f}s")
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
            self._log(f"结果已保存：{output_path}")
        self._log(f"全部完成，总耗时={(time.perf_counter() - total_start) / 60:.1f}min")
        return result

    def _evaluate_one(
        self,
        name: str,
        metadata: dict[str, object],
        index: int = 1,
        total: int = 1,
    ) -> dict[str, float | str]:
        work = self.data[["date", "code", "target", name]].rename(columns={name: "factor"})
        factor = work["factor"].to_numpy(float)
        stage_start = time.perf_counter()
        pearson = _daily_corr_arrays(
            self._date_codes, self._dates, factor, self._target,
            "pearson", self.config.min_cross_section,
        )
        rank_ic = _daily_corr_arrays(
            self._date_codes, self._dates, factor, self._target,
            "spearman", self.config.min_cross_section, self._target_ranks,
        )
        self._log(
            f"因子 {index}/{total} {name} [1/4] IC/RankIC 完成 "
            f"({time.perf_counter() - stage_start:.1f}s)"
        )
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
        stage_start = time.perf_counter()
        sharpe = self._top_decile_sharpe(work)
        self._log(
            f"因子 {index}/{total} {name} [2/4] Top组合 Sharpe 完成 "
            f"({time.perf_counter() - stage_start:.1f}s)"
        )
        stage_start = time.perf_counter()
        robustness = self._perturbation_robustness(work, ric, name)
        self._log(
            f"因子 {index}/{total} {name} [3/4] 扰动鲁棒性完成 "
            f"({time.perf_counter() - stage_start:.1f}s)"
        )
        stage_start = time.perf_counter()
        rank_stability, rank_turnover = self._rank_stability(work)
        self._log(
            f"因子 {index}/{total} {name} [4/4] 排名稳定性完成 "
            f"({time.perf_counter() - stage_start:.1f}s)"
        )
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
        valid = work["factor"].notna() & work["target"].notna()
        sample = work.loc[valid, ["date", "factor", "target"]]
        cutoff = sample.groupby("date", observed=True, sort=False)["factor"].transform(
            "quantile", q=1.0 - self.config.top_quantile
        )
        top = sample.loc[sample["factor"] >= cutoff]
        market_returns = sample.groupby("date", observed=True, sort=False)["target"].mean()
        top_returns = top.groupby("date", observed=True, sort=False)["target"].mean()
        returns = top_returns.sub(market_returns, fill_value=np.nan).dropna()
        std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        return float(returns.mean() / std * math.sqrt(252 / (self.config.horizon - 1))) if std > 0 else 0.0

    def _perturbation_robustness(self, work: pd.DataFrame, baseline: float, factor_name: str) -> float:
        seed = self.config.seed + sum(map(ord, factor_name))
        rng = np.random.default_rng(seed)
        factor = work["factor"].to_numpy(float)
        scale = work.groupby("date", observed=True, sort=False)["factor"].transform("std").fillna(0.0).to_numpy()
        perturbed = factor + rng.normal(0.0, self.config.perturbation_std, len(work)) * scale
        value = _daily_corr_arrays(
            self._date_codes, self._dates, perturbed, self._target,
            "spearman", self.config.min_cross_section, self._target_ranks,
        ).mean()
        if pd.isna(value):
            return 0.0
        denominator = max(abs(baseline), 0.01)
        return float(np.clip(1.0 - abs(float(value) - baseline) / denominator, 0.0, 1.0))

    @staticmethod
    def _rank_stability(work: pd.DataFrame) -> tuple[float, float]:
        ranked = work[["date", "code"]].copy()
        ranked["rank"] = work.groupby("date", observed=True, sort=False)["factor"].rank(pct=True)
        matrix = ranked.pivot(index="date", columns="code", values="rank").sort_index().to_numpy(np.float32)
        if len(matrix) < 2:
            return 0.0, 1.0
        previous, current = matrix[:-1], matrix[1:]
        valid = np.isfinite(previous) & np.isfinite(current)
        count = valid.sum(axis=1)
        previous_zero = np.where(valid, previous, 0.0).astype(np.float64, copy=False)
        current_zero = np.where(valid, current, 0.0).astype(np.float64, copy=False)
        sum_x = previous_zero.sum(axis=1)
        sum_y = current_zero.sum(axis=1)
        numerator = count * (previous_zero * current_zero).sum(axis=1) - sum_x * sum_y
        denominator = np.sqrt(
            np.clip(count * (previous_zero * previous_zero).sum(axis=1) - sum_x * sum_x, 0.0, None)
            * np.clip(count * (current_zero * current_zero).sum(axis=1) - sum_y * sum_y, 0.0, None)
        )
        usable = (count >= 3) & (denominator > EPS)
        correlations = np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan), where=usable
        )
        changes = np.divide(
            np.where(valid, np.abs(current - previous), 0.0).sum(axis=1),
            count,
            out=np.full(count.shape, np.nan, dtype=float),
            where=count >= 3,
        )
        stability = float(np.nanmean(correlations)) if usable.any() else 0.0
        turnover = float(np.nanmean(changes)) if np.isfinite(changes).any() else 1.0
        return stability, turnover

    def _dpp_kernel(self, result: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        data = self.data
        max_rows = max(1, int(self.config.dpp_max_rows))
        if len(data) > max_rows:
            date_counts = data.groupby("date", observed=True, sort=True).size()
            target_dates = max(1, int(len(date_counts) * max_rows / len(data)))
            positions = np.linspace(0, len(date_counts) - 1, target_dates, dtype=int)
            selected_dates = date_counts.index[positions]
            data = data[data["date"].isin(selected_dates)]
            self._log(
                f"DPP 按交易日均匀抽样：{len(data):,}/{len(self.data):,} 行，"
                f"{len(selected_dates)}/{len(date_counts)} 个交易日"
            )
        else:
            self._log(f"DPP 使用全量样本：{len(data):,} 行")

        ranked = data[self.factor_names].groupby(
            data["date"], observed=True, sort=False
        ).rank(pct=True).astype(np.float32)
        correlation = ranked.corr(method="pearson").fillna(0.0)
        np.fill_diagonal(correlation.values, 1.0)
        similarity = np.square(np.clip(correlation.to_numpy(float), -1.0, 1.0))
        quality_map = result.set_index("factor")["base_score"]
        quality = np.exp(np.clip(quality_map.reindex(self.factor_names).to_numpy(float), -4.0, 4.0))
        kernel = quality[:, None] * similarity * quality[None, :]
        kernel += np.eye(len(quality)) * 1e-8
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
