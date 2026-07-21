from __future__ import annotations

import numpy as np
import pandas as pd

from src.gflownet.reward import RewardEvaluator, make_forward_return
from src.operators import apply_binary, apply_cross_sectional, apply_time_series


def test_safe_division_and_cross_section() -> None:
    left = pd.Series([1.0, 2.0, 3.0])
    right = pd.Series([1.0, 0.0, 3.0])
    divided = apply_binary("div", left, right)
    assert divided.iloc[0] == 1.0 and np.isnan(divided.iloc[1])
    dates = pd.Series(pd.to_datetime(["2024-01-01"] * 3))
    ranked = apply_cross_sectional("cs_rank", left, dates)
    assert ranked.tolist() == [1 / 3, 2 / 3, 1.0]


def test_rolling_window_is_grouped_by_security() -> None:
    values = pd.Series([1.0, 2.0, 10.0, 20.0])
    codes = pd.Series(["A", "A", "B", "B"])
    result = apply_time_series("ts_mean", values, codes, 2)
    assert np.isnan(result.iloc[0]) and result.iloc[1] == 1.5
    assert np.isnan(result.iloc[2]) and result.iloc[3] == 15.0


def test_forward_return_uses_tplus5_over_tplus1() -> None:
    data = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=7),
        "code": ["A"] * 7,
        "close": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    })
    label = make_forward_return(data, horizon=5)
    assert label.iloc[0] == 32.0 / 2.0 - 1.0
    assert label.iloc[1] == 64.0 / 4.0 - 1.0
    assert label.iloc[2:].isna().all()

    changed = data.copy()
    changed.loc[0, "close"] = 999.0
    assert make_forward_return(changed, horizon=5).iloc[0] == label.iloc[0]


def test_vectorized_grouped_correlation_matches_pandas_reference() -> None:
    dates = pd.Series(pd.to_datetime([
        "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01",
        "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02",
    ]))
    left = pd.Series([1.0, 2.0, 2.0, 4.0, 8.0, 6.0, 7.0, 5.0])
    right = pd.Series([4.0, 1.0, 2.0, 3.0, 2.0, 4.0, 3.0, 1.0])
    expected = pd.DataFrame({"date": dates, "left": left, "right": right}).groupby(
        "date", observed=True
    ).apply(lambda group: group["left"].corr(group["right"]))

    actual = RewardEvaluator._grouped_correlation(left, right, dates)

    pd.testing.assert_series_equal(actual, expected, check_names=False, rtol=1e-12, atol=1e-12)


def test_vectorized_risk_exposures_match_reference() -> None:
    dates = pd.to_datetime(["2024-01-01"] * 8 + ["2024-01-02"] * 8)
    work = pd.DataFrame({
        "date": dates,
        "factor": [1.0, 2.0, 4.0, 3.0, 8.0, 7.0, 5.0, 6.0] * 2,
        "industry": ["A", "A", "A", "A", "B", "B", "B", "B"] * 2,
        "market_cap": np.tile(np.arange(1.0, 9.0), 2) * 1e8,
    })
    industry_reference = []
    size_reference = []
    for _, group in work.groupby("date", observed=True):
        y = group["factor"].to_numpy(float)
        dummies = pd.get_dummies(group["industry"], dtype=float).to_numpy()
        fitted = dummies @ np.linalg.lstsq(dummies, y, rcond=None)[0]
        industry_reference.append(
            1.0 - np.square(y - fitted).sum() / np.square(y - y.mean()).sum()
        )
        size_reference.append(abs(group["factor"].corr(
            np.log1p(group["market_cap"]), method="spearman"
        )))

    assert np.isclose(
        RewardEvaluator._industry_exposure(work), np.mean(industry_reference), atol=1e-12
    )
    assert np.isclose(
        RewardEvaluator._size_exposure(work), np.mean(size_reference), atol=1e-12
    )


def test_vectorized_reward_matches_original_rank_ic_and_long_ir(
    daily_prices: pd.DataFrame,
) -> None:
    evaluator = RewardEvaluator(daily_prices, horizon=5, min_cross_section=20)
    factor = daily_prices.groupby("code", observed=True)["close"].pct_change(5)
    actual = evaluator.evaluate_factor(factor)
    work = evaluator.data[["date", "code", "_target"]].copy()
    work["factor"] = factor.reindex(work.index)
    work = work.dropna(subset=["factor", "_target"])
    counts = work.groupby("date", observed=True).size()
    work = work[work["date"].isin(counts[counts >= 20].index)]
    rank_ic = work.groupby("date", observed=True)[["factor", "_target"]].apply(
        lambda group: group["factor"].corr(group["_target"], method="spearman")
    ).dropna()

    def long_excess(group: pd.DataFrame) -> float:
        cutoff = group["factor"].quantile(0.9)
        return float(
            group.loc[group["factor"] >= cutoff, "_target"].mean()
            - group["_target"].mean()
        )

    excess = work.groupby("date", observed=True)[["factor", "_target"]].apply(
        long_excess
    ).dropna()
    expected_long_ir = excess.mean() / excess.std(ddof=1) * np.sqrt(252 / 4)

    assert np.isclose(actual.rank_ic, rank_ic.mean(), atol=1e-12)
    assert np.isclose(actual.long_ir, expected_long_ir, atol=1e-12)


def test_low_coverage_factor_receives_penalty(daily_prices: pd.DataFrame) -> None:
    evaluator = RewardEvaluator(
        daily_prices,
        horizon=5,
        min_cross_section=20,
        min_coverage=0.80,
        coverage_penalty_power=2.0,
    )
    factor = daily_prices["close"].astype(float).copy()
    code_order = daily_prices["code"].drop_duplicates().tolist()
    factor[daily_prices["code"].isin(code_order[20:])] = np.nan

    result = evaluator.evaluate_factor(factor)

    assert np.isclose(result.coverage, 20 / len(code_order))
    assert result.valid_date_coverage == 1.0
    assert np.isclose(result.coverage_penalty, (result.coverage / 0.80) ** 2)
    assert result.coverage < evaluator.min_coverage


def test_missing_whole_dates_is_reflected_in_date_coverage(
    daily_prices: pd.DataFrame,
) -> None:
    evaluator = RewardEvaluator(
        daily_prices, horizon=5, min_cross_section=20, min_coverage=0.80
    )
    factor = daily_prices["close"].astype(float).copy()
    eligible_dates = evaluator.data.loc[
        evaluator.data["_target"].notna(), "date"
    ].drop_duplicates()
    removed_dates = set(eligible_dates.iloc[::2])
    factor[evaluator.data["date"].isin(removed_dates)] = np.nan

    result = evaluator.evaluate_factor(factor)

    expected = 1.0 - len(removed_dates) / len(eligible_dates)
    assert np.isclose(result.valid_date_coverage, expected)
    assert result.coverage_penalty < 1.0
