from __future__ import annotations

import numpy as np
import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig
from src.index_enhancement.evaluation import evaluate_prediction_quality
from src.model import LightGBMConfig, LightGBMFusion


def test_prediction_quality_detects_monotonic_top_tail() -> None:
    dates = pd.to_datetime(["2024-01-02"] * 10 + ["2024-01-03"] * 10)
    codes = [f"{index:06d}.XSHE" for index in range(10)] * 2
    scores = np.tile(np.arange(10, dtype=float), 2)
    predictions = pd.DataFrame(
        {"signal_date": dates, "code": codes, "prediction_score": scores}
    )
    labels = pd.DataFrame(
        {"date": dates, "code": codes, "target_excess_return": scores / 100}
    )

    result = evaluate_prediction_quality(predictions, labels, "csi300", top_n=2)

    annual = result["annual_signal_quality"].iloc[0]
    assert np.isclose(annual["rank_ic"], 1.0)
    assert annual["q_high_low"] > 0
    assert np.isclose(annual["quantile_monotonicity"], 1.0)


def test_lightgbm_ranking_relevance_and_top_weights() -> None:
    train = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 5 + ["2024-01-03"] * 5),
        "target": list(range(5)) + list(range(5)),
    })
    fusion = LightGBMFusion(LightGBMConfig(
        rank_bins=5, top_weight_quantile=0.2, top_weight_multiplier=3.0
    ))

    relevance = fusion._ranking_relevance(train)
    weights = fusion._sample_weight(train)

    assert relevance.groupby(train["date"]).max().tolist() == [4, 4]
    assert weights.tolist().count(3.0) == 2


def test_alpha_eval_accepts_external_index_target(daily_prices) -> None:
    factor_matrix = daily_prices[["date", "code"]].copy()
    factor_matrix["factor_001"] = daily_prices.groupby("date")["close"].rank(pct=True)
    target = daily_prices[["date", "code"]].copy()
    target["target_excess_return"] = factor_matrix["factor_001"]
    evaluator = AlphaEval(
        daily_prices,
        factor_matrix[["date", "code", "factor_001"]],
        AlphaEvalConfig(dpp_k=1, dpp_max_rows=10_000, verbose=False),
        target_data=target,
        target_column="target_excess_return",
    )

    result = evaluator.evaluate(output_path=None)

    assert result.iloc[0]["RankIC"] > 0.99
