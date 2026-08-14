from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from types import SimpleNamespace
import json

from src.model import LightGBMConfig, LightGBMFusion
from src.gflownet import execute_saved_alpha_pool
from src.utils import (
    load_config,
    slice_date_range,
    validate_frame_covers_period,
    validate_research_date_split,
)


def test_slice_date_range_is_inclusive() -> None:
    frame = pd.DataFrame({
        "date": pd.date_range("2021-12-30", periods=5),
        "value": range(5),
    })

    result = slice_date_range(frame, "2021-12-31", "2022-01-02")

    assert result["value"].tolist() == [1, 2, 3]


def test_slice_date_range_rejects_empty_period() -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2022-01-01")]})
    with pytest.raises(ValueError, match="has no rows"):
        slice_date_range(frame, "2023-01-01", label="test data")


def test_training_configs_use_2020_2023_and_2024_2026_split() -> None:
    for path in (
        "configs/daily/quick.yaml",
        "configs/daily/training.yaml",
    ):
        config = load_config(path)
        assert validate_research_date_split(config) == {
            "training": "2020-01-01..2023-12-31",
            "out_of_sample": "2024-01-01..2026-12-31",
        }
        assert config["lightgbm"]["train_window_days"] == 1008
        assert config["lightgbm"]["min_train_days"] == 756


def test_date_split_rejects_training_oos_overlap() -> None:
    config = load_config("configs/daily/quick.yaml")
    config["dataset"]["mining_end_date"] = "2024-01-02"
    with pytest.raises(ValueError, match="must not overlap"):
        validate_research_date_split(config)


def test_training_coverage_rejects_stale_daily_price() -> None:
    stale = pd.DataFrame({"date": pd.bdate_range("2021-01-01", "2023-12-29")})
    with pytest.raises(ValueError, match="rebuild daily_price.pkl"):
        validate_frame_covers_period(
            stale,
            "2020-01-01",
            "2023-12-31",
            label="training data",
        )


def test_lightgbm_prediction_period_preserves_training_history() -> None:
    dates = pd.bdate_range("2021-01-01", "2026-12-31").to_numpy()

    start, end = LightGBMFusion._prediction_indices(
        dates,
        min_train_days=252,
        prediction_start_date="2023-01-01",
        prediction_end_date="2026-12-31",
    )

    assert pd.Timestamp(dates[start]) >= pd.Timestamp("2023-01-01")
    assert pd.Timestamp(dates[start - 1]) < pd.Timestamp("2023-01-01")
    assert pd.Timestamp(dates[end - 1]) <= pd.Timestamp("2026-12-31")
    assert start > 252


def test_lightgbm_uses_local_index_excess_labels(tmp_path) -> None:
    label_path = tmp_path / "labels.pkl"
    labels = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "code": ["000001.XSHE", "000002.XSHE"],
        "target_raw_return": [0.02, 0.01],
        "target_excess_return": [0.015, 0.005],
        "target_cross_sectional_rank": [1.0, 0.5],
    })
    labels.to_pickle(label_path)
    fusion = LightGBMFusion(LightGBMConfig(
        label_path=str(label_path), target_type="excess_return"
    ))

    base = fusion._build_training_base(pd.DataFrame())

    assert base["target"].tolist() == [0.015, 0.005]
    assert base.columns.tolist() == ["date", "code", "target"]


class _DummyRegressor:
    def __init__(self, **kwargs) -> None:
        self.feature_importances_ = np.array([1.0])

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_DummyRegressor":
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features.iloc[:, 0].fillna(0.0).to_numpy()


def test_fit_predict_outputs_only_configured_oos_dates(
    daily_prices: pd.DataFrame, tmp_path, monkeypatch
) -> None:
    monkeypatch.setitem(
        sys.modules, "lightgbm", SimpleNamespace(LGBMRegressor=_DummyRegressor)
    )
    factors = daily_prices[["date", "code"]].copy()
    factors["factor_001"] = daily_prices["close"]
    prediction_start = daily_prices["date"].drop_duplicates().iloc[60]
    fusion = LightGBMFusion(LightGBMConfig(
        horizon=5,
        train_window_days=40,
        min_train_days=20,
        refit_interval_days=10,
        n_estimators=2,
        prediction_start_date=str(prediction_start.date()),
    ))

    prediction = fusion.fit_predict(
        daily_prices, factors, ["factor_001"], output_dir=tmp_path
    )

    assert prediction["signal_date"].min() == prediction_start
    assert prediction["signal_date"].max() == daily_prices["date"].max()
    assert prediction["signal_date"].nunique() == 40
    assert (tmp_path / "prediction_score.csv").exists()
    metrics = pd.read_csv(tmp_path / "model_metrics.csv")
    assert metrics["prediction_varying_dates"].gt(0).all()
    assert metrics["constant_prediction_dates"].eq(0).all()
    assert metrics["active_factors_median"].gt(0).all()


def test_safe_daily_spearman_skips_constant_dates_without_warning() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-02", "2026-01-02", "2026-01-02",
            "2026-01-05", "2026-01-05", "2026-01-05",
        ]),
        "prediction_score": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "target": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
    })

    result = LightGBMFusion._safe_daily_spearman(frame)

    assert result.index.tolist() == [pd.Timestamp("2026-01-02")]
    assert np.isclose(result.iloc[0], 1.0)


def test_window_diagnostics_distinguish_constant_target_and_prediction() -> None:
    fusion = LightGBMFusion()
    fusion.feature_names = ["factor_001"]
    frame = pd.DataFrame({
        "date": pd.to_datetime([
            "2026-01-02", "2026-01-02", "2026-01-02",
            "2026-01-05", "2026-01-05", "2026-01-05",
        ]),
        "factor_001": [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "prediction_score": [1.0, 2.0, 3.0, 1.0, 1.0, 1.0],
        "target": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
    })

    result = fusion._window_diagnostics(frame)

    assert result["rank_ic_evaluable_dates"] == 1
    assert result["constant_target_dates"] == 1
    assert result["constant_prediction_dates"] == 1
    assert result["zero_active_factor_dates"] == 1


def test_cross_sectional_zscore_preserves_missing_values() -> None:
    all_missing = pd.Series([np.nan, np.nan])
    partly_missing_constant = pd.Series([1.0, 1.0, np.nan])

    missing_result = LightGBMFusion._cross_sectional_zscore(all_missing)
    constant_result = LightGBMFusion._cross_sectional_zscore(
        partly_missing_constant
    )

    assert missing_result.isna().all()
    assert constant_result.iloc[:2].eq(0.0).all()
    assert pd.isna(constant_result.iloc[2])


def test_saved_alpha_pool_executes_full_history_before_oos_slice(
    daily_prices: pd.DataFrame, tmp_path
) -> None:
    metadata = pd.DataFrame({
        "factor": ["factor_001"],
        "tokens": [json.dumps(["ts_mean", "W5", "close"])],
    })
    metadata_path = tmp_path / "alpha_pool.csv"
    metadata.to_csv(metadata_path, index=False)
    oos_start = daily_prices["date"].drop_duplicates().iloc[60]

    full, oos = execute_saved_alpha_pool(
        daily_prices,
        metadata_path,
        tmp_path / "full.pkl",
        tmp_path / "oos.pkl",
        str(oos_start.date()),
    )

    assert oos is not None
    assert oos["date"].min() == oos_start
    assert oos.loc[oos["date"] == oos_start, "factor_001"].notna().all()
    assert len(full) == len(daily_prices)
