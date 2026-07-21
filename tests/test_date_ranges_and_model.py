from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from types import SimpleNamespace

from src.model import LightGBMConfig, LightGBMFusion
from src.utils import slice_date_range


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
