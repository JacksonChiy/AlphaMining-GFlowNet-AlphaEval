from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.gflownet.reward import make_forward_return


@dataclass
class LightGBMConfig:
    horizon: int = 5
    train_window_days: int = 756
    min_train_days: int = 504
    refit_interval_days: int = 20
    num_leaves: int = 31
    learning_rate: float = 0.03
    n_estimators: int = 500
    seed: int = 42


class LightGBMFusion:
    """Purged rolling-window LightGBM for future five-day cross-sectional returns."""

    def __init__(self, config: LightGBMConfig | None = None) -> None:
        self.config = config or LightGBMConfig()
        self.models: list[object] = []
        self.metrics: list[dict[str, float | str]] = []
        self.feature_names: list[str] = []

    def fit_predict(
        self,
        price: pd.DataFrame,
        factors: pd.DataFrame,
        selected_factors: list[str] | None = None,
        output_dir: str | Path = "results/lightgbm",
    ) -> pd.DataFrame:
        try:
            import lightgbm as lgb
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "LightGBM could not be loaded. Install requirements.txt; on macOS also install "
                "the OpenMP runtime (for example, `brew install libomp`)."
            ) from exc

        keys = ["date", "code"]
        all_factors = [column for column in factors.columns if column not in keys]
        self.feature_names = selected_factors or all_factors
        missing = sorted(set(self.feature_names).difference(all_factors))
        if missing:
            raise ValueError(f"Selected factors missing from matrix: {missing}")
        base = price[keys + ["close"]].copy()
        base["target"] = make_forward_return(price, self.config.horizon).to_numpy()
        data = base.merge(
            factors[keys + self.feature_names], on=keys, how="inner", validate="one_to_one"
        ).sort_values(keys, kind="stable")
        data[self.feature_names] = data.groupby("date", observed=True)[self.feature_names].transform(
            self._cross_sectional_zscore
        )
        dates = np.array(sorted(data["date"].unique()))
        predictions: list[pd.DataFrame] = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start = self.config.min_train_days
        while start < len(dates):
            test_end = min(start + self.config.refit_interval_days, len(dates))
            # Purge `horizon` dates so no training label overlaps the prediction period.
            train_end = start - self.config.horizon
            train_start = max(0, train_end - self.config.train_window_days)
            if train_end - train_start < self.config.min_train_days - self.config.horizon:
                start = test_end
                continue
            train_dates = dates[train_start:train_end]
            test_dates = dates[start:test_end]
            train = data[data["date"].isin(train_dates)].dropna(subset=["target"])
            test = data[data["date"].isin(test_dates)].copy()
            if train.empty or test.empty:
                start = test_end
                continue
            model = lgb.LGBMRegressor(
                objective="regression_l2",
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.config.seed,
                n_jobs=-1,
                verbosity=-1,
            )
            model.fit(train[self.feature_names], train["target"])
            test["prediction_score"] = model.predict(test[self.feature_names])
            predictions.append(test[keys + ["target", "prediction_score"]])
            valid = test.dropna(subset=["target", "prediction_score"])
            daily_ic = valid.groupby("date", observed=True)[["prediction_score", "target"]].apply(
                lambda x: x["prediction_score"].corr(x["target"], method="spearman")
            ).dropna()
            self.metrics.append({
                "train_start": str(pd.Timestamp(train_dates[0]).date()),
                "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "rank_ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
                "train_rows": float(len(train)),
                "test_rows": float(len(test)),
            })
            self.models.append(model)
            start = test_end

        if not predictions:
            raise ValueError(
                "No rolling prediction window was produced. Provide more dates or reduce min_train_days."
            )
        prediction = pd.concat(predictions, ignore_index=True)
        prediction["prediction_rank"] = prediction.groupby("date", observed=True)["prediction_score"].rank(
            pct=True, method="average"
        )
        prediction = prediction.rename(columns={"date": "signal_date"})
        # Future returns remain internal evaluation labels and are never exported
        # to the strategy-facing score file.
        prediction = prediction.drop(columns=["target"])
        prediction.to_csv(output_dir / "prediction_score.csv", index=False)
        pd.DataFrame(self.metrics).to_csv(output_dir / "model_metrics.csv", index=False)
        importance = pd.DataFrame({
            "factor": self.feature_names,
            "importance": self.models[-1].feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(output_dir / "feature_importance.csv", index=False)
        joblib.dump(
            {
                "model": self.models[-1],
                "config": asdict(self.config),
                "features": self.feature_names,
            },
            output_dir / "lgbm_model.joblib",
        )
        return prediction

    @staticmethod
    def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
        std = values.std(ddof=1)
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=values.index)
        return (values - values.mean()) / std

    @staticmethod
    def load(path: str | Path) -> dict[str, object]:
        return joblib.load(Path(path))
