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
    prediction_start_date: str | None = None
    prediction_end_date: str | None = None
    label_path: str | None = None
    target_type: str = "raw_return"
    objective: str = "regression_l2"
    rank_bins: int = 20
    top_weight_quantile: float = 0.20
    top_weight_multiplier: float = 1.0
    save_all_models: bool = True


class LightGBMFusion:
    """Purged rolling LightGBM for close(t+5) / close(t+1) - 1 labels."""

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
        base = self._build_training_base(price)
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

        prediction_start_index, prediction_end_index = self._prediction_indices(
            dates,
            self.config.min_train_days,
            self.config.prediction_start_date,
            self.config.prediction_end_date,
        )
        print(
            "[LightGBM] rolling_setup "
            f"merged_rows={len(data):,} dates={len(dates)} factors={len(self.feature_names)} "
            f"min_train_days={self.config.min_train_days} "
            f"prediction_start={self.config.prediction_start_date} "
            f"prediction_end={self.config.prediction_end_date}",
            flush=True,
        )

        start = prediction_start_index
        window_index = 0
        while start < prediction_end_index:
            test_end = min(
                start + self.config.refit_interval_days, prediction_end_index
            )
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
                print(
                    f"[LightGBM] window_skipped start_index={start} "
                    f"train_rows={len(train)} test_rows={len(test)}",
                    flush=True,
                )
                start = test_end
                continue
            window_index += 1
            print(
                f"[LightGBM] window_start index={window_index:03d} "
                f"train={pd.Timestamp(train_dates[0]).date()}.."
                f"{pd.Timestamp(train_dates[-1]).date()} rows={len(train):,} "
                f"test={pd.Timestamp(test_dates[0]).date()}.."
                f"{pd.Timestamp(test_dates[-1]).date()} rows={len(test):,}",
                flush=True,
            )
            model_class = lgb.LGBMRanker if self.config.objective == "lambdarank" else lgb.LGBMRegressor
            model = model_class(
                objective=self.config.objective,
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
            sample_weight = self._sample_weight(train)
            fit_kwargs: dict[str, object] = {}
            if self.config.top_weight_multiplier > 1.0:
                fit_kwargs["sample_weight"] = sample_weight
            train_target = train["target"]
            if self.config.objective == "lambdarank":
                train = train.sort_values(keys, kind="stable")
                sample_weight = self._sample_weight(train)
                fit_kwargs = {"group": train.groupby("date", observed=True, sort=True).size().tolist()}
                if self.config.top_weight_multiplier > 1.0:
                    fit_kwargs["sample_weight"] = sample_weight
                train_target = self._ranking_relevance(train)
            model.fit(train[self.feature_names], train_target, **fit_kwargs)
            test["prediction_score"] = model.predict(test[self.feature_names])
            predictions.append(test[keys + ["target", "prediction_score"]])
            valid = test.dropna(subset=["target", "prediction_score"]).copy()
            daily_ic = valid.groupby("date", observed=True)[["prediction_score", "target"]].apply(
                lambda x: x["prediction_score"].corr(x["target"], method="spearman")
            ).dropna()
            valid["score_quantile"] = valid.groupby("date", observed=True)[
                "prediction_score"
            ].transform(
                lambda values: pd.qcut(values.rank(method="first"), 5, labels=False) + 1
            )
            quantile_return = valid.groupby(["date", "score_quantile"], observed=True)[
                "target"
            ].mean().unstack()
            q_high_low = (
                quantile_return.iloc[:, -1] - quantile_return.iloc[:, 0]
                if quantile_return.shape[1] >= 2 else pd.Series(dtype=float)
            )
            top_cutoff = valid.groupby("date", observed=True)["prediction_score"].transform(
                "quantile", q=1.0 - self.config.top_weight_quantile
            )
            top_target = valid.loc[valid["prediction_score"] >= top_cutoff].groupby(
                "date", observed=True
            )["target"].mean()
            self.metrics.append({
                "train_start": str(pd.Timestamp(train_dates[0]).date()),
                "train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "rank_ic": float(daily_ic.mean()) if len(daily_ic) else np.nan,
                "positive_rank_ic_ratio": float(daily_ic.gt(0).mean()) if len(daily_ic) else np.nan,
                "q5_q1": float(q_high_low.mean()) if len(q_high_low) else np.nan,
                "positive_q5_q1_ratio": float(q_high_low.gt(0).mean()) if len(q_high_low) else np.nan,
                "top_quantile_mean_target": float(top_target.mean()) if len(top_target) else np.nan,
                "mature_label_dates": float(valid["date"].nunique()),
                "last_mature_label_date": str(valid["date"].max().date()) if len(valid) else "",
                "train_rows": float(len(train)),
                "test_rows": float(len(test)),
            })
            self.models.append(model)
            if self.config.save_all_models:
                joblib.dump(
                    {"model": model, "config": asdict(self.config), "features": self.feature_names},
                    output_dir / f"lgbm_window_{window_index:03d}.joblib",
                )
            print(
                f"[LightGBM] window_complete index={window_index:03d} "
                f"rank_ic={self.metrics[-1]['rank_ic']:.6f}",
                flush=True,
            )
            start = test_end

        if not predictions:
            available_start = str(pd.Timestamp(dates[0]).date()) if len(dates) else "N/A"
            available_end = str(pd.Timestamp(dates[-1]).date()) if len(dates) else "N/A"
            raise ValueError(
                "No rolling prediction window was produced. "
                f"merged_dates={len(dates)}, available={available_start}..{available_end}, "
                f"min_train_days={self.config.min_train_days}, "
                f"prediction_start_date={self.config.prediction_start_date}. "
                "The price/factor matrix must include the training history before the prediction start."
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

    def _sample_weight(self, train: pd.DataFrame) -> np.ndarray:
        """Emphasize the profitable tail without changing chronological sampling."""
        weights = np.ones(len(train), dtype=np.float64)
        multiplier = float(self.config.top_weight_multiplier)
        if multiplier <= 1.0:
            return weights
        quantile = float(self.config.top_weight_quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError("top_weight_quantile must be in (0, 1)")
        cutoffs = train.groupby("date", observed=True)["target"].transform(
            "quantile", q=1.0 - quantile
        )
        weights[train["target"].ge(cutoffs).to_numpy()] = multiplier
        return weights

    def _ranking_relevance(self, train: pd.DataFrame) -> pd.Series:
        """Convert continuous returns to integer relevance levels per date."""
        bins = max(2, int(self.config.rank_bins))
        percentile = train.groupby("date", observed=True)["target"].rank(
            method="average", pct=True
        )
        return np.minimum((percentile * bins).astype(int), bins - 1)

    def _build_training_base(self, price: pd.DataFrame) -> pd.DataFrame:
        """Select the legacy full-market label or a local PIT index label file."""
        keys = ["date", "code"]
        if not self.config.label_path:
            base = price[keys + ["close"]].copy()
            base["target"] = make_forward_return(price, self.config.horizon).to_numpy()
            return base[keys + ["target"]]

        label_path = Path(self.config.label_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Index label file not found: {label_path.resolve()}")
        labels = (
            pd.read_pickle(label_path)
            if label_path.suffix.lower() in {".pkl", ".pickle"}
            else pd.read_csv(label_path)
        )
        target_columns = {
            "raw_return": "target_raw_return",
            "excess_return": "target_excess_return",
            "cross_sectional_rank": "target_cross_sectional_rank",
            "rank": "target_cross_sectional_rank",
        }
        if self.config.target_type not in target_columns:
            raise ValueError(
                "target_type must be raw_return, excess_return, or cross_sectional_rank"
            )
        target_column = target_columns[self.config.target_type]
        missing = {*keys, target_column}.difference(labels.columns)
        if missing:
            raise ValueError(f"Index label file missing columns: {sorted(missing)}")
        base = labels[[*keys, target_column]].rename(columns={target_column: "target"})
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        if base.duplicated(keys).any():
            raise ValueError("Index label file contains duplicate date/code rows")
        print(
            f"[LightGBM] label_source={label_path.resolve()} "
            f"target_type={self.config.target_type} rows={len(base):,} "
            f"valid_targets={base['target'].notna().sum():,}",
            flush=True,
        )
        return base

    @staticmethod
    def _cross_sectional_zscore(values: pd.Series) -> pd.Series:
        std = values.std(ddof=1)
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=values.index)
        return (values - values.mean()) / std

    @staticmethod
    def _prediction_indices(
        dates: np.ndarray,
        min_train_days: int,
        prediction_start_date: str | None,
        prediction_end_date: str | None,
    ) -> tuple[int, int]:
        index = pd.DatetimeIndex(dates)
        start = min_train_days
        if prediction_start_date is not None:
            start = max(
                start,
                int(index.searchsorted(pd.Timestamp(prediction_start_date), side="left")),
            )
        end = len(index)
        if prediction_end_date is not None:
            end = int(index.searchsorted(pd.Timestamp(prediction_end_date), side="right"))
        return start, end

    @staticmethod
    def load(path: str | Path) -> dict[str, object]:
        return joblib.load(Path(path))
