"""P1 signal diagnostics aligned with the actual Top-N index strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .universe import normalize_order_book_id


def evaluate_prediction_quality(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    index_key: str,
    top_n: int,
    quantiles: int = 5,
    target_column: str = "target_excess_return",
) -> dict[str, pd.DataFrame]:
    """Evaluate mature labels, tail monotonicity and strategy-aligned Top-N quality."""
    scores = predictions.copy()
    if "signal_date" in scores:
        scores = scores.rename(columns={"signal_date": "date"})
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize()
    scores["code"] = scores["code"].map(normalize_order_book_id)
    target = labels[["date", "code", target_column]].copy()
    target["date"] = pd.to_datetime(target["date"]).dt.normalize()
    target["code"] = target["code"].map(normalize_order_book_id)
    merged = scores[["date", "code", "prediction_score"]].merge(
        target, on=["date", "code"], how="left", validate="one_to_one"
    )
    merged["mature_label"] = merged[target_column].notna()
    valid = merged[merged["mature_label"]].copy()
    valid["quantile"] = valid.groupby("date", observed=True)["prediction_score"].transform(
        lambda values: pd.qcut(values.rank(method="first"), quantiles, labels=False) + 1
    )
    quantile_daily = valid.groupby(["date", "quantile"], observed=True, as_index=False).agg(
        mean_target=(target_column, "mean"), observations=(target_column, "size")
    )
    quantile_wide = quantile_daily.pivot(index="date", columns="quantile", values="mean_target")

    rows = []
    for date, group in valid.groupby("date", observed=True, sort=True):
        ordered = group.nlargest(min(top_n, len(group)), "prediction_score")
        q_returns = quantile_wide.loc[date].dropna()
        rows.append({
            "index_key": index_key,
            "date": date,
            "prediction_count": int((merged["date"] == date).sum()),
            "mature_label_count": int(len(group)),
            "label_coverage": float(len(group) / (merged["date"] == date).sum()),
            "rank_ic": float(group["prediction_score"].corr(group[target_column], method="spearman")),
            "pearson_ic": float(group["prediction_score"].corr(group[target_column])),
            "top_n_mean_target": float(ordered[target_column].mean()),
            "top_n_positive_ratio": float(ordered[target_column].gt(0).mean()),
            "q_high_low": float(q_returns.iloc[-1] - q_returns.iloc[0]),
            "quantile_monotonicity": float(
                pd.Series(q_returns.index.astype(float)).corr(
                    pd.Series(q_returns.to_numpy()), method="spearman"
                )
            ),
        })
    daily = pd.DataFrame(rows)
    daily["year"] = daily["date"].dt.year
    annual_rows = []
    for year, group in daily.groupby("year", observed=True, sort=True):
        ic_std = group["rank_ic"].std(ddof=1)
        annual_rows.append({
            "index_key": index_key,
            "year": int(year),
            "mature_dates": int(len(group)),
            "rank_ic": float(group["rank_ic"].mean()),
            "rank_ic_ir": float(group["rank_ic"].mean() / ic_std * np.sqrt(252))
            if ic_std > 0 else np.nan,
            "positive_ic_ratio": float(group["rank_ic"].gt(0).mean()),
            "q_high_low": float(group["q_high_low"].mean()),
            "positive_q_high_low_ratio": float(group["q_high_low"].gt(0).mean()),
            "quantile_monotonicity": float(group["quantile_monotonicity"].mean()),
            "top_n_mean_target": float(group["top_n_mean_target"].mean()),
            "label_coverage": float(group["mature_label_count"].sum() / group["prediction_count"].sum()),
        })
    annual = pd.DataFrame(annual_rows)

    ranks = scores.pivot(index="date", columns="code", values="prediction_score").rank(
        axis=1, pct=True
    )
    stability_rows = []
    for lag in (1, 5, 20):
        correlations = [
            ranks.iloc[position].corr(ranks.iloc[position - lag], method="spearman")
            for position in range(lag, len(ranks))
        ]
        finite = [value for value in correlations if np.isfinite(value)]
        stability_rows.append({
            "index_key": index_key,
            "lag": lag,
            "rank_autocorrelation": float(np.mean(finite)) if finite else np.nan,
        })
    return {
        "daily_signal_quality": daily,
        "annual_signal_quality": annual,
        "quantile_daily": quantile_daily.assign(index_key=index_key),
        "rank_stability": pd.DataFrame(stability_rows),
    }


def save_prediction_quality(result: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        frame.to_csv(output / f"{name}.csv", index=False)
