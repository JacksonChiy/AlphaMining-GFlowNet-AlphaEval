"""Stage-0 diagnostics for the frozen index-enhancement baseline.

The module deliberately consumes only persisted local files.  It never calls
RQData or RQAlphaPlus, which makes a diagnostic run reproducible after the
baseline backtests have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from .universe import load_components, normalize_order_book_id


TRADING_DAYS = 252


def file_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> dict:
    """Return an auditable SHA-256 fingerprint without loading a file in memory."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def freeze_baseline_manifest(
    paths: Iterable[str | Path],
    output_path: str | Path,
    metadata: Mapping | None = None,
) -> dict:
    """Persist immutable input identities for one baseline experiment."""
    manifest = {
        "metadata": dict(metadata or {}),
        "files": [file_fingerprint(path) for path in paths],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_price_columns(path: str | Path, include_tradability: bool = False) -> pd.DataFrame:
    """Load only columns needed for labels and tradability diagnostics."""
    header = pd.read_csv(path, nrows=0).columns
    code_column = "code" if "code" in header else "order_book_id"
    candidates = [code_column, "date", "close"]
    if include_tradability:
        candidates.extend([
            "open", "high", "low", "volume", "total_turnover", "amount",
            "limit_up", "limit_down",
        ])
    columns = [column for column in candidates if column in header]
    required = {code_column, "date", "close"}
    if not required.issubset(columns):
        raise ValueError(f"Price file missing columns: {sorted(required.difference(columns))}")
    frame = pd.read_csv(path, usecols=columns)
    return normalize_prices(frame)


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    if "code" not in frame and "order_book_id" in frame:
        frame = frame.rename(columns={"order_book_id": "code"})
    required = {"date", "code", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Price data missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].map(normalize_order_book_id)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


def make_t5_t1_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute close(t+5) / close(t+1) - 1 on each stock's trading-date rows."""
    frame = normalize_prices(prices)
    grouped = frame.groupby("code", observed=True, sort=False)
    entry = grouped["close"].shift(-1)
    exit_ = grouped["close"].shift(-5)
    frame["future_return_t5_t1"] = exit_ / entry - 1.0

    if {"open", "high", "limit_up"}.issubset(frame.columns):
        next_open = grouped["open"].shift(-1)
        next_high = grouped["high"].shift(-1)
        next_limit_up = grouped["limit_up"].shift(-1)
        frame["entry_one_price_limit_up"] = (
            next_open.ge(next_limit_up) & next_high.le(next_limit_up)
        )
    if "volume" in frame:
        frame["entry_suspended"] = grouped["volume"].shift(-1).fillna(0).le(0)
    return frame


def _safe_qcut(values: pd.Series, quantiles: int) -> pd.Series:
    valid = values.notna()
    output = pd.Series(pd.NA, index=values.index, dtype="Int64")
    if valid.sum() < quantiles:
        return output
    ranks = values.loc[valid].rank(method="first")
    output.loc[valid] = pd.qcut(ranks, quantiles, labels=False) + 1
    return output


def evaluate_index_signal(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return daily IC, annual IC summary, and annual quantile returns."""
    scores = predictions.copy()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"]).dt.normalize()
    scores["code"] = scores["code"].map(normalize_order_book_id)
    target = labels[["date", "code", "future_return_t5_t1"]].rename(
        columns={"date": "signal_date"}
    )
    merged = scores.merge(target, on=["signal_date", "code"], how="left", validate="one_to_one")
    index_columns = [column for column in ["index_key", "index_code", "index_name"] if column in merged]
    if not index_columns:
        merged["index_key"] = "unknown"
        index_columns = ["index_key"]

    daily_rows = []
    for keys, group in merged.groupby(index_columns + ["signal_date"], observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        valid = group[["prediction_score", "future_return_t5_t1"]].dropna()
        row = dict(zip(index_columns + ["signal_date"], keys))
        row.update(
            prediction_count=int(len(group)),
            label_count=int(len(valid)),
            label_coverage=float(len(valid) / len(group)) if len(group) else np.nan,
            rank_ic=float(valid.corr(method="spearman").iloc[0, 1]) if len(valid) >= 2 else np.nan,
        )
        daily_rows.append(row)
    daily_ic = pd.DataFrame(daily_rows)
    daily_ic["year"] = pd.to_datetime(daily_ic["signal_date"]).dt.year

    annual_rows = []
    for keys, group in daily_ic.groupby(index_columns + ["year"], observed=True, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group["rank_ic"].dropna()
        mean = values.mean()
        std = values.std(ddof=1)
        row = dict(zip(index_columns + ["year"], keys))
        row.update(
            dates=int(len(group)),
            valid_ic_dates=int(len(values)),
            rank_ic_mean=float(mean) if len(values) else np.nan,
            rank_ic_std=float(std) if len(values) > 1 else np.nan,
            rank_ic_ir=float(mean / std * np.sqrt(TRADING_DAYS))
            if len(values) > 1 and std > 0
            else np.nan,
            positive_ic_ratio=float(values.gt(0).mean()) if len(values) else np.nan,
            label_coverage=float(group["label_count"].sum() / group["prediction_count"].sum()),
        )
        annual_rows.append(row)
    annual_ic = pd.DataFrame(annual_rows)

    merged["year"] = merged["signal_date"].dt.year
    merged["quantile"] = merged.groupby(index_columns + ["signal_date"], observed=True)[
        "prediction_score"
    ].transform(lambda values: _safe_qcut(values, quantiles))
    quantile_return = (
        merged.dropna(subset=["quantile", "future_return_t5_t1"])
        .groupby(index_columns + ["year", "quantile"], observed=True, as_index=False)
        .agg(
            mean_forward_return=("future_return_t5_t1", "mean"),
            observations=("future_return_t5_t1", "size"),
        )
    )
    return daily_ic, annual_ic, quantile_return


def evaluate_universe_coverage(
    predictions: pd.DataFrame, components: pd.DataFrame
) -> pd.DataFrame:
    scores = predictions.copy()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"]).dt.normalize()
    key = str(scores["index_key"].iloc[0])
    expected = components.loc[components["index_key"] == key].groupby("date")["code"].nunique()
    actual = scores.groupby("signal_date")["code"].nunique()
    result = pd.concat([expected.rename("component_count"), actual.rename("prediction_count")], axis=1)
    result = result.loc[result.index.isin(actual.index)].fillna({"prediction_count": 0})
    result["index_key"] = key
    result["prediction_coverage"] = result["prediction_count"] / result["component_count"]
    return result.rename_axis("signal_date").reset_index()


def evaluate_backtest(report_dir: str | Path, index_key: str) -> dict:
    """Recompute comparable return/cost fields from raw RQAlphaPlus exports."""
    root = Path(report_dir)
    portfolio = pd.read_csv(root / "portfolio.csv", parse_dates=["date"])
    trades = pd.read_csv(root / "trades.csv")
    net = pd.to_numeric(portfolio["unit_net_value"], errors="coerce")
    benchmark = pd.to_numeric(portfolio["benchmark_unit_net_value"], errors="coerce")
    daily = net.pct_change().dropna()
    benchmark_daily = benchmark.pct_change().dropna()
    active = daily - benchmark_daily
    years = max((portfolio["date"].iloc[-1] - portfolio["date"].iloc[0]).days / 365.25, 1 / TRADING_DAYS)
    drawdown = net / net.cummax() - 1
    excess_nav = net / benchmark
    excess_drawdown = excess_nav / excess_nav.cummax() - 1
    total_cost = float(pd.to_numeric(trades.get("transaction_cost", 0), errors="coerce").sum())
    notional = float(
        (pd.to_numeric(trades["last_quantity"], errors="coerce") *
         pd.to_numeric(trades["last_price"], errors="coerce")).sum()
    )
    return {
        "index_key": index_key,
        "start_date": str(portfolio["date"].iloc[0].date()),
        "end_date": str(portfolio["date"].iloc[-1].date()),
        "total_return": float(net.iloc[-1] - 1),
        "annual_return": float(net.iloc[-1] ** (1 / years) - 1),
        "benchmark_return": float(benchmark.iloc[-1] - 1),
        "geometric_excess_return": float(net.iloc[-1] / benchmark.iloc[-1] - 1),
        "volatility": float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "tracking_error": float(active.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "information_ratio": float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if active.std(ddof=1) > 0 else np.nan,
        "max_drawdown": float(-drawdown.min()),
        "max_excess_drawdown": float(-excess_drawdown.min()),
        "average_exposure": float((portfolio["market_value"] / portfolio["total_value"]).mean()),
        "initial_cash_days": int(portfolio["market_value"].eq(0).cumprod().sum()),
        "trade_count": int(len(trades)),
        "trade_days": int(pd.to_datetime(trades["datetime"]).dt.normalize().nunique()),
        "traded_notional": notional,
        "transaction_cost": total_cost,
        "cost_over_initial_capital": float(total_cost / portfolio["total_value"].iloc[0]),
    }


def evaluate_backtest_periods(
    report_dir: str | Path, index_key: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return year/month performance, turnover-cost and cash-drag tables."""
    root = Path(report_dir)
    portfolio = pd.read_csv(root / "portfolio.csv", parse_dates=["date"]).sort_values("date")
    trades = pd.read_csv(root / "trades.csv")
    net_return = portfolio["unit_net_value"].pct_change()
    benchmark_return = portfolio["benchmark_unit_net_value"].pct_change()
    returns = pd.DataFrame({
        "date": portfolio["date"],
        "strategy_return": net_return,
        "benchmark_return": benchmark_return,
    })
    returns["active_return"] = returns["strategy_return"] - returns["benchmark_return"]

    def period_table(period: pd.Series, name: str) -> pd.DataFrame:
        rows = []
        for key, group in returns.groupby(period, observed=True, sort=True):
            valid = group.dropna(subset=["strategy_return", "benchmark_return"])
            strategy_total = (1 + valid["strategy_return"]).prod() - 1
            benchmark_total = (1 + valid["benchmark_return"]).prod() - 1
            active_std = valid["active_return"].std(ddof=1)
            wealth = (1 + valid["strategy_return"]).cumprod()
            excess_wealth = (1 + valid["strategy_return"]).cumprod() / (
                1 + valid["benchmark_return"]
            ).cumprod()
            rows.append({
                "index_key": index_key,
                name: str(key),
                "trading_days": int(len(valid)),
                "strategy_return": float(strategy_total),
                "benchmark_return": float(benchmark_total),
                "geometric_excess_return": float((1 + strategy_total) / (1 + benchmark_total) - 1),
                "tracking_error": float(active_std * np.sqrt(TRADING_DAYS)) if len(valid) > 1 else np.nan,
                "information_ratio": float(valid["active_return"].mean() / active_std * np.sqrt(TRADING_DAYS))
                if len(valid) > 1 and active_std > 0 else np.nan,
                "max_drawdown": float(-(wealth / wealth.cummax() - 1).min()) if len(valid) else np.nan,
                "max_excess_drawdown": float(-(excess_wealth / excess_wealth.cummax() - 1).min())
                if len(valid) else np.nan,
            })
        return pd.DataFrame(rows)

    annual = period_table(returns["date"].dt.year, "year")
    monthly = period_table(returns["date"].dt.to_period("M"), "month")

    trade_time = pd.to_datetime(trades["datetime"])
    quantity = pd.to_numeric(trades["last_quantity"], errors="coerce")
    price = pd.to_numeric(trades["last_price"], errors="coerce")
    costs = pd.to_numeric(trades.get("transaction_cost", 0), errors="coerce")
    notional = quantity * price
    elapsed_years = max(
        (portfolio["date"].iloc[-1] - portfolio["date"].iloc[0]).days / 365.25,
        1 / TRADING_DAYS,
    )
    turnover_cost = pd.DataFrame([{
        "index_key": index_key,
        "trade_count": int(len(trades)),
        "trade_days": int(trade_time.dt.normalize().nunique()),
        "traded_notional": float(notional.sum()),
        "transaction_cost": float(costs.sum()),
        "commission": float(pd.to_numeric(trades.get("commission", 0), errors="coerce").sum()),
        "tax": float(pd.to_numeric(trades.get("tax", 0), errors="coerce").sum()),
        "cost_bps_of_notional": float(costs.sum() / notional.sum() * 10_000),
        # Gross traded notional divided by average NAV.  This is intentionally
        # named explicitly rather than conflated with RQAlpha's turnover field.
        "annualized_gross_notional_turnover": float(
            notional.sum() / portfolio["total_value"].mean() / elapsed_years
        ),
    }])

    cash_weight = portfolio["cash"] / portfolio["total_value"]
    zero_exposure = portfolio["market_value"].eq(0)
    # An attribution diagnostic, not a counterfactual backtest: benchmark move
    # while the strategy held no stocks quantifies the initial deployment gap.
    initial_mask = zero_exposure.cumprod().astype(bool)
    initial_benchmark_return = (1 + benchmark_return.loc[initial_mask].dropna()).prod() - 1
    cash_drag = pd.DataFrame([{
        "index_key": index_key,
        "initial_cash_days": int(initial_mask.sum()),
        "first_invested_date": str(portfolio.loc[~zero_exposure, "date"].min().date())
        if (~zero_exposure).any() else None,
        "average_cash_weight": float(cash_weight.mean()),
        "median_cash_weight": float(cash_weight.median()),
        "initial_period_benchmark_return": float(initial_benchmark_return),
    }])
    return annual, monthly, turnover_cost, cash_drag


def run_diagnostics(
    price_path: str | Path,
    component_path: str | Path,
    prediction_paths: Mapping[str, str | Path],
    backtest_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    extra_frozen_paths: Iterable[str | Path] = (),
) -> dict[str, Path]:
    """Run all stage-0 diagnostics and save stable CSV contracts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen_paths: list[Path] = [Path(price_path), Path(component_path)]
    frozen_paths.extend(Path(path) for path in extra_frozen_paths)
    frozen_paths.extend(Path(path) for path in prediction_paths.values())
    for report_dir in backtest_dirs.values():
        report_root = Path(report_dir)
        frozen_paths.extend(
            report_root / name
            for name in [
                "backtest_effective_config.json", "portfolio.csv", "trades.csv"
            ]
            if (report_root / name).exists()
        )
    manifest_path = output / "baseline_manifest.json"
    freeze_baseline_manifest(
        frozen_paths,
        manifest_path,
        metadata={
            "label": "close(t+5) / close(t+1) - 1",
            "indexes": list(prediction_paths),
        },
    )
    labels = make_t5_t1_labels(load_price_columns(price_path))
    components = load_components(component_path)
    daily_frames, annual_frames, quantile_frames, coverage_frames = [], [], [], []
    baseline_rows = []
    annual_performance, monthly_performance = [], []
    turnover_cost_rows, cash_drag_rows = [], []
    for index_key, path in prediction_paths.items():
        print(f"[Diagnostics] signal index={index_key}", flush=True)
        predictions = pd.read_csv(path)
        daily, annual, quantile = evaluate_index_signal(predictions, labels)
        daily_frames.append(daily)
        annual_frames.append(annual)
        quantile_frames.append(quantile)
        coverage_frames.append(evaluate_universe_coverage(predictions, components))
        baseline_rows.append(evaluate_backtest(backtest_dirs[index_key], index_key))
        annual_bt, monthly_bt, turnover_bt, cash_bt = evaluate_backtest_periods(
            backtest_dirs[index_key], index_key
        )
        annual_performance.append(annual_bt)
        monthly_performance.append(monthly_bt)
        turnover_cost_rows.append(turnover_bt)
        cash_drag_rows.append(cash_bt)
    outputs = {
        "baseline_manifest": manifest_path,
        "daily_ic": output / "daily_ic.csv",
        "annual_ic": output / "annual_ic.csv",
        "quantile_return": output / "quantile_return.csv",
        "universe_coverage": output / "universe_coverage.csv",
        "baseline_comparison": output / "baseline_comparison.csv",
        "annual_performance": output / "annual_performance.csv",
        "monthly_performance": output / "monthly_performance.csv",
        "turnover_cost": output / "turnover_cost.csv",
        "cash_drag": output / "cash_drag.csv",
    }
    pd.concat(daily_frames, ignore_index=True).to_csv(outputs["daily_ic"], index=False)
    pd.concat(annual_frames, ignore_index=True).to_csv(outputs["annual_ic"], index=False)
    pd.concat(quantile_frames, ignore_index=True).to_csv(outputs["quantile_return"], index=False)
    pd.concat(coverage_frames, ignore_index=True).to_csv(outputs["universe_coverage"], index=False)
    pd.DataFrame(baseline_rows).to_csv(outputs["baseline_comparison"], index=False)
    pd.concat(annual_performance, ignore_index=True).to_csv(outputs["annual_performance"], index=False)
    pd.concat(monthly_performance, ignore_index=True).to_csv(outputs["monthly_performance"], index=False)
    pd.concat(turnover_cost_rows, ignore_index=True).to_csv(outputs["turnover_cost"], index=False)
    pd.concat(cash_drag_rows, ignore_index=True).to_csv(outputs["cash_drag"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结并诊断三指数增强基线")
    parser.add_argument("--config", default="configs/index_enhancement/default.yaml")
    parser.add_argument("--price", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    prediction_root = Path(config.get("build", {}).get("output_root", "results/index_enhancement"))
    backtest_root = Path(config.get("backtest", {}).get("output_root", "results/index_enhancement_backtest"))
    diagnostic_config = config.get("diagnostics", {})
    price_path = args.price or config.get("labels", {}).get("price_file", "data/price.csv")
    output_dir = args.output_dir or diagnostic_config.get(
        "output_root", "results/index_enhancement_diagnostics"
    )
    index_keys = list(config["indexes"])
    prediction_paths = {}
    for key in index_keys:
        candidates = [
            prediction_root / key / "prediction_score.csv",
            prediction_root / key / "prediction_score.csv.gz",
        ]
        prediction_paths[key] = next(
            (path for path in candidates if path.exists()), candidates[0]
        )
    outputs = run_diagnostics(
        price_path,
        config.get("component_data", {}).get("file", "data/index_components.csv.gz"),
        prediction_paths,
        {key: backtest_root / key for key in index_keys},
        output_dir,
        extra_frozen_paths=[args.config],
    )
    print(f"[Diagnostics] complete files={len(outputs)} output={Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
