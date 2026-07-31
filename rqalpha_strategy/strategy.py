"""RQAlphaPlus daily Top-N strategy using strictly lagged LightGBM scores."""

from __future__ import annotations

import os
import re
from math import isfinite
from pathlib import Path

import pandas as pd

from src.index_enhancement.portfolio_optimizer import (
    PortfolioOptimizerConfig,
    latest_weight_date,
    load_weight_history_for_strategy,
    optimize_benchmark_portfolio,
)


DEFAULT_SMOOTHING_WEIGHTS = (0.5, 0.3, 0.2)


def _prediction_path() -> Path:
    configured = os.environ.get("ALPHAMINING_PREDICTIONS")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "results/lightgbm/prediction_score.csv").resolve()


def normalize_order_book_id(value: str) -> str:
    value = str(value).strip().upper()
    replacements = {".SH": ".XSHG", ".SZ": ".XSHE", ".BJ": ".XBSE"}
    for source, target in replacements.items():
        if value.endswith(source):
            return value[: -len(source)] + target
    if re.fullmatch(r"\d{6}", value):
        if value.startswith(("5", "6", "9")):
            return f"{value}.XSHG"
        if value.startswith(("4", "8")):
            return f"{value}.XBSE"
        return f"{value}.XSHE"
    return value


def parse_smoothing_weights(value: str | None) -> tuple[float, ...]:
    """Parse newest-to-oldest rank smoothing weights from an environment value."""
    if not value:
        return DEFAULT_SMOOTHING_WEIGHTS
    try:
        weights = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid rank smoothing weights: {value!r}") from exc
    if not weights or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("Rank smoothing weights must contain non-negative values with a positive sum")
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def build_smoothed_scores(
    scores: pd.DataFrame,
    weights: tuple[float, ...] = DEFAULT_SMOOTHING_WEIGHTS,
) -> dict:
    """Return date-indexed score frames with cross-sectional rank smoothing.

    The first weight applies to the current signal date, followed by the previous
    signal dates. Only stocks present in the current cross-section are eligible;
    missing historical ranks are reweighted over the observations that exist.
    """
    if not weights or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative and sum to a positive value")
    normalized_weights = tuple(weight / sum(weights) for weight in weights)
    frames = {
        date: group.groupby("code", as_index=False)["prediction_score"].mean()
        for date, group in scores.groupby("signal_date", sort=True)
    }
    signal_dates = sorted(frames)
    rank_maps = {}
    for date in signal_dates:
        frame = frames[date].copy()
        frame["raw_rank_score"] = frame["prediction_score"].rank(
            method="average", pct=True, ascending=True
        )
        frames[date] = frame
        rank_maps[date] = frame.set_index("code")["raw_rank_score"]

    output = {}
    for date_index, date in enumerate(signal_dates):
        current = frames[date].copy()
        numerator = pd.Series(0.0, index=current.index)
        denominator = pd.Series(0.0, index=current.index)
        for lag, weight in enumerate(normalized_weights):
            historical_index = date_index - lag
            if historical_index < 0 or weight == 0:
                continue
            historical_date = signal_dates[historical_index]
            historical_rank = current["code"].map(rank_maps[historical_date])
            valid = historical_rank.notna()
            numerator.loc[valid] += weight * historical_rank.loc[valid]
            denominator.loc[valid] += weight
        current["smoothed_rank_score"] = numerator / denominator.where(denominator > 0)
        current = current.sort_values(
            ["smoothed_rank_score", "prediction_score", "code"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        current["smoothed_rank"] = range(1, len(current) + 1)
        output[date] = current
    return output


def select_turnover_controlled_targets(
    ranked_codes: list[str],
    current_holdings: list[str],
    holding_days: dict[str, int],
    top_n: int,
    hold_buffer_rank: int,
    max_replacement_ratio: float,
    min_holding_days: int,
) -> list[str]:
    """Select targets with a rank buffer, minimum hold and replacement cap."""
    ranked_codes = list(dict.fromkeys(ranked_codes))
    current_holdings = list(dict.fromkeys(current_holdings))
    rank = {code: index + 1 for index, code in enumerate(ranked_codes)}
    max_replacements = max(1, int(top_n * max_replacement_ratio))
    max_replacements = min(top_n, max_replacements)

    # Young holdings are protected first. Existing names inside the wider rank
    # buffer are then preferred over new entrants near the Top-N boundary.
    protected = sorted(
        (code for code in current_holdings if holding_days.get(code, 0) < min_holding_days),
        key=lambda code: rank.get(code, float("inf")),
    )
    buffered = sorted(
        (
            code
            for code in current_holdings
            if code not in protected and rank.get(code, float("inf")) <= hold_buffer_rank
        ),
        key=lambda code: rank[code],
    )
    retained = (protected + buffered)[:top_n]

    # Even if names leave the buffer, replace only a bounded number per rebalance.
    minimum_retained = max(0, min(len(current_holdings), top_n) - max_replacements)
    if len(retained) < minimum_retained:
        remaining = sorted(
            (code for code in current_holdings if code not in retained),
            key=lambda code: rank.get(code, float("inf")),
        )
        retained.extend(remaining[: minimum_retained - len(retained)])

    targets = retained[:top_n]
    for code in ranked_codes:
        if len(targets) >= top_n:
            break
        if code not in targets:
            targets.append(code)
    return targets


def init(context):
    path = _prediction_path()
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    scores = pd.read_csv(path)
    required = {"signal_date", "code", "prediction_score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    scores["signal_date"] = pd.to_datetime(scores["signal_date"]).dt.date
    scores["code"] = scores["code"].map(normalize_order_book_id)
    scores["prediction_score"] = pd.to_numeric(scores["prediction_score"], errors="coerce")
    scores = scores.dropna(subset=["prediction_score"])
    scores = scores[scores["prediction_score"].map(isfinite)]
    context.rank_smoothing_weights = parse_smoothing_weights(
        os.environ.get("ALPHAMINING_RANK_SMOOTHING_WEIGHTS")
    )
    context.scores_by_date = build_smoothed_scores(scores, context.rank_smoothing_weights)
    context.signal_dates = sorted(context.scores_by_date)
    context.top_n = int(os.environ.get("ALPHAMINING_TOP_N", "30"))
    context.rebalance_days = int(os.environ.get("ALPHAMINING_REBALANCE_DAYS", "5"))
    context.cash_buffer = float(os.environ.get("ALPHAMINING_CASH_BUFFER", "0.98"))
    context.hold_buffer_rank = int(
        os.environ.get("ALPHAMINING_HOLD_BUFFER_RANK", str(context.top_n * 2))
    )
    context.max_replacement_ratio = float(
        os.environ.get("ALPHAMINING_MAX_REPLACEMENT_RATIO", "0.25")
    )
    context.min_holding_days = int(os.environ.get("ALPHAMINING_MIN_HOLDING_DAYS", "10"))
    context.portfolio_mode = os.environ.get(
        "ALPHAMINING_PORTFOLIO_MODE", "equal_weight"
    ).strip().lower()
    if context.portfolio_mode not in {"equal_weight", "benchmark_optimized"}:
        raise ValueError(
            "ALPHAMINING_PORTFOLIO_MODE must be equal_weight or benchmark_optimized"
        )
    context.weights_by_date = {}
    context.weight_dates = []
    context.optimizer_config = None
    if context.portfolio_mode == "benchmark_optimized":
        weight_path = os.environ.get("ALPHAMINING_INDEX_WEIGHTS")
        index_key = os.environ.get("ALPHAMINING_INDEX_KEY")
        if not weight_path or not index_key:
            raise ValueError(
                "benchmark_optimized mode requires ALPHAMINING_INDEX_WEIGHTS "
                "and ALPHAMINING_INDEX_KEY"
            )
        context.weights_by_date = load_weight_history_for_strategy(
            weight_path,
            index_key,
            start_date=min(context.signal_dates),
            end_date=max(context.signal_dates),
        )
        context.weight_dates = sorted(context.weights_by_date)
        context.optimizer_config = PortfolioOptimizerConfig(
            max_names=_env_int("ALPHAMINING_OPTIMIZER_MAX_NAMES", context.top_n * 2),
            alpha_top_n=context.top_n,
            cash_buffer=context.cash_buffer,
            alpha_strength=_env_float("ALPHAMINING_ALPHA_STRENGTH", 0.05),
            risk_aversion=_env_float("ALPHAMINING_RISK_AVERSION", 1.0),
            turnover_penalty=_env_float("ALPHAMINING_TURNOVER_PENALTY", 0.01),
            max_active_weight=_env_float("ALPHAMINING_MAX_ACTIVE_WEIGHT", 0.01),
            max_stock_weight=_env_float("ALPHAMINING_MAX_STOCK_WEIGHT", 0.10),
            max_rebalance_turnover=_env_float(
                "ALPHAMINING_MAX_REBALANCE_TURNOVER", 0.20
            ),
            estimated_buy_cost=_env_float("ALPHAMINING_ESTIMATED_BUY_COST", 0.0018),
            estimated_sell_cost=_env_float("ALPHAMINING_ESTIMATED_SELL_COST", 0.0023),
        )
    context.trading_day_count = 0
    context.last_signal_date = None
    context.holding_days = {}
    print(
        "[Strategy] turnover_control "
        f"smoothing_weights={context.rank_smoothing_weights} "
        f"hold_buffer_rank={context.hold_buffer_rank} "
        f"max_replacement_ratio={context.max_replacement_ratio:.2f} "
        f"min_holding_days={context.min_holding_days} "
        f"portfolio_mode={context.portfolio_mode}",
        flush=True,
    )


def _latest_lagged_signal(context, current_date):
    # Strict inequality is intentional: same-day close-derived scores cannot trade the same bar.
    candidates = [date for date in context.signal_dates if date < current_date]
    return candidates[-1] if candidates else None


def _current_portfolio_weights(context, bar_dict) -> dict[str, float]:
    total_value = float(getattr(context.portfolio, "total_value", 0.0) or 0.0)
    if total_value <= 0:
        return {}
    output = {}
    for code, position in getattr(context.portfolio, "positions", {}).items():
        quantity = float(
            getattr(position, "quantity", getattr(position, "total_quantity", 0)) or 0
        )
        if quantity <= 0:
            continue
        market_value = getattr(position, "market_value", None)
        if market_value is None:
            bar = bar_dict.get(code) if hasattr(bar_dict, "get") else None
            last = float(getattr(bar, "last", 0.0) or 0.0) if bar is not None else 0.0
            market_value = quantity * last
        weight = float(market_value or 0.0) / total_value
        if weight > 0:
            output[str(code)] = weight
    return output


def handle_bar(context, bar_dict):
    context.trading_day_count += 1
    positions = getattr(context.portfolio, "positions", {})
    current_holdings = [
        str(code)
        for code, position in positions.items()
        if (getattr(position, "quantity", getattr(position, "total_quantity", 0)) or 0) > 0
    ]
    current_set = set(current_holdings)
    context.holding_days = {
        code: context.holding_days.get(code, 0) + 1 for code in current_holdings
    }
    if (context.trading_day_count - 1) % context.rebalance_days != 0:
        return
    current_date = context.now.date()
    signal_date = _latest_lagged_signal(context, current_date)
    if signal_date is None or signal_date == context.last_signal_date:
        return
    ranked = context.scores_by_date[signal_date]
    if ranked.empty:
        return
    optimizer_diagnostics = None
    if context.portfolio_mode == "benchmark_optimized":
        weight_date = latest_weight_date(context.weight_dates, signal_date)
        if weight_date is None:
            print(
                f"[Strategy] no benchmark weights signal_date={signal_date}", flush=True
            )
            return
        optimized, optimizer_diagnostics = optimize_benchmark_portfolio(
            ranked,
            context.weights_by_date[weight_date],
            current_weights=_current_portfolio_weights(context, bar_dict),
            config=context.optimizer_config,
        )
        target_portfolio = optimized.set_index("code")["target_weight"].to_dict()
        targets = list(target_portfolio)
    else:
        targets = select_turnover_controlled_targets(
            ranked_codes=ranked["code"].tolist(),
            current_holdings=current_holdings,
            holding_days=context.holding_days,
            top_n=context.top_n,
            hold_buffer_rank=context.hold_buffer_rank,
            max_replacement_ratio=context.max_replacement_ratio,
            min_holding_days=context.min_holding_days,
        )
        if not targets:
            return
        weight = context.cash_buffer / len(targets)
        target_portfolio = dict.fromkeys(targets, weight)
    added = len(set(targets) - current_set)
    removed = len(current_set - set(targets))
    print(
        f"[Strategy] rebalance trade_date={current_date} signal_date={signal_date} "
        f"holdings={len(current_holdings)} targets={len(targets)} "
        f"added={added} removed={removed}",
        flush=True,
    )
    if optimizer_diagnostics is not None:
        print(
            "[Strategy] optimizer "
            + " ".join(f"{key}={value}" for key, value in optimizer_diagnostics.items()),
            flush=True,
        )
    # RQAlphaPlus performs stock T+1 checks, lot rounding, price-limit checks,
    # commissions, stamp duty, slippage, and trade recording inside its engine.
    order_target_portfolio(target_portfolio)
    context.last_signal_date = signal_date
