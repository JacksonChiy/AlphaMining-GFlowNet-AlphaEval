"""RQAlphaPlus daily Top-N strategy using strictly lagged LightGBM scores."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


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
    scores = scores.dropna(subset=["prediction_score"]).sort_values(
        ["signal_date", "prediction_score"], ascending=[True, False]
    )
    context.scores_by_date = {date: group for date, group in scores.groupby("signal_date", sort=True)}
    context.signal_dates = sorted(context.scores_by_date)
    context.top_n = int(os.environ.get("ALPHAMINING_TOP_N", "30"))
    context.rebalance_days = int(os.environ.get("ALPHAMINING_REBALANCE_DAYS", "5"))
    context.cash_buffer = float(os.environ.get("ALPHAMINING_CASH_BUFFER", "0.98"))
    context.trading_day_count = 0
    context.last_signal_date = None


def _latest_lagged_signal(context, current_date):
    # Strict inequality is intentional: same-day close-derived scores cannot trade the same bar.
    candidates = [date for date in context.signal_dates if date < current_date]
    return candidates[-1] if candidates else None


def handle_bar(context, bar_dict):
    context.trading_day_count += 1
    if (context.trading_day_count - 1) % context.rebalance_days != 0:
        return
    current_date = context.now.date()
    signal_date = _latest_lagged_signal(context, current_date)
    if signal_date is None or signal_date == context.last_signal_date:
        return
    candidates = context.scores_by_date[signal_date].head(context.top_n)
    if candidates.empty:
        return
    weight = context.cash_buffer / len(candidates)
    target_portfolio = dict.fromkeys(candidates["code"].tolist(), weight)
    # RQAlphaPlus performs stock T+1 checks, lot rounding, price-limit checks,
    # commissions, stamp duty, slippage, and trade recording inside its engine.
    order_target_portfolio(target_portfolio)
    context.last_signal_date = signal_date

