from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


BACKTEST_DEFAULTS = {
    "initial_cash": 1_000_000,
    "benchmark": "000300.XSHG",
    "top_n": 30,
    "rebalance_days": 5,
    "slippage": 0.001,
    "cash_buffer": 0.98,
    "rank_smoothing_weights": [0.5, 0.3, 0.2],
    "hold_buffer_rank": 200,
    "max_replacement_ratio": 0.25,
    "min_holding_days": 10,
}


def load_backtest_settings(
    config_path: str | Path | None,
    **overrides: float | int | str | None,
) -> dict[str, Any]:
    """Load backtest settings from YAML, with explicit CLI values taking precedence."""
    settings: dict[str, Any] = dict(BACKTEST_DEFAULTS)
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Backtest config not found: {path.resolve()}")
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        section = config.get("backtest")
        if not isinstance(section, dict):
            raise ValueError(f"Config has no backtest mapping: {path.resolve()}")
        for key in BACKTEST_DEFAULTS:
            if section.get(key) is not None:
                settings[key] = section[key]
    for key, value in overrides.items():
        if key in settings and value is not None:
            settings[key] = value
    settings["initial_cash"] = float(settings["initial_cash"])
    settings["benchmark"] = str(settings["benchmark"])
    settings["top_n"] = int(settings["top_n"])
    settings["rebalance_days"] = int(settings["rebalance_days"])
    settings["slippage"] = float(settings["slippage"])
    settings["cash_buffer"] = float(settings["cash_buffer"])
    raw_weights = settings["rank_smoothing_weights"]
    if isinstance(raw_weights, str):
        raw_weights = [item.strip() for item in raw_weights.split(",") if item.strip()]
    settings["rank_smoothing_weights"] = [float(weight) for weight in raw_weights]
    settings["hold_buffer_rank"] = int(settings["hold_buffer_rank"])
    settings["max_replacement_ratio"] = float(settings["max_replacement_ratio"])
    settings["min_holding_days"] = int(settings["min_holding_days"])
    if settings["initial_cash"] <= 0:
        raise ValueError("initial_cash must be positive")
    if settings["top_n"] <= 0 or settings["rebalance_days"] <= 0:
        raise ValueError("top_n and rebalance_days must be positive")
    if settings["slippage"] < 0:
        raise ValueError("slippage must be non-negative")
    if not 0 < settings["cash_buffer"] <= 1:
        raise ValueError("cash_buffer must be in (0, 1]")
    weights = settings["rank_smoothing_weights"]
    if not weights or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("rank_smoothing_weights must be non-negative and sum to a positive value")
    weight_sum = sum(weights)
    settings["rank_smoothing_weights"] = [weight / weight_sum for weight in weights]
    if settings["hold_buffer_rank"] < settings["top_n"]:
        raise ValueError("hold_buffer_rank must be greater than or equal to top_n")
    if not 0 < settings["max_replacement_ratio"] <= 1:
        raise ValueError("max_replacement_ratio must be in (0, 1]")
    if settings["min_holding_days"] < 0:
        raise ValueError("min_holding_days must be non-negative")
    return settings


def build_config(
    predictions: str | Path,
    bundle_path: str | Path,
    output_dir: str | Path,
    initial_cash: float = 1_000_000,
    benchmark: str = "000300.XSHG",
    slippage: float = 0.001,
) -> dict:
    scores = pd.read_csv(predictions, usecols=["signal_date"])
    dates = pd.to_datetime(scores["signal_date"])
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_date = dates.min().date()
    end_date = dates.max().date() + timedelta(days=10)
    return {
        "base": {
            "data_bundle_path": str(Path(bundle_path).expanduser().resolve()),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "frequency": "1d",
            "accounts": {"STOCK": initial_cash},
            "rqdatac_uri": "disabled",
            "auto_update_bundle": False,
            # This strategy only trades A-shares. Keep the currently compatible
            # zero rate explicit instead of relying on a changing framework default.
            "capital_gain_tax_rate": 0.0,
        },
        "extra": {"log_level": "info"},
        "mod": {
            "sys_accounts": {"stock_t1": True},
            "sys_simulation": {
                "enabled": True,
                "matching_type": "current_bar",
                "slippage_model": "PriceRatioSlippage",
                "slippage": slippage,
                "price_limit": True,
                "volume_limit": True,
                "volume_percent": 0.25,
            },
            "sys_transaction_cost": {
                "stock_min_commission": 5,
                "stock_commission_multiplier": 1,
                "tax_multiplier": 1,
                "pit_tax": True,
            },
            # Disable non-stock modules so a stock-only backtest never attempts
            # an RQData update for missing option/fund instrument bundles.
            "option": {"enabled": False},
            "fund": {"enabled": False},
            "convertible": {"enabled": False},
            "spot": {"enabled": False},
            "rqfactor": {"enabled": False},
            "sys_analyser": {
                "enabled": True,
                "benchmark": benchmark,
                "record": True,
                "strategy_name": "GFlowNet-AlphaEval-LGBM-Daily",
                "output_file": str(output_dir / "backtest_result.pkl"),
                "report_save_path": str(output_dir),
                "plot": True,
                "plot_save_file": str(output_dir / "equity_curve.png"),
            },
            "sys_progress": {"enabled": True, "show": True},
        },
    }


def run(args: argparse.Namespace) -> dict:
    try:
        import rqalpha_plus
    except ImportError as exc:
        raise RuntimeError(
            "RQAlphaPlus is licensed software. Install it from your authorized Ricequant channel "
            "and prepare its local bundle before running this stage."
        ) from exc
    prediction_path = Path(args.predictions).resolve()
    strategy_path = Path(__file__).with_name("strategy.py")
    os.environ["ALPHAMINING_PREDICTIONS"] = str(prediction_path)
    os.environ["ALPHAMINING_TOP_N"] = str(args.top_n)
    os.environ["ALPHAMINING_REBALANCE_DAYS"] = str(args.rebalance_days)
    os.environ["ALPHAMINING_CASH_BUFFER"] = str(args.cash_buffer)
    os.environ["ALPHAMINING_RANK_SMOOTHING_WEIGHTS"] = ",".join(
        str(weight) for weight in args.rank_smoothing_weights
    )
    os.environ["ALPHAMINING_HOLD_BUFFER_RANK"] = str(args.hold_buffer_rank)
    os.environ["ALPHAMINING_MAX_REPLACEMENT_RATIO"] = str(args.max_replacement_ratio)
    os.environ["ALPHAMINING_MIN_HOLDING_DAYS"] = str(args.min_holding_days)
    config = build_config(
        prediction_path,
        args.bundle,
        args.output_dir,
        args.initial_cash,
        args.benchmark,
        args.slippage,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effective_settings = {
        "predictions": str(prediction_path),
        "bundle": str(Path(args.bundle).expanduser().resolve()),
        "output_dir": str(output.resolve()),
        "initial_cash": float(args.initial_cash),
        "benchmark": str(args.benchmark),
        "top_n": int(args.top_n),
        "rebalance_days": int(args.rebalance_days),
        "slippage": float(args.slippage),
        "cash_buffer": float(args.cash_buffer),
        "rank_smoothing_weights": list(args.rank_smoothing_weights),
        "hold_buffer_rank": int(args.hold_buffer_rank),
        "max_replacement_ratio": float(args.max_replacement_ratio),
        "min_holding_days": int(args.min_holding_days),
    }
    print(
        "[Backtest] effective_settings "
        + " ".join(f"{key}={value}" for key, value in effective_settings.items()),
        flush=True,
    )
    (output / "backtest_effective_config.json").write_text(
        json.dumps(effective_settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = rqalpha_plus.run_file(str(strategy_path), config=config)
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    (output / "backtest_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    required_outputs = [
        output / "backtest_result.pkl",
        output / "equity_curve.png",
        output / "backtest_summary.json",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"RQAlphaPlus completed without required report artifacts: {missing}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--predictions", default="results/lightgbm/prediction_score.csv")
    parser.add_argument("--bundle", default="~/.rqalpha-plus/bundle")
    parser.add_argument("--output-dir", default="results/backtest_report")
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--rebalance-days", type=int, default=None)
    parser.add_argument("--slippage", type=float, default=None)
    parser.add_argument("--cash-buffer", type=float, default=None)
    parser.add_argument(
        "--rank-smoothing-weights",
        default=None,
        help="Newest-to-oldest comma-separated rank weights, for example 0.5,0.3,0.2",
    )
    parser.add_argument("--hold-buffer-rank", type=int, default=None)
    parser.add_argument("--max-replacement-ratio", type=float, default=None)
    parser.add_argument("--min-holding-days", type=int, default=None)
    args = parser.parse_args()
    settings = load_backtest_settings(
        args.config,
        initial_cash=args.initial_cash,
        benchmark=args.benchmark,
        top_n=args.top_n,
        rebalance_days=args.rebalance_days,
        slippage=args.slippage,
        cash_buffer=args.cash_buffer,
        rank_smoothing_weights=args.rank_smoothing_weights,
        hold_buffer_rank=args.hold_buffer_rank,
        max_replacement_ratio=args.max_replacement_ratio,
        min_holding_days=args.min_holding_days,
    )
    for key, value in settings.items():
        setattr(args, key, value)
    run(args)


if __name__ == "__main__":
    main()
