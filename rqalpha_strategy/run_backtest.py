from __future__ import annotations

import argparse
import copy
import importlib.metadata
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
    "capital_gain_tax_rate": 0.0,
    "stock_min_commission": 5.0,
    "stock_commission_multiplier": 1.0,
    "tax_multiplier": 1.0,
    "pit_tax": True,
    "volume_percent": 0.25,
    "portfolio_mode": "equal_weight",
    "index_weights": None,
    "index_key": None,
    "optimizer_max_names": 200,
    "alpha_strength": 0.05,
    "risk_aversion": 1.0,
    "turnover_penalty": 0.01,
    "max_active_weight": 0.01,
    "max_stock_weight": 0.10,
    "max_rebalance_turnover": 0.20,
    "estimated_buy_cost": 0.0018,
    "estimated_sell_cost": 0.0023,
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
    settings["capital_gain_tax_rate"] = float(settings["capital_gain_tax_rate"])
    settings["stock_min_commission"] = float(settings["stock_min_commission"])
    settings["stock_commission_multiplier"] = float(
        settings["stock_commission_multiplier"]
    )
    settings["tax_multiplier"] = float(settings["tax_multiplier"])
    settings["pit_tax"] = bool(settings["pit_tax"])
    settings["volume_percent"] = float(settings["volume_percent"])
    settings["portfolio_mode"] = str(settings["portfolio_mode"]).strip().lower()
    settings["index_weights"] = (
        str(settings["index_weights"]) if settings["index_weights"] else None
    )
    settings["index_key"] = str(settings["index_key"]) if settings["index_key"] else None
    settings["optimizer_max_names"] = int(settings["optimizer_max_names"])
    for key in (
        "alpha_strength",
        "risk_aversion",
        "turnover_penalty",
        "max_active_weight",
        "max_stock_weight",
        "max_rebalance_turnover",
        "estimated_buy_cost",
        "estimated_sell_cost",
    ):
        settings[key] = float(settings[key])
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
    if settings["capital_gain_tax_rate"] < 0:
        raise ValueError("capital_gain_tax_rate must be non-negative")
    if settings["stock_min_commission"] < 0:
        raise ValueError("stock_min_commission must be non-negative")
    if settings["stock_commission_multiplier"] < 0 or settings["tax_multiplier"] < 0:
        raise ValueError("transaction-cost multipliers must be non-negative")
    if not 0 < settings["volume_percent"] <= 1:
        raise ValueError("volume_percent must be in (0, 1]")
    if settings["portfolio_mode"] not in {"equal_weight", "benchmark_optimized"}:
        raise ValueError("portfolio_mode must be equal_weight or benchmark_optimized")
    if settings["portfolio_mode"] == "benchmark_optimized":
        if not settings["index_weights"] or not settings["index_key"]:
            raise ValueError(
                "benchmark_optimized mode requires index_weights and index_key"
            )
    if settings["optimizer_max_names"] <= 0:
        raise ValueError("optimizer_max_names must be positive")
    for key in (
        "alpha_strength",
        "risk_aversion",
        "turnover_penalty",
        "max_active_weight",
        "max_stock_weight",
        "max_rebalance_turnover",
        "estimated_buy_cost",
        "estimated_sell_cost",
    ):
        if settings[key] < 0:
            raise ValueError(f"{key} must be non-negative")
    return settings


def build_config(
    predictions: str | Path,
    bundle_path: str | Path,
    output_dir: str | Path,
    initial_cash: float = 1_000_000,
    benchmark: str = "000300.XSHG",
    slippage: float = 0.001,
    capital_gain_tax_rate: float = 0.0,
    stock_min_commission: float = 5.0,
    stock_commission_multiplier: float = 1.0,
    tax_multiplier: float = 1.0,
    pit_tax: bool = True,
    volume_percent: float = 0.25,
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
            "capital_gain_tax_rate": capital_gain_tax_rate,
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
                "volume_percent": volume_percent,
            },
            "sys_transaction_cost": {
                "stock_min_commission": stock_min_commission,
                "stock_commission_multiplier": stock_commission_multiplier,
                "tax_multiplier": tax_multiplier,
                "pit_tax": pit_tax,
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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_config(value, key: str = ""):
    """Remove credential-like values before persisting a resolved config."""
    sensitive = ("password", "passwd", "token", "secret", "credential")
    if any(part in key.lower() for part in sensitive):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_config(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_config(item, key) for item in value]
    return _json_safe(value)


def resolve_rqalpha_config(config: dict, strategy_path: str | Path) -> dict:
    """Resolve RQAlphaPlus defaults exactly as ``run_file`` does."""
    from rqalpha.utils.config import parse_config

    candidate = copy.deepcopy(config)
    candidate.setdefault("base", {})["strategy_file"] = str(Path(strategy_path).resolve())
    resolved = parse_config(candidate)
    as_dict = resolved.convert_to_dict() if hasattr(resolved, "convert_to_dict") else dict(resolved)
    return _sanitize_config(as_dict)


def resolved_transaction_cost_assumptions(resolved: dict) -> dict:
    """Make RQAlpha's internal base rates explicit in the experiment record."""
    transaction = resolved.get("mod", {}).get("sys_transaction_cost", {})
    commission_multiplier = float(transaction.get("stock_commission_multiplier", 1.0))
    tax_multiplier = float(transaction.get("tax_multiplier", 1.0))
    pit_tax = bool(transaction.get("pit_tax", False))
    return {
        "rqalpha_stock_commission_base_rate": 0.0008,
        "stock_commission_multiplier": commission_multiplier,
        "effective_stock_commission_rate": 0.0008 * commission_multiplier,
        "stock_min_commission": float(transaction.get("stock_min_commission", 5.0)),
        "pit_tax": pit_tax,
        "stamp_duty_rate_before_2023_08_28": 0.001 * tax_multiplier,
        "stamp_duty_rate_from_2023_08_28": (0.0005 if pit_tax else 0.001) * tax_multiplier,
        "tax_multiplier": tax_multiplier,
        "note": "Rates are resolved from RQAlpha's stock transaction-cost model and multipliers.",
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run(args: argparse.Namespace) -> dict:
    try:
        import rqalpha_plus
    except ImportError as exc:
        raise RuntimeError(
            "RQAlphaPlus is licensed software. Install it from your authorized Ricequant channel "
            "and prepare its local bundle before running this stage."
        ) from exc
    # RQAlphaPlus is intentionally local-only in this project.  Do not inherit
    # Colab/Git proxy variables into the licensed local backtest runtime.
    for proxy_name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ):
        os.environ.pop(proxy_name, None)
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
    os.environ["ALPHAMINING_PORTFOLIO_MODE"] = str(args.portfolio_mode)
    if args.index_weights:
        os.environ["ALPHAMINING_INDEX_WEIGHTS"] = str(
            Path(args.index_weights).expanduser().resolve()
        )
    if args.index_key:
        os.environ["ALPHAMINING_INDEX_KEY"] = str(args.index_key)
    optimizer_environment = {
        "ALPHAMINING_OPTIMIZER_MAX_NAMES": args.optimizer_max_names,
        "ALPHAMINING_ALPHA_STRENGTH": args.alpha_strength,
        "ALPHAMINING_RISK_AVERSION": args.risk_aversion,
        "ALPHAMINING_TURNOVER_PENALTY": args.turnover_penalty,
        "ALPHAMINING_MAX_ACTIVE_WEIGHT": args.max_active_weight,
        "ALPHAMINING_MAX_STOCK_WEIGHT": args.max_stock_weight,
        "ALPHAMINING_MAX_REBALANCE_TURNOVER": args.max_rebalance_turnover,
        "ALPHAMINING_ESTIMATED_BUY_COST": args.estimated_buy_cost,
        "ALPHAMINING_ESTIMATED_SELL_COST": args.estimated_sell_cost,
    }
    for key, value in optimizer_environment.items():
        os.environ[key] = str(value)
    config = build_config(
        prediction_path,
        args.bundle,
        args.output_dir,
        args.initial_cash,
        args.benchmark,
        args.slippage,
        args.capital_gain_tax_rate,
        args.stock_min_commission,
        args.stock_commission_multiplier,
        args.tax_multiplier,
        args.pit_tax,
        args.volume_percent,
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
        "portfolio_mode": str(args.portfolio_mode),
        "index_weights": str(Path(args.index_weights).expanduser().resolve())
        if args.index_weights else None,
        "index_key": args.index_key,
        "optimizer_max_names": int(args.optimizer_max_names),
        "alpha_strength": float(args.alpha_strength),
        "risk_aversion": float(args.risk_aversion),
        "turnover_penalty": float(args.turnover_penalty),
        "max_active_weight": float(args.max_active_weight),
        "max_stock_weight": float(args.max_stock_weight),
        "max_rebalance_turnover": float(args.max_rebalance_turnover),
        "estimated_buy_cost": float(args.estimated_buy_cost),
        "estimated_sell_cost": float(args.estimated_sell_cost),
    }
    resolved_rqalpha_config = resolve_rqalpha_config(config, strategy_path)
    effective_settings["rqalpha_versions"] = {
        "rqalpha_plus": _package_version("rqalpha-plus"),
        "rqalpha": _package_version("rqalpha"),
    }
    effective_settings["transaction_cost_assumptions"] = (
        resolved_transaction_cost_assumptions(resolved_rqalpha_config)
    )
    print(
        "[Backtest] effective_settings "
        + " ".join(f"{key}={value}" for key, value in effective_settings.items()),
        flush=True,
    )
    (output / "backtest_effective_config.json").write_text(
        json.dumps(effective_settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "rqalpha_effective_config.json").write_text(
        json.dumps(resolved_rqalpha_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = rqalpha_plus.run_file(str(strategy_path), config=config)
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    if not summary and (output / "portfolio.csv").exists() and (output / "trades.csv").exists():
        from src.index_enhancement.diagnostics import evaluate_backtest

        summary = evaluate_backtest(output, str(args.index_key or args.benchmark))
    (output / "backtest_summary.json").write_text(
        json.dumps(dict(summary), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    required_outputs = [
        output / "backtest_result.pkl",
        output / "equity_curve.png",
        output / "backtest_summary.json",
        output / "backtest_effective_config.json",
        output / "rqalpha_effective_config.json",
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
    parser.add_argument("--capital-gain-tax-rate", type=float, default=None)
    parser.add_argument("--stock-min-commission", type=float, default=None)
    parser.add_argument("--stock-commission-multiplier", type=float, default=None)
    parser.add_argument("--tax-multiplier", type=float, default=None)
    parser.add_argument(
        "--pit-tax", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--volume-percent", type=float, default=None)
    parser.add_argument(
        "--portfolio-mode",
        choices=["equal_weight", "benchmark_optimized"],
        default=None,
    )
    parser.add_argument("--index-weights", default=None)
    parser.add_argument("--index-key", default=None)
    parser.add_argument("--optimizer-max-names", type=int, default=None)
    parser.add_argument("--alpha-strength", type=float, default=None)
    parser.add_argument("--risk-aversion", type=float, default=None)
    parser.add_argument("--turnover-penalty", type=float, default=None)
    parser.add_argument("--max-active-weight", type=float, default=None)
    parser.add_argument("--max-stock-weight", type=float, default=None)
    parser.add_argument("--max-rebalance-turnover", type=float, default=None)
    parser.add_argument("--estimated-buy-cost", type=float, default=None)
    parser.add_argument("--estimated-sell-cost", type=float, default=None)
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
        capital_gain_tax_rate=args.capital_gain_tax_rate,
        stock_min_commission=args.stock_min_commission,
        stock_commission_multiplier=args.stock_commission_multiplier,
        tax_multiplier=args.tax_multiplier,
        pit_tax=args.pit_tax,
        volume_percent=args.volume_percent,
        portfolio_mode=args.portfolio_mode,
        index_weights=args.index_weights,
        index_key=args.index_key,
        optimizer_max_names=args.optimizer_max_names,
        alpha_strength=args.alpha_strength,
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
        max_active_weight=args.max_active_weight,
        max_stock_weight=args.max_stock_weight,
        max_rebalance_turnover=args.max_rebalance_turnover,
        estimated_buy_cost=args.estimated_buy_cost,
        estimated_sell_cost=args.estimated_sell_cost,
    )
    for key, value in settings.items():
        setattr(args, key, value)
    run(args)


if __name__ == "__main__":
    main()
