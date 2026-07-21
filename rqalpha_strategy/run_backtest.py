from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd


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
    config = build_config(
        prediction_path,
        args.bundle,
        args.output_dir,
        args.initial_cash,
        args.benchmark,
        args.slippage,
    )
    result = rqalpha_plus.run_file(str(strategy_path), config=config)
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    output = Path(args.output_dir)
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
    parser.add_argument("--predictions", default="results/lightgbm/prediction_score.csv")
    parser.add_argument("--bundle", default="~/.rqalpha-plus/bundle")
    parser.add_argument("--output-dir", default="results/backtest_report")
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--benchmark", default="000300.XSHG")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--slippage", type=float, default=0.001)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
