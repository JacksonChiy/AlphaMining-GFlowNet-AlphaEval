from __future__ import annotations

import argparse
import shutil
from argparse import Namespace
from pathlib import Path

import pandas as pd

from rqalpha_strategy.run_backtest import run as run_rqalpha
from src.alpha_eval import AlphaEval, AlphaEvalConfig
from src.data_loader import prepare_price_csv
from src.gflownet.run_training import run as run_gflownet
from src.model import LightGBMConfig, LightGBMFusion
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily research pipeline")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Only for smoke tests; formal training enforces an NVIDIA A100.",
    )
    parser.add_argument("--rqalpha-bundle", default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    price = prepare_price_csv(
        config["dataset"]["file"],
        config["dataset"]["output"],
        "results/data_quality_report.json",
    )
    experiment_dir = run_gflownet(args.config, not args.allow_non_a100, args.pool_size)
    factor_matrix = pd.read_pickle("results/alpha_factor_matrix.pkl")
    metadata = pd.read_csv("results/alpha_pool.csv")
    eval_values = dict(config["alpha_eval"])
    eval_values["horizon"] = config["dataset"]["horizon"]
    evaluation = AlphaEval(price, factor_matrix, AlphaEvalConfig(**eval_values)).evaluate(metadata)
    selected = evaluation.loc[evaluation["dpp_selected"].astype(bool), "factor"].tolist()
    prediction = LightGBMFusion(LightGBMConfig(**config["lightgbm"])).fit_predict(
        price, factor_matrix, selected, "results/lightgbm"
    )
    shutil.copy2("results/alpha_eval_result.csv", experiment_dir / "alpha_eval_result.csv")
    shutil.copy2("results/lightgbm/model_metrics.csv", experiment_dir / "lgbm_model_metrics.csv")
    prediction.to_csv(experiment_dir / "prediction_score.csv", index=False)

    if args.rqalpha_bundle:
        backtest_args = Namespace(
            predictions="results/lightgbm/prediction_score.csv",
            bundle=args.rqalpha_bundle,
            output_dir=str(experiment_dir / "backtest_report"),
            initial_cash=config["backtest"]["initial_cash"],
            benchmark=config["backtest"]["benchmark"],
            top_n=config["backtest"]["top_n"],
            rebalance_days=config["backtest"]["rebalance_days"],
            slippage=config["backtest"]["slippage"],
        )
        run_rqalpha(backtest_args)
    else:
        print("RQAlphaPlus skipped: pass --rqalpha-bundle with an authorized local bundle.")
    print("Pipeline artifacts:", experiment_dir.resolve())


if __name__ == "__main__":
    main()
