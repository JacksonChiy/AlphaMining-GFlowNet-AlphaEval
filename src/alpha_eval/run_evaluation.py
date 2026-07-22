from __future__ import annotations

import argparse

import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig
from src.utils import (
    load_config,
    slice_date_range,
    validate_frame_covers_period,
    validate_research_date_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--price", default="data/daily_price.pkl")
    parser.add_argument("--factors", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--metadata", default="results/alpha_pool.csv")
    parser.add_argument("--output", default="results/alpha_eval_result.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"[AlphaEval] date_split={validate_research_date_split(config)}", flush=True)
    values = dict(config["alpha_eval"])
    values["horizon"] = int(config["dataset"]["horizon"])
    price = pd.read_pickle(args.price)
    factors = pd.read_pickle(args.factors)
    price = slice_date_range(
        price,
        config["dataset"].get("mining_start_date"),
        config["dataset"].get("mining_end_date"),
        label="AlphaEval price data",
    )
    factors = slice_date_range(
        factors,
        config["dataset"].get("mining_start_date"),
        config["dataset"].get("mining_end_date"),
        label="AlphaEval factor data",
    )
    validate_frame_covers_period(
        price,
        config["dataset"]["mining_start_date"],
        config["dataset"]["mining_end_date"],
        label="AlphaEval price data",
    )
    validate_frame_covers_period(
        factors,
        config["dataset"]["mining_start_date"],
        config["dataset"]["mining_end_date"],
        label="AlphaEval factor data",
    )
    metadata = pd.read_csv(args.metadata)
    result = AlphaEval(price, factors, AlphaEvalConfig(**values)).evaluate(metadata, args.output)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
