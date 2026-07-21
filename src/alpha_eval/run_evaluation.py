from __future__ import annotations

import argparse

import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--price", default="data/daily_price.pkl")
    parser.add_argument("--factors", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--metadata", default="results/alpha_pool.csv")
    parser.add_argument("--output", default="results/alpha_eval_result.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    values = dict(config["alpha_eval"])
    values["horizon"] = int(config["dataset"]["horizon"])
    price = pd.read_pickle(args.price)
    factors = pd.read_pickle(args.factors)
    metadata = pd.read_csv(args.metadata)
    result = AlphaEval(price, factors, AlphaEvalConfig(**values)).evaluate(metadata, args.output)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

