from __future__ import annotations

import argparse

import pandas as pd

from src.model import LightGBMConfig, LightGBMFusion
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--price", default="data/daily_price.pkl")
    parser.add_argument("--factors", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--evaluation", default="results/alpha_eval_result.csv")
    parser.add_argument("--output-dir", default="results/lightgbm")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = pd.read_csv(args.evaluation)
    selected = evaluation.loc[evaluation["dpp_selected"].astype(bool), "factor"].tolist()
    fusion = LightGBMFusion(LightGBMConfig(**config["lightgbm"]))
    prediction = fusion.fit_predict(
        pd.read_pickle(args.price), pd.read_pickle(args.factors), selected, args.output_dir
    )
    print(prediction.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()

