from __future__ import annotations

import argparse

import pandas as pd

from src.model import LightGBMConfig, LightGBMFusion
from src.utils import (
    load_config,
    validate_frame_covers_period,
    validate_research_date_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--price", default="data/daily_price.pkl")
    parser.add_argument("--factors", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--evaluation", default="results/alpha_eval_result.csv")
    parser.add_argument("--output-dir", default="results/lightgbm")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"[LightGBM] date_split={validate_research_date_split(config)}", flush=True)
    evaluation = pd.read_csv(args.evaluation)
    selected = evaluation.loc[evaluation["dpp_selected"].astype(bool), "factor"].tolist()
    fusion = LightGBMFusion(LightGBMConfig(**config["lightgbm"]))
    price = pd.read_pickle(args.price)
    factors = pd.read_pickle(args.factors)
    validate_frame_covers_period(
        price,
        config["dataset"]["mining_start_date"],
        config["dataset"]["mining_end_date"],
        label="LightGBM price data",
    )
    validate_frame_covers_period(
        factors,
        config["dataset"]["mining_start_date"],
        config["dataset"]["mining_end_date"],
        label="LightGBM factor data",
    )
    prediction = fusion.fit_predict(
        price, factors, selected, args.output_dir
    )
    print(prediction.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
