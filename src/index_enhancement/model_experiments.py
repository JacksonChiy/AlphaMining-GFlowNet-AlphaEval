"""Run reproducible P2 LightGBM objective experiments for each index."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.index_enhancement.universe import normalize_order_book_id_series
from src.model import LightGBMConfig, LightGBMFusion
from src.utils import load_config


def run_experiments(
    training_config: str | Path,
    experiment_config: str | Path,
    price_path: str | Path,
    factor_path: str | Path,
    global_evaluation_path: str | Path,
    label_root: str | Path,
    alpha_eval_root: str | Path,
    output_root: str | Path,
    indexes: list[str],
    experiments: list[str],
) -> None:
    config = load_config(training_config)
    with Path(experiment_config).open(encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    unknown = sorted(set(experiments).difference(matrix["experiments"]))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    price = pd.read_pickle(price_path)
    factors = pd.read_pickle(factor_path)
    for frame, label in ((price, "price"), (factors, "factors")):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["code"] = normalize_order_book_id_series(frame["code"])
        if frame.duplicated(["date", "code"]).any():
            raise ValueError(f"{label} contains duplicate keys after code normalization")
    print(
        "[P2] normalized_inputs "
        f"price={price['date'].min().date()}..{price['date'].max().date()} "
        f"factors={factors['date'].min().date()}..{factors['date'].max().date()} "
        f"factor_code_example={factors['code'].iloc[0]}",
        flush=True,
    )
    global_evaluation = pd.read_csv(global_evaluation_path)
    for index_key in indexes:
        index_evaluation = Path(alpha_eval_root) / index_key / "alpha_eval_result.csv"
        evaluation = pd.read_csv(index_evaluation) if index_evaluation.exists() else global_evaluation
        selected = evaluation.loc[evaluation["dpp_selected"].astype(bool), "factor"].tolist()
        if not selected:
            raise ValueError(f"No selected factors for {index_key}")
        for experiment in experiments:
            values = dict(config["lightgbm"])
            values.update(matrix["experiments"][experiment])
            values["label_path"] = str(Path(label_root) / index_key / "labels.pkl")
            values["codes_are_normalized"] = True
            output = Path(output_root) / experiment / index_key
            print(
                f"[P2] index={index_key} experiment={experiment} factors={len(selected)}",
                flush=True,
            )
            LightGBMFusion(LightGBMConfig(**values)).fit_predict(
                price, factors, selected, output
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行三指数P2 LightGBM实验矩阵")
    parser.add_argument("--training-config", default="configs/daily/training.yaml")
    parser.add_argument("--experiment-config", default="configs/index_enhancement/model_experiments.yaml")
    parser.add_argument("--price", default="data/daily_price.pkl")
    parser.add_argument("--factors", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--evaluation", default="results/alpha_eval_result.csv")
    parser.add_argument("--labels", default="results/index_enhancement_labels")
    parser.add_argument("--index-alpha-eval", default="results/index_alpha_eval")
    parser.add_argument("--output", default="results/index_model_experiments")
    parser.add_argument("--indexes", default="csi300,csi500,csi1000")
    parser.add_argument(
        "--experiments",
        default="cross_sectional_rank,excess_huber,excess_top_weighted,lambdarank",
    )
    args = parser.parse_args()
    run_experiments(
        args.training_config,
        args.experiment_config,
        args.price,
        args.factors,
        args.evaluation,
        args.labels,
        args.index_alpha_eval,
        args.output,
        [value.strip() for value in args.indexes.split(",") if value.strip()],
        [value.strip() for value in args.experiments.split(",") if value.strip()],
    )


if __name__ == "__main__":
    main()
