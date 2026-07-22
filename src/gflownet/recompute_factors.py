from __future__ import annotations

import argparse

import pandas as pd

from src.gflownet.factor_pool import execute_saved_alpha_pool
from src.operators import configure_time_series_from_mapping, get_time_series_backend_info
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute an existing alpha pool on complete price history"
    )
    parser.add_argument("--config", default="configs/quick_training_config.yaml")
    parser.add_argument("--metadata", default="results/alpha_pool.csv")
    parser.add_argument("--matrix", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--oos-matrix", default="results/alpha_factor_matrix_oos.pkl")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_time_series_from_mapping(config.get("operators"))
    print(f"[FactorPool] time_series_backend={get_time_series_backend_info()}", flush=True)
    execute_saved_alpha_pool(
        pd.read_pickle(config["dataset"]["output"]),
        args.metadata,
        args.matrix,
        args.oos_matrix,
        config["dataset"].get("out_of_sample_start_date"),
        config["dataset"].get("out_of_sample_end_date"),
    )


if __name__ == "__main__":
    main()
