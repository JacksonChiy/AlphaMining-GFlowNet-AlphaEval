"""P3: evaluate and DPP-select factors independently inside each index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig
from src.index_enhancement.universe import normalize_order_book_id_series
from src.utils import load_config, slice_date_range


def _normalize_index_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["code"] = normalize_order_book_id_series(result["code"])
    if result[["date", "code"]].isna().any().any():
        raise ValueError(f"{label} contains invalid date/code keys")
    if result.duplicated(["date", "code"]).any():
        raise ValueError(f"{label} contains duplicate keys after code normalization")
    return result


def run_index_alpha_eval(
    config_path: str | Path,
    price_path: str | Path,
    factor_path: str | Path,
    metadata_path: str | Path,
    label_root: str | Path,
    output_root: str | Path,
    indexes: list[str],
    target_column: str = "target_excess_return",
) -> dict[str, Path]:
    config = load_config(config_path)
    values = dict(config["alpha_eval"])
    values["horizon"] = int(config["dataset"]["horizon"])
    price = pd.read_pickle(price_path)
    factors = pd.read_pickle(factor_path)
    price = _normalize_index_keys(price, "指数AlphaEval行情")
    factors = _normalize_index_keys(factors, "指数AlphaEval因子")
    metadata = pd.read_csv(metadata_path)
    start = config["dataset"].get("mining_start_date")
    end = config["dataset"].get("mining_end_date")
    price = slice_date_range(price, start, end, label="指数AlphaEval行情")
    factors = slice_date_range(factors, start, end, label="指数AlphaEval因子")
    outputs: dict[str, Path] = {}
    manifest: dict[str, object] = {"target_column": target_column, "indexes": {}}
    for index_key in indexes:
        label_path = Path(label_root) / index_key / "labels.pkl"
        labels = _normalize_index_keys(pd.read_pickle(label_path), f"{index_key}标签")
        labels = slice_date_range(labels, start, end, label=f"{index_key}标签")
        valid_labels = labels.dropna(subset=[target_column])
        index_factors = factors.merge(
            valid_labels[["date", "code"]], on=["date", "code"], how="inner", validate="one_to_one"
        )
        if index_factors.empty:
            raise ValueError(
                f"{index_key} labels and factors have no date/code overlap after "
                "normalization; audit their date ranges and code formats"
            )
        output_dir = Path(output_root) / index_key
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "alpha_eval_result.csv"
        print(
            f"[IndexAlphaEval] index={index_key} rows={len(index_factors):,} "
            f"dates={index_factors['date'].nunique()}",
            flush=True,
        )
        result = AlphaEval(
            price,
            index_factors,
            AlphaEvalConfig(**values),
            target_data=valid_labels,
            target_column=target_column,
        ).evaluate(metadata, output)
        selected = result.loc[result["dpp_selected"].astype(bool), "factor"]
        selected.to_csv(output_dir / "selected_factors.csv", index=False)
        outputs[index_key] = output
        manifest["indexes"][index_key] = {
            "rows": int(len(index_factors)),
            "dates": int(index_factors["date"].nunique()),
            "selected_factors": int(len(selected)),
            "output": str(output.resolve()),
        }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    (Path(output_root) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="三指数独立AlphaEval与DPP筛选")
    parser.add_argument("--config", default="configs/daily/training.yaml")
    parser.add_argument("--price", default="data/daily_price.pkl")
    parser.add_argument("--factors", default="results/alpha_factor_matrix.pkl")
    parser.add_argument("--metadata", default="results/alpha_pool.csv")
    parser.add_argument("--labels", default="results/index_enhancement_labels")
    parser.add_argument("--output", default="results/index_alpha_eval")
    parser.add_argument("--indexes", default="csi300,csi500,csi1000")
    parser.add_argument("--target-column", default="target_excess_return")
    args = parser.parse_args()
    run_index_alpha_eval(
        args.config, args.price, args.factors, args.metadata, args.labels, args.output,
        [value.strip() for value in args.indexes.split(",") if value.strip()],
        args.target_column,
    )


if __name__ == "__main__":
    main()
