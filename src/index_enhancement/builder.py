"""Build point-in-time index-constrained prediction files from local data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from .universe import load_components, normalize_order_book_id


def build_index_input(
    predictions: pd.DataFrame,
    components: pd.DataFrame,
    index_key: str,
) -> pd.DataFrame:
    """Filter daily model scores to historical constituents of one index."""
    required_predictions = {"signal_date", "code", "prediction_score"}
    missing = required_predictions.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    subset = components.loc[
        components["index_key"] == index_key,
        ["date", "index_key", "index_code", "index_name", "code"],
    ].copy()
    if subset.empty:
        raise ValueError(f"No component rows found for index_key={index_key!r}")

    scores = predictions.loc[:, ["signal_date", "code", "prediction_score"]].copy()
    scores["signal_date"] = pd.to_datetime(scores["signal_date"]).dt.normalize()
    scores["code"] = scores["code"].map(normalize_order_book_id)
    scores["prediction_score"] = pd.to_numeric(scores["prediction_score"], errors="coerce")
    scores = scores.dropna(subset=["prediction_score"])
    if scores.duplicated(["signal_date", "code"]).any():
        scores = scores.groupby(["signal_date", "code"], as_index=False)[
            "prediction_score"
        ].mean()

    available_dates = pd.Index(subset["date"].unique())
    missing_dates = sorted(set(scores["signal_date"].unique()) - set(available_dates))
    if missing_dates:
        first = pd.Timestamp(missing_dates[0]).date()
        last = pd.Timestamp(missing_dates[-1]).date()
        raise ValueError(
            f"Component file does not cover {len(missing_dates)} prediction dates "
            f"for {index_key}: {first} to {last}"
        )

    merged = scores.merge(
        subset,
        left_on=["signal_date", "code"],
        right_on=["date", "code"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="date")
    if merged.empty:
        raise ValueError(f"No predictions matched the {index_key} component universe")
    merged["prediction_rank"] = merged.groupby("signal_date")["prediction_score"].rank(
        method="first", ascending=False
    )
    merged["universe_size"] = merged.groupby("signal_date")["code"].transform("size")
    return merged.sort_values(["signal_date", "prediction_rank", "code"]).reset_index(drop=True)


def build_all_index_inputs(
    prediction_path: str | Path,
    component_path: str | Path,
    output_root: str | Path,
    index_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    """Create one local prediction file per configured index."""
    predictions = pd.read_csv(prediction_path)
    components = load_components(component_path)
    output_root = Path(output_root)
    outputs: dict[str, Path] = {}
    manifest: dict[str, Any] = {
        "prediction_source": str(Path(prediction_path).resolve()),
        "component_source": str(Path(component_path).resolve()),
        "indexes": {},
    }
    for index_key, spec in index_specs.items():
        print(f"[IndexEnhancement] build index={index_key}", flush=True)
        frame = build_index_input(predictions, components, index_key)
        output_dir = output_root / index_key
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "prediction_score.csv.gz"
        frame.to_csv(output, index=False, compression="gzip")
        outputs[index_key] = output
        manifest["indexes"][index_key] = {
            "order_book_id": spec["order_book_id"],
            "name": spec.get("name", index_key),
            "output": str(output.resolve()),
            "rows": int(len(frame)),
            "dates": int(frame["signal_date"].nunique()),
            "min_universe_size": int(frame["universe_size"].min()),
            "max_universe_size": int(frame["universe_size"].max()),
        }
        print(
            f"[IndexEnhancement] saved index={index_key} rows={len(frame):,} "
            f"dates={frame['signal_date'].nunique()} file={output.resolve()}",
            flush=True,
        )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地成分股文件生成三套指数增强预测输入")
    parser.add_argument("--config", default="configs/index_enhancement.yaml")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--components", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    component_config = config.get("component_data", {})
    build_config = config.get("build", {})
    build_all_index_inputs(
        prediction_path=args.predictions
        or build_config.get("predictions", "results/lightgbm/prediction_score.csv"),
        component_path=args.components
        or component_config.get("file", "data/index_components.csv.gz"),
        output_root=args.output_root
        or build_config.get("output_root", "results/index_enhancement"),
        index_specs=config["indexes"],
    )


if __name__ == "__main__":
    main()
