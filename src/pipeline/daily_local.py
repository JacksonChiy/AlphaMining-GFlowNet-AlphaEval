from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.alpha_eval import AlphaEval, AlphaEvalConfig
from src.data_loader import prepare_price_csv
from src.gflownet import execute_saved_alpha_pool
from src.gflownet.run_training import run as run_gflownet
from src.model import LightGBMConfig, LightGBMFusion
from src.operators import configure_time_series_from_mapping
from src.utils import load_config, slice_date_range, validate_research_date_split


DAILY_LOCAL_STAGES = ("prepare", "gflownet", "alpha_eval", "lightgbm", "backtest")


@dataclass(frozen=True)
class DailyLocalPaths:
    prepared_price: Path
    data_quality_report: Path
    checkpoint: Path
    training_metrics: Path
    trajectory_metrics: Path
    alpha_pool: Path
    factor_matrix: Path
    oos_factor_matrix: Path
    alpha_eval: Path
    lightgbm_dir: Path
    backtest_dir: Path
    manifest: Path

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DailyLocalPaths":
        outputs = config.get("outputs", {})
        dataset = config["dataset"]
        return cls(
            prepared_price=Path(dataset.get("output", "data/daily_price.pkl")),
            data_quality_report=Path(
                outputs.get("data_quality_report", "results/daily_local/data_quality_report.json")
            ),
            checkpoint=Path(
                outputs.get("checkpoint", "checkpoints/gflownet_daily_local_best.pt")
            ),
            training_metrics=Path(
                outputs.get("metrics", "results/daily_local/gflownet_training_metrics.csv")
            ),
            trajectory_metrics=Path(
                outputs.get(
                    "trajectory_metrics",
                    "results/daily_local/gflownet_trajectory_metrics.csv",
                )
            ),
            alpha_pool=Path(outputs.get("alpha_pool", "results/daily_local/alpha_pool.csv")),
            factor_matrix=Path(
                outputs.get("factor_matrix", "results/daily_local/alpha_factor_matrix.pkl")
            ),
            oos_factor_matrix=Path(
                outputs.get(
                    "oos_factor_matrix",
                    "results/daily_local/alpha_factor_matrix_oos.pkl",
                )
            ),
            alpha_eval=Path(
                outputs.get("alpha_eval", "results/daily_local/alpha_eval_result.csv")
            ),
            lightgbm_dir=Path(
                outputs.get("lightgbm_dir", "results/daily_local/lightgbm")
            ),
            backtest_dir=Path(
                outputs.get("backtest_dir", "results/daily_local/backtest_report")
            ),
            manifest=Path(
                outputs.get("pipeline_manifest", "results/daily_local/pipeline_manifest.json")
            ),
        )


def build_stage_sequence(
    from_stage: str = "prepare", to_stage: str = "lightgbm"
) -> tuple[str, ...]:
    if from_stage not in DAILY_LOCAL_STAGES:
        raise ValueError(f"Unknown local daily start stage: {from_stage}")
    if to_stage not in DAILY_LOCAL_STAGES:
        raise ValueError(f"Unknown local daily end stage: {to_stage}")
    start = DAILY_LOCAL_STAGES.index(from_stage)
    end = DAILY_LOCAL_STAGES.index(to_stage)
    if start > end:
        raise ValueError("from_stage must not be later than to_stage")
    return DAILY_LOCAL_STAGES[start : end + 1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_files(stage: str, paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Local daily stage '{stage}' requires missing artifacts: {missing}. "
            "Run from an earlier stage or check outputs in the YAML config."
        )


def _prepare_data(
    config: dict[str, Any], paths: DailyLocalPaths, reuse_prepared_data: bool
) -> dict[str, Any]:
    if reuse_prepared_data:
        _require_files("prepare", [paths.prepared_price])
        data = pd.read_pickle(paths.prepared_price)
        print(
            f"[DailyLocal] prepare_reused path={paths.prepared_price} rows={len(data):,}",
            flush=True,
        )
    else:
        dataset = config["dataset"]
        filter_keys = (
            "start_date",
            "end_date",
            "max_stocks",
            "universe_start_date",
            "universe_end_date",
            "chunksize",
        )
        filters = {
            key: dataset[key] for key in filter_keys if dataset.get(key) is not None
        }
        data = prepare_price_csv(
            dataset["file"],
            paths.prepared_price,
            paths.data_quality_report,
            **filters,
        )
    return {
        "rows": len(data),
        "dates": int(data["date"].nunique()),
        "stocks": int(data["code"].nunique()),
        "output": str(paths.prepared_price),
        "reused": reuse_prepared_data,
    }


def _run_gflownet_stage(
    config_path: str | Path,
    config: dict[str, Any],
    paths: DailyLocalPaths,
    pool_size: int | None,
    reuse_alpha_pool: bool,
) -> dict[str, Any]:
    if reuse_alpha_pool:
        _require_files("gflownet", [paths.prepared_price, paths.alpha_pool])
        configure_time_series_from_mapping(config.get("operators"))
        data = pd.read_pickle(paths.prepared_price)
        matrix, oos = execute_saved_alpha_pool(
            data,
            metadata_path=paths.alpha_pool,
            matrix_path=paths.factor_matrix,
            oos_matrix_path=paths.oos_factor_matrix,
            oos_start_date=config["dataset"].get("out_of_sample_start_date"),
            oos_end_date=config["dataset"].get("out_of_sample_end_date"),
        )
        return {
            "mode": "reuse_alpha_pool",
            "factors": len(pd.read_csv(paths.alpha_pool)),
            "factor_rows": len(matrix),
            "oos_rows": len(oos) if oos is not None else 0,
        }

    _require_files("gflownet", [paths.prepared_price])
    experiment_dir = run_gflownet(
        str(config_path), require_a100=False, pool_size=pool_size, device="cpu"
    )
    return {
        "mode": "train",
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(paths.checkpoint),
        "alpha_pool": str(paths.alpha_pool),
    }


def _run_alpha_eval_stage(
    config: dict[str, Any], paths: DailyLocalPaths
) -> dict[str, Any]:
    _require_files(
        "alpha_eval", [paths.prepared_price, paths.factor_matrix, paths.alpha_pool]
    )
    price = pd.read_pickle(paths.prepared_price)
    factors = pd.read_pickle(paths.factor_matrix)
    mining_price = slice_date_range(
        price,
        config["dataset"].get("mining_start_date"),
        config["dataset"].get("mining_end_date"),
        label="Local AlphaEval price data",
    )
    mining_factors = slice_date_range(
        factors,
        config["dataset"].get("mining_start_date"),
        config["dataset"].get("mining_end_date"),
        label="Local AlphaEval factor data",
    )
    values = dict(config["alpha_eval"])
    values["horizon"] = int(config["dataset"]["horizon"])
    metadata = pd.read_csv(paths.alpha_pool)
    result = AlphaEval(
        mining_price, mining_factors, AlphaEvalConfig(**values)
    ).evaluate(metadata, paths.alpha_eval)
    selected = result.loc[result["dpp_selected"].astype(bool), "factor"].tolist()
    if not selected:
        raise ValueError("AlphaEval selected no factors; inspect factor coverage and scores")
    return {
        "evaluated_factors": len(result),
        "selected_factors": len(selected),
        "output": str(paths.alpha_eval),
    }


def _run_lightgbm_stage(
    config: dict[str, Any], paths: DailyLocalPaths
) -> dict[str, Any]:
    _require_files(
        "lightgbm", [paths.prepared_price, paths.factor_matrix, paths.alpha_eval]
    )
    price = pd.read_pickle(paths.prepared_price)
    factors = pd.read_pickle(paths.factor_matrix)
    evaluation = pd.read_csv(paths.alpha_eval)
    selected = evaluation.loc[
        evaluation["dpp_selected"].astype(bool), "factor"
    ].tolist()
    if not selected:
        raise ValueError("No DPP-selected factors are available for local LightGBM")
    prediction = LightGBMFusion(
        LightGBMConfig(**dict(config["lightgbm"]))
    ).fit_predict(price, factors, selected, paths.lightgbm_dir)
    return {
        "selected_factors": len(selected),
        "prediction_rows": len(prediction),
        "prediction_dates": int(prediction["signal_date"].nunique()),
        "output": str(paths.lightgbm_dir / "prediction_score.csv"),
    }


def _run_backtest_stage(
    config_path: str | Path,
    paths: DailyLocalPaths,
    rqalpha_bundle: str | Path | None,
) -> dict[str, Any]:
    if rqalpha_bundle is None:
        raise ValueError("The backtest stage requires --rqalpha-bundle")
    prediction_path = paths.lightgbm_dir / "prediction_score.csv"
    _require_files("backtest", [prediction_path])
    from rqalpha_strategy.run_backtest import load_backtest_settings, run

    settings = load_backtest_settings(config_path)
    arguments = Namespace(
        predictions=str(prediction_path),
        bundle=str(rqalpha_bundle),
        output_dir=str(paths.backtest_dir),
        **settings,
    )
    run(arguments)
    return {
        "output": str(paths.backtest_dir),
        "effective_config": str(paths.backtest_dir / "backtest_effective_config.json"),
    }


def run_daily_local(
    config_path: str | Path = "configs/daily/local.yaml",
    *,
    from_stage: str = "prepare",
    to_stage: str = "lightgbm",
    pool_size: int | None = None,
    reuse_prepared_data: bool = False,
    reuse_alpha_pool: bool = False,
    rqalpha_bundle: str | Path | None = None,
) -> dict[str, Any]:
    """Run a resumable daily CPU pipeline entirely on the local machine."""
    config_path = Path(config_path)
    config = load_config(config_path)
    date_split = validate_research_date_split(config)
    stages = build_stage_sequence(from_stage, to_stage)
    if "backtest" in stages and rqalpha_bundle is None:
        raise ValueError("The backtest stage requires --rqalpha-bundle")
    paths = DailyLocalPaths.from_config(config)
    manifest: dict[str, Any] = {
        "pipeline": "daily_local_cpu",
        "config": str(config_path),
        "device": "cpu",
        "date_split": date_split,
        "requested_stages": list(stages),
        "started_at_utc": _utc_now(),
        "status": "running",
        "paths": {key: str(value) for key, value in asdict(paths).items()},
        "stages": {},
    }
    _write_manifest(paths.manifest, manifest)
    print(
        f"[DailyLocal] pipeline_start stages={','.join(stages)} "
        f"config={config_path} manifest={paths.manifest}",
        flush=True,
    )
    runners = {
        "prepare": lambda: _prepare_data(config, paths, reuse_prepared_data),
        "gflownet": lambda: _run_gflownet_stage(
            config_path, config, paths, pool_size, reuse_alpha_pool
        ),
        "alpha_eval": lambda: _run_alpha_eval_stage(config, paths),
        "lightgbm": lambda: _run_lightgbm_stage(config, paths),
        "backtest": lambda: _run_backtest_stage(
            config_path, paths, rqalpha_bundle
        ),
    }
    try:
        for index, stage in enumerate(stages, start=1):
            started = datetime.now(timezone.utc)
            manifest["stages"][stage] = {
                "status": "running",
                "started_at_utc": started.isoformat(),
            }
            _write_manifest(paths.manifest, manifest)
            print(
                f"[DailyLocal] stage_start index={index}/{len(stages)} stage={stage}",
                flush=True,
            )
            details = runners[stage]()
            finished = datetime.now(timezone.utc)
            manifest["stages"][stage] = {
                "status": "completed",
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "seconds": round((finished - started).total_seconds(), 3),
                "details": details,
            }
            _write_manifest(paths.manifest, manifest)
            print(
                f"[DailyLocal] stage_complete stage={stage} "
                f"seconds={(finished - started).total_seconds():.1f}",
                flush=True,
            )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = _utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_manifest(paths.manifest, manifest)
        raise
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = _utc_now()
    _write_manifest(paths.manifest, manifest)
    print(f"[DailyLocal] pipeline_complete manifest={paths.manifest}", flush=True)
    return manifest
