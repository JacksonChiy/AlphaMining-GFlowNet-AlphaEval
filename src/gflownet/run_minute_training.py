from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data_loader.dolphindb_minute import (
    DolphinDBMinuteLoader,
    MinuteDolphinDBConfig,
    create_dolphindb_session,
    load_minute_cache,
    prepare_dolphindb_minute_data,
)
from src.data_loader.minute_memmap import (
    DolphinDBMinuteRAMStore,
    MinuteMemMapConfig,
    MinuteMemMapStore,
)
from src.gflownet.minute_factor_pool import (
    save_minute_alpha_pool,
    save_minute_alpha_pool_from_cache,
    save_minute_alpha_pool_from_dolphindb_stream,
    save_minute_alpha_pool_from_memmap,
)
from src.gflownet.minute_grammar import MinuteVocabulary
from src.gflownet.minute_reward import (
    DolphinDBStreamingMinuteRewardEvaluator,
    MinuteRewardEvaluator,
)
from src.gflownet.memmap_reward import MemMapMinuteRewardEvaluator
from src.gflownet.minute_trainer import MinuteGFlowNetTrainer
from src.gflownet.model import GFlowNetPolicy, PolicyConfig
from src.gflownet.run_training import gpu_report
from src.gflownet.trainer import TrainerConfig
from src.utils import create_experiment, load_config, seed_everything, slice_date_range


def _load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Minute training input does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported minute input format: {suffix}; use pkl, parquet or csv")


def run(
    config_path: str,
    require_a100: bool = True,
    pool_size: int | None = None,
    device: str | torch.device | None = None,
) -> Path:
    config = load_config(config_path)
    dataset = config["dataset"]
    pool_size = int(pool_size or config.get("pipeline", {}).get("pool_size", 100))
    if pool_size <= 0:
        raise ValueError("pipeline.pool_size must be positive")
    hardware = gpu_report(require_a100)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"[MinuteGFlowNet] hardware={hardware} training_device={target_device}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(int(config["training"]["seed"]))
    experiment_dir = create_experiment(config_path)
    print(f"[MinuteGFlowNet] experiment_id={experiment_dir.name}", flush=True)

    ddb_cache: Path | None = None
    ddb_loader: DolphinDBMinuteLoader | None = None
    memmap_store: MinuteMemMapStore | None = None
    ddb_session = None
    minute_data: pd.DataFrame | None
    try:
        if str(dataset.get("source", "local")).lower() == "dolphindb":
            values = dataset.get("dolphindb", {})
            load_mode = str(values.get("load_mode", "cache")).lower()
            if load_mode == "memmap":
                memmap_config = MinuteMemMapConfig.from_mapping(
                    dataset.get("memmap", {})
                )
                memmap_store = MinuteMemMapStore(memmap_config)
                daily_data = pd.read_csv(memmap_store.daily_file)
                minute_data = None
                print(
                    f"[MinuteGFlowNet] memmap_enabled remote_ddb_queries_during_training=0 "
                    f"root={memmap_config.root} dates={len(memmap_store.dates):,} "
                    f"stocks={len(memmap_store.stocks):,}",
                    flush=True,
                )
            elif load_mode == "ram":
                ddb_config = MinuteDolphinDBConfig.from_mapping(dataset, values)
                ddb_session = create_dolphindb_session(values)
                ddb_loader = DolphinDBMinuteLoader(ddb_config, ddb_session)
                memory_values = dict(dataset.get("memory", {}))
                memory_values.setdefault("root", "results/minute_ram_metadata")
                memory_values.setdefault(
                    "block_cache_dir", "results/minute_ram_block_cache"
                )
                memory_values.setdefault("reward_parallel_backend", "threading")
                memory_config = MinuteMemMapConfig.from_mapping(memory_values)

                def ram_loader_factory() -> DolphinDBMinuteLoader:
                    return DolphinDBMinuteLoader(
                        ddb_config, create_dolphindb_session(values)
                    )

                memmap_store = DolphinDBMinuteRAMStore(
                    ddb_loader,
                    memory_config,
                    loader_factory=(
                        ram_loader_factory if memory_config.build_workers > 1 else None
                    ),
                    max_ram_gb=float(memory_values.get("max_ram_gb", 0.0)),
                    reserve_ram_gb=float(memory_values.get("reserve_ram_gb", 64.0)),
                )
                daily_data = memmap_store.daily_data
                minute_data = None
                ddb_session.close()
                ddb_session = None
                ddb_loader = None
                print(
                    f"[MinuteGFlowNet] ddb_ram_enabled raw_minute_files=false "
                    f"root={memory_config.root} dates={len(memmap_store.dates):,} "
                    f"stocks={len(memmap_store.stocks):,} "
                    "remote_ddb_queries_during_training=0",
                    flush=True,
                )
            else:
                ddb_config = MinuteDolphinDBConfig.from_mapping(dataset, values)
            if load_mode == "stream":
                ddb_session = create_dolphindb_session(values)
                ddb_loader = DolphinDBMinuteLoader(ddb_config, ddb_session)
                audit = ddb_loader.audit()
                if not audit.passed:
                    raise ValueError(
                        "DolphinDB stream field audit failed: confirm adjusted OHLC and set "
                        "dataset.dolphindb.prices_are_adjusted=true"
                    )
                print(
                    f"[MinuteGFlowNet] ddb_stream_enabled raw_minute_files=false "
                    f"source_range={audit.source_min_date}..{audit.source_max_date} "
                    f"rows={audit.source_rows:,}",
                    flush=True,
                )
                daily_data = ddb_loader.build_daily_in_memory(
                    ddb_config.start_date, ddb_config.end_date
                )
                minute_data = None
            elif load_mode == "cache":
                ddb_cache, daily_path = prepare_dolphindb_minute_data(dataset)
                minute_data = load_minute_cache(
                    ddb_cache,
                    dataset.get("mining_start_date"),
                    dataset.get("mining_end_date"),
                )
                daily_data = _load_frame(daily_path)
        else:
            minute_data = _load_frame(dataset["minute_file"])
            daily_data = _load_frame(dataset["daily_file"])
        return _run_loaded_pipeline(
            config=config,
            dataset=dataset,
            pool_size=pool_size,
            target_device=target_device,
            experiment_dir=experiment_dir,
            minute_data=minute_data,
            daily_data=daily_data,
            ddb_cache=ddb_cache,
            ddb_loader=ddb_loader,
            memmap_store=memmap_store,
        )
    finally:
        if ddb_session is not None:
            ddb_session.close()


def _run_loaded_pipeline(
    config: dict,
    dataset: dict,
    pool_size: int,
    target_device: torch.device,
    experiment_dir: Path,
    minute_data: pd.DataFrame | None,
    daily_data: pd.DataFrame,
    ddb_cache: Path | None,
    ddb_loader: DolphinDBMinuteLoader | None,
    memmap_store: MinuteMemMapStore | None,
) -> Path:
    frames = [(daily_data, "daily")]
    if minute_data is not None:
        frames.insert(0, (minute_data, "minute"))
    for frame, label in frames:
        if not {"date", "code"}.issubset(frame.columns):
            raise ValueError(f"{label} input must contain date and code")
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["code"] = frame["code"].astype(str)
    mining_start = dataset.get("mining_start_date")
    mining_end = dataset.get("mining_end_date")
    daily_mining = slice_date_range(
        daily_data, mining_start, mining_end, label="daily reward data"
    )
    minute_mining = None
    if minute_data is not None:
        minute_mining = slice_date_range(
            minute_data, mining_start, mining_end, label="minute mining data"
        )
        print(
            f"[MinuteGFlowNet] minute_data rows={len(minute_mining):,} "
            f"dates={minute_mining['date'].nunique():,} stocks={minute_mining['code'].nunique():,} "
            f"start={minute_mining['date'].min()} end={minute_mining['date'].max()}",
            flush=True,
        )
    print(
        f"[MinuteGFlowNet] daily_reward_data rows={len(daily_mining):,} "
        f"dates={daily_mining['date'].nunique():,} stocks={daily_mining['code'].nunique():,}",
        flush=True,
    )
    reward_options = dict(config["reward"])
    block_cache_max_entries = int(reward_options.pop("block_cache_max_entries", 256))
    if memmap_store is not None:
        evaluator = MemMapMinuteRewardEvaluator(
            memmap_store,
            daily_mining,
            start_date=str(mining_start),
            end_date=str(mining_end),
            block_cache_max_entries=block_cache_max_entries,
            **reward_options,
        )
    elif ddb_loader is not None:
        evaluator = DolphinDBStreamingMinuteRewardEvaluator(
            ddb_loader,
            daily_mining,
            start_date=str(mining_start),
            end_date=str(mining_end),
            block_cache_max_entries=block_cache_max_entries,
            **reward_options,
        )
    else:
        if minute_mining is None:
            raise AssertionError("Local/cache minute mode requires minute data")
        evaluator = MinuteRewardEvaluator(minute_mining, daily_mining, **reward_options)
    policy_values = dict(config["model"])
    policy_values.pop("name", None)
    model = GFlowNetPolicy(PolicyConfig(**policy_values), MinuteVocabulary())
    print(
        f"[MinuteGFlowNet] model_parameters={sum(p.numel() for p in model.parameters()):,} "
        f"actions={len(model.vocabulary.action_tokens)}",
        flush=True,
    )
    training_values = dict(config["training"])
    training_values.pop("seed", None)
    trainer = MinuteGFlowNetTrainer(
        model,
        evaluator,
        TrainerConfig(seed=int(config["training"]["seed"]), **training_values),
        device=target_device,
    )
    checkpoint = Path(config.get("outputs", {}).get("checkpoint", "checkpoints/gflownet_minute_best.pt"))
    metrics = trainer.train(checkpoint)
    outputs = config.get("outputs", {})
    metrics_path = Path(outputs.get("metrics", "results/minute_gflownet_training_metrics.csv"))
    trajectory_path = Path(outputs.get("trajectory_metrics", "results/minute_gflownet_trajectory_metrics.csv"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(trainer.trajectory_history).to_csv(trajectory_path, index=False)
    metrics.to_csv(experiment_dir / "model_metrics.csv", index=False)

    loaded = MinuteGFlowNetTrainer.load_checkpoint(
        checkpoint, evaluator, device=target_device
    )
    print(f"[MinuteGFlowNet] alpha_pool_generation_start target_size={pool_size}", flush=True)
    pool_attempts = int(config.get("pipeline", {}).get("pool_attempts", 2000))
    pool = loaded.generate_pool(size=pool_size, attempts=pool_attempts)
    save_arguments = {
        "metadata_path": outputs.get("alpha_pool", "results/minute_alpha_pool.csv"),
        "matrix_path": outputs.get("factor_matrix", "results/minute_alpha_factor_matrix.pkl"),
        "min_coverage": evaluator.min_coverage,
    }
    if memmap_store is not None:
        ddb_values = dataset["dolphindb"]
        metadata, matrix = save_minute_alpha_pool_from_memmap(
            pool,
            memmap_store,
            daily_data,
            start_date=str(ddb_values.get("start_date", dataset.get("mining_start_date"))),
            end_date=str(ddb_values.get("end_date", dataset.get("out_of_sample_end_date"))),
            **save_arguments,
        )
    elif ddb_loader is not None:
        ddb_values = dataset["dolphindb"]
        metadata, matrix = save_minute_alpha_pool_from_dolphindb_stream(
            pool,
            ddb_loader,
            daily_data,
            start_date=str(ddb_values.get("start_date", dataset.get("mining_start_date"))),
            end_date=str(ddb_values.get("end_date", dataset.get("out_of_sample_end_date"))),
            **save_arguments,
        )
    elif ddb_cache is not None:
        metadata, matrix = save_minute_alpha_pool_from_cache(
            pool, ddb_cache, daily_data, **save_arguments
        )
    else:
        if minute_data is None:
            raise AssertionError("Local minute mode requires minute data")
        metadata, matrix = save_minute_alpha_pool(
            pool, minute_data, daily_data, **save_arguments
        )
    metadata.to_csv(experiment_dir / "factor_results.csv", index=False)
    print(
        f"[MinuteGFlowNet] complete factors={len(metadata)} matrix_rows={len(matrix):,} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the report chart-27~30 minute grammar")
    parser.add_argument("--config", default="configs/minute_training_config.yaml")
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only training")
    parser.add_argument("--pool-size", type=int, default=None)
    args = parser.parse_args()
    run(
        args.config,
        require_a100=not (args.allow_non_a100 or args.cpu),
        pool_size=args.pool_size,
        device="cpu" if args.cpu else None,
    )


if __name__ == "__main__":
    main()
