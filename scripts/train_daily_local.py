from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable daily pipeline locally on CPU"
    )
    parser.add_argument("--config", default="configs/daily/local.yaml")
    parser.add_argument(
        "--from-stage",
        choices=("prepare", "gflownet", "alpha_eval", "lightgbm", "backtest"),
        default="prepare",
    )
    parser.add_argument(
        "--to-stage",
        choices=("prepare", "gflownet", "alpha_eval", "lightgbm", "backtest"),
        default=None,
    )
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--reuse-prepared-data", action="store_true")
    parser.add_argument("--reuse-alpha-pool", action="store_true")
    parser.add_argument("--rqalpha-bundle", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    # Freeze BLAS and PyTorch CPU settings before importing numerical pipeline modules.
    from scripts.train_cpu import configure_environment

    runtime = configure_environment(config_path, args.threads)
    from src.runtime_logging import build_training_log_path, tee_console_output

    log_dir = args.log_dir or config.get("outputs", {}).get(
        "log_dir", "results/daily_local/logs"
    )
    log_path = build_training_log_path(
        "daily_local", log_dir=log_dir, log_file=args.log_file
    )
    exit_code: int | None = None
    with tee_console_output(log_path) as active_log_path:
        try:
            import torch

            torch.set_num_threads(runtime["torch_threads"])
            try:
                torch.set_num_interop_threads(runtime["interop_threads"])
            except RuntimeError:
                pass
            print(
                f"[DailyLocal] runtime_start logical_cpus={runtime['logical_cpus']} "
                f"torch_threads={torch.get_num_threads()} "
                f"interop_threads={torch.get_num_interop_threads()} "
                f"blas_threads={runtime['blas_threads']} cuda_visible=False "
                f"config={config_path} log={active_log_path}",
                flush=True,
            )
            if torch.cuda.is_available():
                raise RuntimeError("Local daily launcher failed to hide CUDA")
            from src.pipeline.daily_local import run_daily_local

            to_stage = args.to_stage or (
                "backtest" if args.rqalpha_bundle else "lightgbm"
            )
            run_daily_local(
                config_path,
                from_stage=args.from_stage,
                to_stage=to_stage,
                pool_size=args.pool_size,
                reuse_prepared_data=args.reuse_prepared_data,
                reuse_alpha_pool=args.reuse_alpha_pool,
                rqalpha_bundle=args.rqalpha_bundle,
            )
        except KeyboardInterrupt:
            exit_code = 130
            traceback.print_exc(file=sys.stderr)
        except BaseException:
            exit_code = 1
            traceback.print_exc(file=sys.stderr)
        finally:
            print(
                f"[DailyLocal] runtime_end status="
                f"{'completed' if exit_code is None else 'failed'} "
                f"finished_at={datetime.now().astimezone().isoformat()} "
                f"log={active_log_path}",
                flush=True,
            )
    if exit_code is not None:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
