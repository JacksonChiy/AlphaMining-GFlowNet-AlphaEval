from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_environment(config_path: str | Path, threads_override: int | None = None) -> dict[str, int]:
    """Configure CPU libraries before importing NumPy, pandas or PyTorch."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    runtime = config.get("cpu_runtime", {})
    logical_cpus = os.cpu_count() or 1
    torch_threads = int(threads_override or runtime.get("torch_threads", max(1, logical_cpus // 2)))
    interop_threads = int(runtime.get("interop_threads", min(2, torch_threads)))
    blas_threads = int(runtime.get("blas_threads", 1))
    if min(torch_threads, interop_threads, blas_threads) < 1:
        raise ValueError("All CPU thread counts must be positive")

    # Keep concurrent reward workers from each spawning a full BLAS thread pool.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    return {
        "logical_cpus": logical_cpus,
        "torch_threads": torch_threads,
        "interop_threads": interop_threads,
        "blas_threads": blas_threads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only GFlowNet training launcher")
    parser.add_argument("--mode", choices=("daily", "minute"), default="minute")
    parser.add_argument("--config", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Log directory; defaults to outputs.log_dir or results/logs",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Explicit log file path; overrides --log-dir and outputs.log_dir",
    )
    args = parser.parse_args()
    config_path = args.config or (
        "configs/minute_training_cpu.yaml"
        if args.mode == "minute"
        else "configs/training_cpu.yaml"
    )
    with Path(config_path).open("r", encoding="utf-8") as stream:
        launcher_config = yaml.safe_load(stream) or {}
    configured_log_dir = (
        launcher_config.get("outputs", {}).get("log_dir", "results/logs")
    )

    # This module intentionally has no NumPy/Pandas/PyTorch imports. The CPU
    # environment must be frozen before those libraries are loaded.
    from src.runtime_logging import build_training_log_path, tee_console_output

    log_path = build_training_log_path(
        args.mode,
        log_dir=args.log_dir or configured_log_dir,
        log_file=args.log_file,
    )
    exit_code: int | None = None
    with tee_console_output(log_path) as active_log_path:
        print(
            f"[CPUTraining] log_start file={active_log_path} "
            f"started_at={datetime.now().astimezone().isoformat()}",
            flush=True,
        )
        try:
            _run_training(args, config_path)
        except KeyboardInterrupt:
            exit_code = 130
            print(
                f"[CPUTraining] log_end status=interrupted file={active_log_path}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
        except BaseException as error:
            exit_code = 1
            print(
                f"[CPUTraining] log_end status=failed "
                f"exception={type(error).__name__} file={active_log_path}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
        else:
            print(
                f"[CPUTraining] log_end status=completed file={active_log_path} "
                f"finished_at={datetime.now().astimezone().isoformat()}",
                flush=True,
            )
    if exit_code is not None:
        raise SystemExit(exit_code)


def _run_training(args: argparse.Namespace, config_path: str) -> None:
    report = configure_environment(config_path, args.threads)

    # Imports deliberately happen after the CPU environment is frozen.
    import torch

    torch.set_num_threads(report["torch_threads"])
    try:
        torch.set_num_interop_threads(report["interop_threads"])
    except RuntimeError:
        # PyTorch only allows this setting before parallel work starts.
        pass
    print(
        "[CPUTraining] runtime_start "
        f"mode={args.mode} logical_cpus={report['logical_cpus']} "
        f"torch_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} "
        f"blas_threads={report['blas_threads']} cuda_visible=False "
        f"config={config_path}",
        flush=True,
    )
    if torch.cuda.is_available():
        raise RuntimeError("CPU launcher failed to hide CUDA; aborting to preserve reproducibility")

    if args.mode == "minute":
        from src.gflownet.run_minute_training import run
    else:
        from src.gflownet.run_training import run
    run(
        config_path,
        require_a100=False,
        pool_size=args.pool_size,
        device="cpu",
    )


if __name__ == "__main__":
    main()
