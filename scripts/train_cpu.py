from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


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
    args = parser.parse_args()
    config_path = args.config or (
        "configs/minute_training_cpu.yaml"
        if args.mode == "minute"
        else "configs/training_cpu.yaml"
    )
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
