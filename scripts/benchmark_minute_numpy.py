from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.expression.minute import minute_expression_from_tokens
from src.gflownet.numpy_minute_executor import NumpyMinuteBlockExecutor


REPRESENTATIVE_EXPRESSIONS = (
    ("r_mean", "m_ma", "W20", "close"),
    ("r_std", "m_rank", "close"),
    ("r_corr", "close", "vol"),
    ("r_skew", "close"),
    ("r_kurt", "close"),
    ("r_slope", "close"),
    ("r_mean", "m_top", "W20", "close"),
    ("r_wmean", "close", "vol"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="分钟NumPy执行器本机性能基准")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--minutes", type=int, default=241)
    parser.add_argument("--stocks", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.days, args.minutes, args.stocks, args.repeats) < 1:
        raise ValueError("days/minutes/stocks/repeats must be positive")

    rng = np.random.default_rng(42)
    shape = (args.days, args.minutes, args.stocks)
    mask = np.ones(shape, dtype=bool)
    mask[:, :, ::23] = False
    close = rng.lognormal(2.3, 0.02, shape).astype(np.float32)
    volume = rng.lognormal(7.0, 1.0, shape).astype(np.float32)
    nodes = [
        minute_expression_from_tokens(tokens).block_nodes()[0]
        for tokens in REPRESENTATIVE_EXPRESSIONS
    ]
    durations = []
    for repeat in range(1, args.repeats + 1):
        started = time.perf_counter()
        NumpyMinuteBlockExecutor(mask, {"close": close, "vol": volume}).execute(nodes)
        duration = time.perf_counter() - started
        durations.append(duration)
        print(
            f"[MinuteNumpyBenchmark] repeat={repeat}/{args.repeats} "
            f"seconds={duration:.4f}",
            flush=True,
        )
    median = float(np.median(durations))
    elements = int(np.prod(shape)) * len(nodes)
    print(
        f"[MinuteNumpyBenchmark] shape={shape} blocks={len(nodes)} "
        f"median_seconds={median:.4f} million_block_elements_per_second="
        f"{elements / max(median, 1e-12) / 1_000_000:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
