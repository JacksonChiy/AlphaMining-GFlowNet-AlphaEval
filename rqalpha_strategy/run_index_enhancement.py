"""Run three isolated RQAlphaPlus backtests from persisted index-universe files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def build_backtest_command(
    index_key: str,
    spec: dict,
    prediction_path: Path,
    output_dir: Path,
    base_config: str | Path,
    bundle: str | Path,
    portfolio_mode: str | None = None,
    index_weights: str | Path | None = None,
    optimizer: dict | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rqalpha_strategy.run_backtest",
        "--config",
        str(base_config),
        "--bundle",
        str(bundle),
        "--predictions",
        str(prediction_path),
        "--output-dir",
        str(output_dir),
        "--benchmark",
        str(spec["order_book_id"]),
        "--top-n",
        str(int(spec["top_n"])),
        "--hold-buffer-rank",
        str(int(spec["hold_buffer_rank"])),
    ]
    if portfolio_mode:
        command.extend(["--portfolio-mode", str(portfolio_mode)])
    if index_weights:
        command.extend(
            ["--index-weights", str(index_weights), "--index-key", str(index_key)]
        )
    if spec.get("optimizer_max_names") is not None:
        command.extend(["--optimizer-max-names", str(int(spec["optimizer_max_names"]))])
    flags = {
        "alpha_strength": "--alpha-strength",
        "risk_aversion": "--risk-aversion",
        "turnover_penalty": "--turnover-penalty",
        "max_active_weight": "--max-active-weight",
        "max_stock_weight": "--max-stock-weight",
        "max_rebalance_turnover": "--max-rebalance-turnover",
        "estimated_buy_cost": "--estimated-buy-cost",
        "estimated_sell_cost": "--estimated-sell-cost",
    }
    for key, flag in flags.items():
        if optimizer and optimizer.get(key) is not None:
            command.extend([flag, str(optimizer[key])])
    return command


def run_all(
    config_path: str | Path,
    selected_indexes: list[str] | None = None,
    dry_run: bool = False,
) -> list[list[str]]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    specs = config["indexes"]
    selected = selected_indexes or list(specs)
    unknown = sorted(set(selected).difference(specs))
    if unknown:
        raise ValueError(f"Unknown index keys: {unknown}; available={sorted(specs)}")
    build = config.get("build", {})
    backtest = config.get("backtest", {})
    prediction_root = Path(build.get("output_root", "results/index_enhancement"))
    output_root = Path(
        backtest.get("output_root", "results/index_enhancement_backtest")
    )
    commands = []
    for index_key in selected:
        prediction_dir = Path(specs[index_key].get("prediction_dir", prediction_root / index_key))
        # A freshly trained index-specific model writes plain CSV.  Prefer it
        # over the legacy compressed file produced by the full-market filter.
        candidates = [
            prediction_dir / "prediction_score.csv",
            prediction_dir / "prediction_score.csv.gz",
        ]
        prediction_path = next((path for path in candidates if path.exists()), candidates[0])
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Index prediction file not found: {prediction_path.resolve()}. "
                "Run python -m src.index_enhancement.builder first."
            )
        output_dir = output_root / index_key
        command = build_backtest_command(
            index_key=index_key,
            spec=specs[index_key],
            prediction_path=prediction_path,
            output_dir=output_dir,
            base_config=backtest.get("base_config", "configs/training_config.yaml"),
            bundle=backtest.get("bundle", "~/.rqalpha-plus/bundle"),
            portfolio_mode=backtest.get("portfolio_mode"),
            index_weights=backtest.get("index_weights"),
            optimizer=backtest.get("optimizer", {}),
        )
        commands.append(command)
        print(f"[IndexBacktest] index={index_key} command={' '.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, check=True)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description="分别运行沪深300、中证500和中证1000增强回测")
    parser.add_argument("--config", default="configs/index_enhancement.yaml")
    parser.add_argument(
        "--indexes",
        default="csi300,csi500,csi1000",
        help="Comma-separated subset: csi300,csi500,csi1000",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = [item.strip() for item in args.indexes.split(",") if item.strip()]
    run_all(args.config, selected_indexes=selected, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
