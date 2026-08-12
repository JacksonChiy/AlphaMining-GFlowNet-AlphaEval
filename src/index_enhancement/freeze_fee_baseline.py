"""Freeze fee-consistent experiment outputs with fingerprints and diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .diagnostics import (
    evaluate_backtest,
    evaluate_backtest_periods,
    freeze_baseline_manifest,
)


def _fee_audit(trades: pd.DataFrame) -> dict:
    quantity = pd.to_numeric(trades["last_quantity"], errors="coerce")
    price = pd.to_numeric(trades["last_price"], errors="coerce")
    notional = quantity * price
    commission = pd.to_numeric(trades["commission"], errors="coerce")
    tax = pd.to_numeric(trades["tax"], errors="coerce")
    commission_rates = (commission / notional).loc[(commission > 5.0) & (notional > 0)]
    tax_rates = (tax / notional).loc[(tax > 0) & (notional > 0)]
    return {
        "commission_rate_median": float(commission_rates.median()),
        "commission_rate_p10": float(commission_rates.quantile(0.10)),
        "commission_rate_p90": float(commission_rates.quantile(0.90)),
        "sell_tax_rate_median": float(tax_rates.median()),
        "total_commission": float(commission.sum()),
        "total_tax": float(tax.sum()),
        "total_transaction_cost": float(
            pd.to_numeric(trades["transaction_cost"], errors="coerce").sum()
        ),
        "traded_notional": float(notional.sum()),
    }


def freeze_fee_baseline(config_path: str | Path) -> Path:
    source_config = Path(config_path)
    with source_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, output / "config.yaml")
    report_root = output / "backtest_report"
    expected_commission = float(config["fee_assumptions"]["stock_commission_rate"])
    expected_tax = float(config["fee_assumptions"]["stamp_duty_rate"])
    summaries, annuals, turnovers, fee_rows, frozen = [], [], [], [], [source_config]

    for experiment, root_value in config["experiments"].items():
        root = Path(root_value)
        for index_key in config["indexes"]:
            backtest = root / "index_enhancement_backtest" / index_key
            prediction = root / "index_enhancement" / index_key
            required = [
                backtest / "portfolio.csv",
                backtest / "trades.csv",
                prediction / "prediction_score.csv",
                prediction / "model_metrics.csv",
                prediction / "lgbm_model.joblib",
            ]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Cannot freeze experiment={experiment} index={index_key}; missing={missing}"
                )
            summary = evaluate_backtest(backtest, index_key)
            summary["experiment"] = experiment
            summaries.append(summary)
            annual, _, turnover, _ = evaluate_backtest_periods(backtest, index_key)
            annual.insert(0, "experiment", experiment)
            turnover.insert(0, "experiment", experiment)
            annuals.append(annual)
            turnovers.append(turnover)
            trades = pd.read_csv(backtest / "trades.csv")
            audit = _fee_audit(trades)
            audit.update(experiment=experiment, index_key=index_key)
            fee_rows.append(audit)
            if not np.isclose(
                audit["commission_rate_median"], expected_commission, rtol=0, atol=1e-10
            ):
                raise ValueError(
                    f"Commission mismatch {experiment}/{index_key}: "
                    f"{audit['commission_rate_median']} != {expected_commission}"
                )
            if not np.isclose(
                audit["sell_tax_rate_median"], expected_tax, rtol=0, atol=1e-10
            ):
                raise ValueError(
                    f"Tax mismatch {experiment}/{index_key}: "
                    f"{audit['sell_tax_rate_median']} != {expected_tax}"
                )
            source_files = required + [
                backtest / "summary.xlsx",
                backtest / "equity_curve.png",
                backtest / "backtest_effective_config.json",
                root / "colab_training_config.yaml",
                root / "alpha_pool.csv",
            ]
            frozen.extend(path for path in source_files if path.exists())
            destination = report_root / experiment / index_key
            destination.mkdir(parents=True, exist_ok=True)
            for name in (
                "summary.xlsx",
                "equity_curve.png",
                "backtest_effective_config.json",
                "rqalpha_effective_config.json",
            ):
                source = backtest / name
                if source.exists():
                    shutil.copy2(source, destination / name)

    pd.DataFrame(summaries).to_csv(output / "performance_summary.csv", index=False)
    pd.concat(annuals, ignore_index=True).to_csv(
        output / "annual_performance.csv", index=False
    )
    pd.concat(turnovers, ignore_index=True).to_csv(
        output / "turnover_cost.csv", index=False
    )
    fee_frame = pd.DataFrame(fee_rows)
    fee_frame.to_csv(output / "fee_audit.csv", index=False)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    git_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    metadata = {
        "baseline_id": config["baseline_id"],
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "fee_assumptions": config["fee_assumptions"],
        "experiments": config["experiments"],
        "indexes": config["indexes"],
        "fee_audit_passed": True,
    }
    manifest = freeze_baseline_manifest(
        list(dict.fromkeys(path for path in frozen if path.exists())),
        output / "baseline_manifest.json",
        metadata=metadata,
    )
    (output / "experiment_manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "manifest_files": len(manifest["files"]),
                "performance_rows": len(summaries),
                "fee_audit_rows": len(fee_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[FeeBaseline] complete output={output.resolve()} "
        f"experiments={len(config['experiments'])} fee_rows={len(fee_rows)}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结统一万8手续费实验基线")
    parser.add_argument("--config", default="configs/baselines/fee_0008.yaml")
    args = parser.parse_args()
    freeze_fee_baseline(args.config)


if __name__ == "__main__":
    main()
