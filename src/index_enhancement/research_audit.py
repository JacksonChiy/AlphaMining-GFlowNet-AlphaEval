"""P0/P1 baseline freeze and strategy-aligned research audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .diagnostics import evaluate_backtest, evaluate_backtest_periods, freeze_baseline_manifest
from .evaluation import evaluate_prediction_quality, save_prediction_quality


def _prediction_path(root: Path, key: str) -> Path:
    candidates = [root / key / "prediction_score.csv", root / key / "prediction_score.csv.gz"]
    return next((path for path in candidates if path.exists()), candidates[0])


def run_audit(config_path: str | Path, baseline_id: str) -> Path:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    experiment = Path("experiments") / baseline_id
    diagnostics = experiment / "diagnostics"
    experiment.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, experiment / "index_enhancement_config.yaml")
    prediction_root = Path(config["build"]["output_root"])
    backtest_root = Path(config["backtest"]["output_root"])
    label_root = Path(config["labels"]["output_root"])
    frozen: list[Path] = [Path(config_path)]
    comparison, annuals, monthlys, turnovers, cash_rows = [], [], [], [], []
    quality_frames: dict[str, list[pd.DataFrame]] = {}
    for key, spec in config["indexes"].items():
        prediction = _prediction_path(prediction_root, key)
        label = label_root / key / "labels.pkl"
        report = backtest_root / key
        frozen.extend([
            prediction,
            prediction.parent / "lgbm_model.joblib",
            prediction.parent / "model_metrics.csv",
            prediction.parent / "feature_importance.csv",
            report / "backtest_effective_config.json",
            report / "portfolio.csv",
            report / "trades.csv",
        ])
        frozen.extend(sorted(prediction.parent.glob("lgbm_window_*.joblib")))
        comparison.append(evaluate_backtest(report, key))
        annual, monthly, turnover, cash = evaluate_backtest_periods(report, key)
        annuals.append(annual); monthlys.append(monthly)
        turnovers.append(turnover); cash_rows.append(cash)
        if label.exists():
            result = evaluate_prediction_quality(
                pd.read_csv(prediction), pd.read_pickle(label), key, int(spec["top_n"])
            )
            save_prediction_quality(result, diagnostics / key)
            for name, frame in result.items():
                quality_frames.setdefault(name, []).append(frame)
        else:
            print(f"[Audit] labels missing, skip signal audit: {label}", flush=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(diagnostics / "backtest_comparison.csv", index=False)
    pd.concat(annuals).to_csv(diagnostics / "annual_performance.csv", index=False)
    pd.concat(monthlys).to_csv(diagnostics / "monthly_performance.csv", index=False)
    pd.concat(turnovers).to_csv(diagnostics / "turnover_cost.csv", index=False)
    pd.concat(cash_rows).to_csv(diagnostics / "cash_drag.csv", index=False)
    for name, frames in quality_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(diagnostics / f"all_{name}.csv", index=False)
    manifest = freeze_baseline_manifest(
        [path for path in frozen if path.exists()],
        experiment / "baseline_manifest.json",
        metadata={
            "baseline_id": baseline_id,
            "created_at": datetime.now().astimezone().isoformat(),
            "training_period": "2020-01-01..2023-12-31",
            "oos_period": "2024-01-01..2026-12-31",
            "target": "index excess close(t+5)/close(t+1)-1",
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "git_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
            ).stdout.strip()),
        },
    )
    (experiment / "audit_summary.json").write_text(
        json.dumps({"manifest_files": len(manifest["files"]), "indexes": list(config["indexes"])},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Audit] complete: {experiment.resolve()}", flush=True)
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结P0基线并生成P1研究诊断")
    parser.add_argument("--config", default="configs/index_enhancement.yaml")
    parser.add_argument("--baseline-id", default="baseline_v2_index_excess_l2_equal_weight")
    args = parser.parse_args()
    run_audit(args.config, args.baseline_id)


if __name__ == "__main__":
    main()
