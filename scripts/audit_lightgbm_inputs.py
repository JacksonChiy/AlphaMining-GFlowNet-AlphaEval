from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gflownet.reward import make_forward_return


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit daily labels and minute factor variation before LightGBM"
    )
    parser.add_argument(
        "--price", default="results/minute_ppu_ddb_ram/daily_price.pkl"
    )
    parser.add_argument(
        "--factors", default="results/minute_ppu_ddb_ram/alpha_factor_matrix.pkl"
    )
    parser.add_argument(
        "--evaluation", default="results/minute_ppu_ddb_ram/alpha_eval_result.csv"
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--output", default="results/minute_ppu_ddb_ram/lightgbm_input_audit.csv"
    )
    args = parser.parse_args()

    price = _read_frame(Path(args.price))
    factors = _read_frame(Path(args.factors))
    evaluation_path = Path(args.evaluation)
    keys = ["date", "code"]
    for frame, label in ((price, "price"), (factors, "factors")):
        missing = sorted(set(keys).difference(frame.columns))
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["code"] = frame["code"].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} contains duplicate date/code rows")

    all_factors = [column for column in factors.columns if column not in keys]
    selected = all_factors
    if evaluation_path.exists():
        evaluation = pd.read_csv(evaluation_path)
        if "dpp_selected" not in evaluation or "factor" not in evaluation:
            raise ValueError("evaluation must contain factor and dpp_selected")
        selected = evaluation.loc[
            evaluation["dpp_selected"].astype(bool), "factor"
        ].astype(str).tolist()
        missing = sorted(set(selected).difference(all_factors))
        if missing:
            raise ValueError(f"Selected factors missing from matrix: {missing}")
    if not selected:
        raise ValueError("No selected factors to audit")

    base = price[keys + ["close"]].copy()
    base["target"] = make_forward_return(price, args.horizon).to_numpy()
    data = base[keys + ["target"]].merge(
        factors[keys + selected], on=keys, how="inner", validate="one_to_one"
    )
    rows = []
    for year, work in data.groupby(data["date"].dt.year, observed=True, sort=True):
        target_grouped = work.groupby("date", observed=True)["target"]
        target_count = target_grouped.count()
        target_unique = target_grouped.nunique(dropna=True)
        factor_values = work[selected].to_numpy(dtype=float, copy=False)
        factor_std = work.groupby("date", observed=True)[selected].std(ddof=0)
        active = factor_std.gt(1e-12).sum(axis=1)
        rows.append({
            "year": int(year),
            "rows": int(len(work)),
            "dates": int(work["date"].nunique()),
            "stocks": int(work["code"].nunique()),
            "selected_factors": int(len(selected)),
            "factor_finite_ratio": float(np.isfinite(factor_values).mean()),
            "active_factors_min": int(active.min()) if len(active) else 0,
            "active_factors_median": float(active.median()) if len(active) else 0.0,
            "zero_active_factor_dates": int(active.eq(0).sum()),
            "mature_label_dates": int(target_count.gt(0).sum()),
            "target_varying_dates": int(target_unique.gt(1).sum()),
            "constant_target_dates": int(
                (target_count.ge(2) & target_unique.le(1)).sum()
            ),
            "last_mature_label_date": (
                str(target_count.index[target_count.gt(0)].max().date())
                if target_count.gt(0).any() else ""
            ),
        })
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"[LightGBMAudit] selected_factors={len(selected)} output={output}")
    print(result.to_string(index=False))
    problems = result.loc[
        result["zero_active_factor_dates"].gt(0)
        | result["target_varying_dates"].eq(0)
    ]
    if len(problems):
        print(
            "[LightGBMAudit] WARNING problematic_years="
            + json.dumps(problems["year"].tolist()),
            flush=True,
        )


if __name__ == "__main__":
    main()
