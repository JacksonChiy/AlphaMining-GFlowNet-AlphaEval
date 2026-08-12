"""Point-in-time labels for index-enhancement research.

The signal is observed after the close on date ``t``.  Its forward return is
the close-to-close return from ``t+1`` to ``t+horizon``.  Entry/exit
tradability therefore belongs to the label only; it must never be exposed as
a feature available on signal date ``t``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .universe import load_components, normalize_order_book_id
from .weights import load_index_weights


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _normalise_price(price: pd.DataFrame) -> pd.DataFrame:
    """Return the raw fields needed by label construction without imputing them."""
    code_column = _first_present(price.columns, ("code", "order_book_id"))
    amount_column = _first_present(price.columns, ("amount", "total_turnover"))
    required = {"date", "close", "volume"}
    missing = required.difference(price.columns)
    if code_column is None:
        missing.add("code/order_book_id")
    if missing:
        raise ValueError(f"Price data missing label fields: {sorted(missing)}")

    columns = ["date", code_column, "close", "volume"]
    for optional in (amount_column, "open", "high", "low", "limit_up", "limit_down"):
        if optional is not None and optional in price.columns and optional not in columns:
            columns.append(optional)
    result = price.loc[:, columns].copy()
    result = result.rename(columns={code_column: "code"})
    if amount_column is not None:
        result = result.rename(columns={amount_column: "amount"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["code"] = result["code"].map(normalize_order_book_id)
    numeric = [name for name in result.columns if name not in {"date", "code"}]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["date", "code"])
    if result.duplicated(["date", "code"]).any():
        raise ValueError("Price data contains duplicate date/code rows")
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _tradability_on_date(frame: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Compute close-auction tradability from raw daily price-limit fields."""
    active = frame["close"].gt(0) & frame["volume"].gt(0)
    if "amount" in frame:
        active &= frame["amount"].gt(0)

    limit_up = frame.get("limit_up", pd.Series(np.nan, index=frame.index))
    limit_down = frame.get("limit_down", pd.Series(np.nan, index=frame.index))
    limit_available = limit_up.gt(0) & limit_down.gt(0)
    at_limit_up = limit_available & frame["close"].ge(limit_up * (1.0 - tolerance))
    at_limit_down = limit_available & frame["close"].le(limit_down * (1.0 + tolerance))

    # One-price flags are useful diagnostics, while the stricter close-limit
    # flags match this pipeline's close-price entry/exit label convention.
    if {"open", "high", "low"}.issubset(frame.columns):
        scale = frame["close"].abs().clip(lower=1.0)
        one_price = (
            frame["high"].sub(frame["low"]).abs().le(scale * tolerance)
            & frame["open"].sub(frame["close"]).abs().le(scale * tolerance)
        )
    else:
        one_price = pd.Series(False, index=frame.index)

    return pd.DataFrame(
        {
            "suspended": ~active,
            "limit_data_available": limit_available,
            "at_limit_up": at_limit_up,
            "at_limit_down": at_limit_down,
            "one_price_limit_up": one_price & at_limit_up,
            "one_price_limit_down": one_price & at_limit_down,
            "buyable": active & ~at_limit_up,
            "sellable": active & ~at_limit_down,
        },
        index=frame.index,
    )


def build_forward_labels(
    price: pd.DataFrame,
    horizon: int = 5,
    *,
    price_limit_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Build leakage-safe ``t+5/t+1`` return and tradability labels.

    Dates are shifted on the global trading calendar and then joined by code.
    Consequently, a missing stock row on an intended entry/exit date produces
    a missing label instead of silently shifting to the stock's next quote.
    If the processed pickle lacks ``limit_up``/``limit_down``, suspension is
    still enforced and ``limit_data_available`` is false; no price-limit rule
    is guessed from board codes or rounded returns.
    """
    if horizon <= 1:
        raise ValueError("horizon must be greater than 1 for a t+1/t+horizon label")
    if price_limit_tolerance < 0:
        raise ValueError("price_limit_tolerance must be non-negative")

    daily = _normalise_price(price)
    flags = _tradability_on_date(daily, price_limit_tolerance)
    daily = pd.concat([daily, flags], axis=1)
    dates = pd.Index(daily["date"].drop_duplicates().sort_values())
    calendar = pd.DataFrame({"date": dates})
    calendar["entry_date"] = pd.Series(dates, index=calendar.index).shift(-1)
    calendar["exit_date"] = pd.Series(dates, index=calendar.index).shift(-horizon)

    result = daily[["date", "code"]].merge(calendar, on="date", how="left")
    entry_columns = {
        "date": "entry_date",
        "close": "entry_close",
        "suspended": "entry_suspended",
        "limit_data_available": "entry_limit_data_available",
        "at_limit_up": "entry_at_limit_up",
        "one_price_limit_up": "entry_one_price_limit_up",
        "buyable": "entry_buyable",
    }
    exit_columns = {
        "date": "exit_date",
        "close": "exit_close",
        "suspended": "exit_suspended",
        "limit_data_available": "exit_limit_data_available",
        "at_limit_down": "exit_at_limit_down",
        "one_price_limit_down": "exit_one_price_limit_down",
        "sellable": "exit_sellable",
    }
    entry = daily[["date", "code", *[name for name in entry_columns if name != "date"]]].rename(
        columns=entry_columns
    )
    exit_ = daily[["date", "code", *[name for name in exit_columns if name != "date"]]].rename(
        columns=exit_columns
    )
    result = result.merge(entry, on=["entry_date", "code"], how="left", validate="many_to_one")
    result = result.merge(exit_, on=["exit_date", "code"], how="left", validate="many_to_one")
    result["target_raw_return"] = result["exit_close"] / result["entry_close"] - 1.0
    result["tradable"] = result["entry_buyable"].eq(True) & result[
        "exit_sellable"
    ].eq(True)
    result.loc[~np.isfinite(result["target_raw_return"]), "target_raw_return"] = np.nan
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _normalise_weights(weights: pd.DataFrame) -> pd.DataFrame:
    code_column = _first_present(weights.columns, ("code", "order_book_id"))
    weight_column = _first_present(weights.columns, ("benchmark_weight", "weight"))
    missing = {"date"}.difference(weights.columns)
    if code_column is None:
        missing.add("code/order_book_id")
    if weight_column is None:
        missing.add("benchmark_weight/weight")
    if missing:
        raise ValueError(f"Index weights missing fields: {sorted(missing)}")
    keep = ["date", code_column, weight_column]
    for optional in ("index_key", "index_code"):
        if optional in weights:
            keep.append(optional)
    result = weights.loc[:, keep].rename(
        columns={code_column: "code", weight_column: "benchmark_weight"}
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["code"] = result["code"].map(normalize_order_book_id)
    result["benchmark_weight"] = pd.to_numeric(
        result["benchmark_weight"], errors="coerce"
    )
    result = result.dropna(subset=["date", "code", "benchmark_weight"])
    if (result["benchmark_weight"] < 0).any():
        raise ValueError("Index weights must be non-negative")
    return result


def build_index_labels(
    price: pd.DataFrame,
    components: pd.DataFrame,
    weights: pd.DataFrame,
    index_key: str,
    horizon: int = 5,
    *,
    min_weight_coverage: float = 0.95,
    tradable_only: bool = True,
    price_limit_tolerance: float = 1e-6,
    forward_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create raw, benchmark-excess and cross-sectional-rank index labels."""
    if not 0 < min_weight_coverage <= 1:
        raise ValueError("min_weight_coverage must be in (0, 1]")
    required_components = {"date", "index_key", "code"}
    missing = required_components.difference(components.columns)
    if missing:
        raise ValueError(f"Components missing fields: {sorted(missing)}")

    members = components.loc[
        components["index_key"] == index_key, ["date", "code"]
    ].copy()
    members["date"] = pd.to_datetime(members["date"], errors="coerce").dt.normalize()
    members["code"] = members["code"].map(normalize_order_book_id)
    members = members.drop_duplicates(["date", "code"])
    if members.empty:
        raise ValueError(f"No components found for index_key={index_key!r}")

    weight_data = _normalise_weights(weights)
    if "index_key" in weight_data:
        weight_data = weight_data[weight_data["index_key"] == index_key]
    weighted_members = members.merge(
        weight_data.drop(columns=["index_key", "index_code"], errors="ignore"),
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    total_weight = weighted_members.groupby("date", observed=True)[
        "benchmark_weight"
    ].transform("sum")
    weighted_members["benchmark_weight"] = weighted_members["benchmark_weight"] / total_weight

    labels = forward_labels
    if labels is None:
        labels = build_forward_labels(
            price, horizon=horizon, price_limit_tolerance=price_limit_tolerance
        )
    result = weighted_members.merge(labels, on=["date", "code"], how="left", validate="one_to_one")
    valid_return_weight = result["benchmark_weight"].where(
        result["target_raw_return"].notna(), 0.0
    )
    result["weight_coverage"] = valid_return_weight.groupby(
        result["date"], observed=True
    ).transform("sum")
    weighted_return = (result["benchmark_weight"] * result["target_raw_return"]).where(
        result["target_raw_return"].notna(), 0.0
    )
    result["benchmark_return"] = weighted_return.groupby(
        result["date"], observed=True
    ).transform("sum") / result["weight_coverage"].replace(0.0, np.nan)
    result.loc[result["weight_coverage"] < min_weight_coverage, "benchmark_return"] = np.nan
    result["target_excess_return"] = (
        result["target_raw_return"] - result["benchmark_return"]
    )

    eligible = result["target_excess_return"].notna()
    if tradable_only:
        eligible &= result["tradable"].eq(True)
    result["target_cross_sectional_rank"] = np.nan
    result.loc[eligible, "target_cross_sectional_rank"] = result.loc[eligible].groupby(
        "date", observed=True
    )["target_excess_return"].rank(method="average", pct=True)
    if tradable_only:
        result.loc[~eligible, ["target_raw_return", "target_excess_return"]] = np.nan
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def load_price_for_components(
    path: str | Path,
    component_codes: Iterable[str],
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Read raw CSV fields for relevant stocks without loading the 700MB file at once."""
    source = Path(path)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    code_column = _first_present(header, ("code", "order_book_id"))
    if code_column is None:
        raise ValueError("Price CSV must contain code or order_book_id")
    amount_column = _first_present(header, ("amount", "total_turnover"))
    usecols = [code_column, "date", "close", "volume"]
    for column in (amount_column, "open", "high", "low", "limit_up", "limit_down"):
        if column and column in header and column not in usecols:
            usecols.append(column)
    wanted = {normalize_order_book_id(value) for value in component_codes}
    parts: list[pd.DataFrame] = []
    rows_read = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(source, usecols=usecols, chunksize=chunksize), start=1
    ):
        rows_read += len(chunk)
        normalized = chunk[code_column].map(normalize_order_book_id)
        selected = chunk.loc[normalized.isin(wanted)].copy()
        if not selected.empty:
            selected[code_column] = normalized.loc[selected.index]
            parts.append(selected)
        print(
            f"[Labels] price_chunk={chunk_index:03d} rows_read={rows_read:,} "
            f"rows_kept={sum(len(part) for part in parts):,}",
            flush=True,
        )
    if not parts:
        raise ValueError("No price rows matched the configured index constituents")
    return pd.concat(parts, ignore_index=True)


def build_all_index_labels(
    price_path: str | Path,
    component_path: str | Path,
    weight_path: str | Path,
    output_root: str | Path,
    index_keys: Iterable[str],
    *,
    horizon: int = 5,
    min_weight_coverage: float = 0.95,
    tradable_only: bool = True,
) -> dict[str, Path]:
    """Build the three local label datasets; no external data API is used."""
    components = load_components(component_path)
    weights = load_index_weights(weight_path)
    price = load_price_for_components(price_path, components["code"].unique())
    print(f"[Labels] build_forward_labels rows={len(price):,} horizon={horizon}", flush=True)
    forward = build_forward_labels(price, horizon=horizon)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    manifest: dict[str, object] = {
        "label_definition": f"close(t+{horizon}) / close(t+1) - 1",
        "price_source": str(Path(price_path).resolve()),
        "component_source": str(Path(component_path).resolve()),
        "weight_source": str(Path(weight_path).resolve()),
        "min_weight_coverage": min_weight_coverage,
        "tradability_policy": "drop" if tradable_only else "keep",
        "indexes": {},
    }
    for index_key in index_keys:
        print(f"[Labels] index_start index={index_key}", flush=True)
        frame = build_index_labels(
            price,
            components,
            weights,
            index_key,
            horizon=horizon,
            min_weight_coverage=min_weight_coverage,
            tradable_only=tradable_only,
            forward_labels=forward,
        )
        directory = output_root / index_key
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "labels.pkl"
        frame.to_pickle(output)
        outputs[index_key] = output
        manifest["indexes"][index_key] = {
            "output": str(output.resolve()),
            "rows": int(len(frame)),
            "dates": int(frame["date"].nunique()),
            "valid_targets": int(frame["target_excess_return"].notna().sum()),
            "tradable_ratio": float(frame["tradable"].eq(True).mean()),
            "mean_weight_coverage": float(frame.groupby("date")["weight_coverage"].first().mean()),
        }
        print(
            f"[Labels] index_done index={index_key} rows={len(frame):,} "
            f"valid={frame['target_excess_return'].notna().sum():,} file={output.resolve()}",
            flush=True,
        )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地行情、成分和权重生成指数增强标签")
    parser.add_argument("--config", default="configs/index_enhancement/default.yaml")
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    label_config = config.get("labels", {})
    build_all_index_labels(
        label_config.get("price_file", "data/price.csv"),
        config.get("component_data", {}).get("file", "data/index_components.csv.gz"),
        config.get("weight_data", {}).get("file", "data/index_weights.csv.gz"),
        label_config.get("output_root", "results/index_enhancement_labels"),
        config["indexes"].keys(),
        horizon=int(label_config.get("horizon", 5)),
        min_weight_coverage=float(label_config.get("min_weight_coverage", 0.95)),
        tradable_only=label_config.get("tradability_policy", "drop") == "drop",
    )


if __name__ == "__main__":
    main()
