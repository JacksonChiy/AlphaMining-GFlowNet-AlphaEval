from __future__ import annotations

import pandas as pd


def validate_frame_covers_period(
    frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    *,
    date_column: str = "date",
    label: str = "data",
    boundary_tolerance_days: int = 7,
) -> dict[str, str]:
    if date_column not in frame:
        raise ValueError(f"{label} is missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"{label} contains no valid dates")
    expected_start = pd.Timestamp(start_date)
    expected_end = pd.Timestamp(end_date)
    actual_start = dates.min()
    actual_end = dates.max()
    tolerance = pd.Timedelta(days=boundary_tolerance_days)
    if actual_start > expected_start + tolerance:
        raise ValueError(
            f"{label} starts at {actual_start.date()}, later than configured "
            f"training start {expected_start.date()}; rebuild daily_price.pkl"
        )
    if actual_end < expected_end - tolerance:
        raise ValueError(
            f"{label} ends at {actual_end.date()}, earlier than configured "
            f"training end {expected_end.date()}; rebuild daily_price.pkl"
        )
    return {"actual_start": str(actual_start.date()), "actual_end": str(actual_end.date())}


def validate_research_date_split(config: dict[str, object]) -> dict[str, str]:
    """Validate the train/OOS boundary shared by mining, fusion, and backtesting."""
    dataset = config.get("dataset")
    lightgbm = config.get("lightgbm")
    if not isinstance(dataset, dict) or not isinstance(lightgbm, dict):
        raise ValueError("Config must contain dataset and lightgbm mappings")
    required_dataset = (
        "start_date",
        "end_date",
        "mining_start_date",
        "mining_end_date",
        "out_of_sample_start_date",
        "out_of_sample_end_date",
    )
    missing = [key for key in required_dataset if dataset.get(key) is None]
    if missing:
        raise ValueError(f"Dataset date split is incomplete: {missing}")
    required_lightgbm = ("prediction_start_date", "prediction_end_date")
    missing_lightgbm = [key for key in required_lightgbm if lightgbm.get(key) is None]
    if missing_lightgbm:
        raise ValueError(f"LightGBM prediction date split is incomplete: {missing_lightgbm}")

    parsed = {key: pd.Timestamp(dataset[key]).normalize() for key in required_dataset}
    prediction_start = pd.Timestamp(lightgbm.get("prediction_start_date")).normalize()
    prediction_end = pd.Timestamp(lightgbm.get("prediction_end_date")).normalize()
    if parsed["start_date"] > parsed["mining_start_date"]:
        raise ValueError("Dataset start_date must not be later than mining_start_date")
    if parsed["mining_start_date"] > parsed["mining_end_date"]:
        raise ValueError("mining_start_date must not be later than mining_end_date")
    if parsed["mining_end_date"] >= parsed["out_of_sample_start_date"]:
        raise ValueError("Training and out-of-sample periods must not overlap")
    if parsed["out_of_sample_start_date"] > parsed["out_of_sample_end_date"]:
        raise ValueError("out_of_sample_start_date must not be later than its end date")
    if parsed["end_date"] < parsed["out_of_sample_end_date"]:
        raise ValueError("Dataset end_date must cover the out-of-sample end date")
    if prediction_start != parsed["out_of_sample_start_date"]:
        raise ValueError("LightGBM prediction start must equal out-of-sample start")
    if prediction_end != parsed["out_of_sample_end_date"]:
        raise ValueError("LightGBM prediction end must equal out-of-sample end")
    return {
        "training": (
            f"{parsed['mining_start_date'].date()}..{parsed['mining_end_date'].date()}"
        ),
        "out_of_sample": (
            f"{parsed['out_of_sample_start_date'].date()}.."
            f"{parsed['out_of_sample_end_date'].date()}"
        ),
    }


def slice_date_range(
    frame: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    date_column: str = "date",
    label: str = "data",
) -> pd.DataFrame:
    """Return an inclusive date slice while preserving rows needed before the slice elsewhere."""
    if date_column not in frame:
        raise ValueError(f"{label} is missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    mask = dates.notna()
    if start_date is not None:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= dates <= pd.Timestamp(end_date)
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError(
            f"{label} has no rows in configured range "
            f"[{start_date or '-inf'}, {end_date or '+inf'}]"
        )
    return result
