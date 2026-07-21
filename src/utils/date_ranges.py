from __future__ import annotations

import pandas as pd


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
