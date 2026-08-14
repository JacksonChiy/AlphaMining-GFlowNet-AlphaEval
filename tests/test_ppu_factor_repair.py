from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.repair_ppu_oos_factors import (
    merge_repaired_factor_range,
    validate_repaired_range,
)


def _matrix(dates: list[str], value_offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.to_datetime(dates)):
        for code_index, code in enumerate(("A", "B", "C")):
            rows.append({
                "date": date,
                "code": code,
                "minute_factor_001": value_offset + date_index + code_index,
                "minute_factor_002": value_offset + date_index - code_index,
            })
    return pd.DataFrame(rows)


def test_merge_repaired_factor_range_replaces_only_requested_dates() -> None:
    existing = _matrix(["2025-12-31", "2026-01-05", "2026-01-06"])
    existing.loc[existing["date"].dt.year.eq(2026), [
        "minute_factor_001", "minute_factor_002"
    ]] = np.nan
    repaired = _matrix(["2025-12-01", "2026-01-05", "2026-01-06"], 100.0)
    factors = ["minute_factor_001", "minute_factor_002"]

    result = merge_repaired_factor_range(
        existing, repaired, factors, "2026-01-01", "2026-01-31"
    )

    old = result.loc[result["date"].eq(pd.Timestamp("2025-12-31"))]
    new = result.loc[result["date"].dt.year.eq(2026)]
    assert old["minute_factor_001"].max() < 10
    assert new["minute_factor_001"].min() >= 100
    assert not result.duplicated(["date", "code"]).any()


def test_validate_repaired_range_rejects_all_nan_dates() -> None:
    frame = _matrix(["2026-01-05", "2026-01-06"])
    frame.loc[frame["date"].eq(pd.Timestamp("2026-01-06")), [
        "minute_factor_001", "minute_factor_002"
    ]] = np.nan
    with pytest.raises(ValueError, match="no cross-sectional variation"):
        validate_repaired_range(
            frame,
            ["minute_factor_001", "minute_factor_002"],
            "2026-01-01",
            "2026-01-31",
            0.5,
        )


def test_validate_repaired_range_reports_healthy_matrix() -> None:
    frame = _matrix(["2026-01-05", "2026-01-06"])
    report = validate_repaired_range(
        frame,
        ["minute_factor_001", "minute_factor_002"],
        "2026-01-01",
        "2026-01-31",
        0.8,
    )
    assert report["factor_finite_ratio"] == 1.0
    assert report["zero_active_factor_dates"] == 0
