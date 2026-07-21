from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=100)
    rows = []
    for stock_index in range(30):
        code = f"{stock_index + 1:06d}.XSHE"
        returns = rng.normal(0.0005 + stock_index * 0.00001, 0.015, len(dates))
        close = 10 * np.exp(np.cumsum(returns))
        for index, date in enumerate(dates):
            rows.append({
                "date": date,
                "code": code,
                "open": close[index] * (1 + rng.normal(0, 0.002)),
                "high": close[index] * 1.01,
                "low": close[index] * 0.99,
                "close": close[index],
                "volume": float(rng.integers(100_000, 2_000_000)),
                "vwap": close[index] * (1 + rng.normal(0, 0.001)),
                "industry": f"I{stock_index % 5}",
                "market_cap": float((stock_index + 1) * 1e8),
            })
    return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)

