"""Shared pytest fixtures: synthetic, deterministic price/return data.

Tests use synthetic data (fixed RNG seed) rather than live yfinance downloads
so the suite is fast, deterministic, and runs offline / in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_daily_returns() -> pd.DataFrame:
    """~3 years of daily returns for 4 assets with a known covariance structure."""
    rng = np.random.default_rng(42)
    n_days = 750
    tickers = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D"]

    true_cov = np.array([
        [0.00040, 0.00010, 0.00005, 0.00000],
        [0.00010, 0.00030, 0.00002, 0.00001],
        [0.00005, 0.00002, 0.00020, 0.00000],
        [0.00000, 0.00001, 0.00000, 0.00010],
    ])
    mean = np.array([0.00040, 0.00030, 0.00020, 0.00010])

    data = rng.multivariate_normal(mean, true_cov, size=n_days)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(data, columns=tickers, index=index)


@pytest.fixture
def synthetic_prices(synthetic_daily_returns: pd.DataFrame) -> pd.DataFrame:
    return 100 * (1 + synthetic_daily_returns).cumprod()
