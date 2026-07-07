"""Historical stress-window analysis and bootstrapped Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk_engine.config import (
    MONTE_CARLO_HORIZONS_YEARS,
    MONTE_CARLO_PATHS,
    STRESS_WINDOWS,
    TRADING_DAYS_PER_YEAR,
)
from portfolio_risk_engine.risk.metrics import max_drawdown


def portfolio_daily_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Static (buy-and-hold-of-daily-rebalance) weighted portfolio return series."""
    aligned_weights = weights.reindex(asset_returns.columns).fillna(0.0)
    return asset_returns.mul(aligned_weights, axis=1).sum(axis=1)


@dataclass
class StressResult:
    window_name: str
    start: str
    end: str
    total_return: float
    max_drawdown: float
    n_trading_days: int
    excluded_assets: tuple[str, ...] = ()


def run_stress_tests(
    raw_asset_returns: pd.DataFrame,
    weights: pd.Series,
    windows: dict[str, tuple[str, str]] | None = None,
) -> list[StressResult]:
    """Replay the portfolio through named historical stress windows.

    `raw_asset_returns` should come from an *extended-history* price panel
    (see `config.STRESS_TEST_START_DATE`) that has NOT been jointly aligned
    across assets -- individual columns may contain leading NaNs for tickers
    that didn't exist yet. For each window we keep only the assets with
    complete data across that specific window, rescale the target weights
    to sum to 1 over those assets, and report which (if any) were excluded.
    Windows with no assets available, or falling entirely outside the
    available price history, are skipped.
    """
    windows = windows or STRESS_WINDOWS

    results = []
    for name, (start, end) in windows.items():
        window_returns = raw_asset_returns.loc[start:end]
        if window_returns.empty:
            continue

        target_assets = [t for t in weights.index if t in window_returns.columns]
        available = [t for t in target_assets if window_returns[t].notna().all()]
        excluded = tuple(t for t in target_assets if t not in available)
        if not available:
            continue

        w = weights.reindex(available).astype(float)
        w = w / w.sum()

        port_returns = window_returns[available].mul(w, axis=1).sum(axis=1)
        cumulative = (1 + port_returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1)
        dd, _, _ = max_drawdown(cumulative)
        results.append(
            StressResult(
                window_name=name,
                start=str(port_returns.index.min().date()),
                end=str(port_returns.index.max().date()),
                total_return=total_return,
                max_drawdown=dd,
                n_trading_days=len(port_returns),
                excluded_assets=excluded,
            )
        )
    return results


@dataclass
class MonteCarloResult:
    horizon_years: int
    paths: np.ndarray  # shape (n_paths,) cumulative total return per path
    mean: float
    median: float
    std: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    prob_of_loss: float


def run_monte_carlo(
    portfolio_returns: pd.Series,
    horizons_years: tuple[int, ...] = MONTE_CARLO_HORIZONS_YEARS,
    n_paths: int = MONTE_CARLO_PATHS,
    block_size: int = 20,
    random_seed: int | None = 42,
) -> dict[int, MonteCarloResult]:
    """Bootstrap simulation of cumulative portfolio return distributions.

    Uses a block (stationary) bootstrap -- resampling contiguous chunks of
    `block_size` trading days rather than single days -- so that volatility
    clustering and short-horizon autocorrelation in the historical return
    series are partially preserved, rather than assuming i.i.d. daily draws.
    """
    rng = np.random.default_rng(random_seed)
    returns = portfolio_returns.dropna().values
    n_obs = len(returns)
    results: dict[int, MonteCarloResult] = {}

    for horizon in horizons_years:
        n_days = horizon * TRADING_DAYS_PER_YEAR
        n_blocks = int(np.ceil(n_days / block_size))
        outcomes = np.empty(n_paths)

        for i in range(n_paths):
            start_idxs = rng.integers(0, n_obs - block_size, size=n_blocks)
            path = np.concatenate([returns[s : s + block_size] for s in start_idxs])[:n_days]
            outcomes[i] = np.prod(1 + path) - 1

        results[horizon] = MonteCarloResult(
            horizon_years=horizon,
            paths=outcomes,
            mean=float(np.mean(outcomes)),
            median=float(np.median(outcomes)),
            std=float(np.std(outcomes, ddof=1)),
            percentile_5=float(np.percentile(outcomes, 5)),
            percentile_25=float(np.percentile(outcomes, 25)),
            percentile_75=float(np.percentile(outcomes, 75)),
            percentile_95=float(np.percentile(outcomes, 95)),
            prob_of_loss=float(np.mean(outcomes < 0)),
        )

    return results
