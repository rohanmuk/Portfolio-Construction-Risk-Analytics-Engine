"""Return and risk estimation: annualized returns/vol and covariance estimators.

Two covariance estimators are provided:

- Sample covariance: the textbook maximum-likelihood estimate. With N assets
  and T observations, it is unbiased but noisy whenever T is not much larger
  than N -- the estimation error concentrates in the extreme eigenvalues,
  which is exactly what a mean-variance optimizer (which loves to bet
  aggressively on the largest eigenvalue) will overfit to.
- Ledoit-Wolf shrinkage: shrinks the sample covariance toward a structured
  target (a scaled identity matrix) using a data-driven shrinkage intensity
  that minimizes expected estimation error (Ledoit & Wolf, 2004). This
  trades a small amount of bias for a large reduction in variance, which
  empirically produces more stable, less concentrated optimal portfolios.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_risk_engine.config import TRADING_DAYS_PER_YEAR


class CovarianceMethod(str, Enum):
    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit_wolf"


@dataclass
class MarketStatistics:
    """Container for annualized inputs to the optimizer."""

    mean_returns: pd.Series  # annualized arithmetic mean return per asset
    cov_matrix: pd.DataFrame  # annualized covariance matrix
    method: CovarianceMethod
    shrinkage_intensity: float | None = None  # only set for Ledoit-Wolf


def compute_daily_returns(prices: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """Simple (or log) daily returns from a price panel."""
    if log_returns:
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def annualize_mean_returns(daily_returns: pd.DataFrame) -> pd.Series:
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def annualize_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    return daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sample_covariance(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def ledoit_wolf_covariance(daily_returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Annualized Ledoit-Wolf shrinkage covariance matrix and shrinkage intensity."""
    lw = LedoitWolf().fit(daily_returns.values)
    cov = pd.DataFrame(
        lw.covariance_ * TRADING_DAYS_PER_YEAR,
        index=daily_returns.columns,
        columns=daily_returns.columns,
    )
    return cov, float(lw.shrinkage_)


def compute_market_statistics(
    prices: pd.DataFrame,
    method: CovarianceMethod = CovarianceMethod.LEDOIT_WOLF,
    log_returns: bool = False,
) -> MarketStatistics:
    """Compute annualized mean returns and covariance from a price panel."""
    daily_returns = compute_daily_returns(prices, log_returns=log_returns)
    mean_returns = annualize_mean_returns(daily_returns)

    shrinkage_intensity = None
    if method == CovarianceMethod.SAMPLE:
        cov_matrix = sample_covariance(daily_returns)
    elif method == CovarianceMethod.LEDOIT_WOLF:
        cov_matrix, shrinkage_intensity = ledoit_wolf_covariance(daily_returns)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown covariance method: {method}")

    return MarketStatistics(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        method=method,
        shrinkage_intensity=shrinkage_intensity,
    )


def portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    return float(np.dot(weights, mean_returns.values))


def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return float(np.sqrt(weights @ cov_matrix.values @ weights))


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> float:
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol
