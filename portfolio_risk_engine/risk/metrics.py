"""Historical risk metrics: VaR, CVaR, drawdown, downside deviation, and beta.

All functions operate on a return series (daily simple returns, portfolio or
asset level) unless noted otherwise. VaR/CVaR are computed via historical
simulation (the empirical quantile of realized returns) rather than a
parametric (Gaussian) assumption, since asset returns are fat-tailed and a
normal-distribution VaR systematically understates tail risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk_engine.config import TRADING_DAYS_PER_YEAR


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) Value at Risk, as a positive loss fraction.

    E.g. VaR_95% = 0.02 means there is a 5% chance of losing 2% or more of
    portfolio value on a given day (based on the empirical return distribution).
    """
    alpha = 1 - confidence
    return float(-np.percentile(returns.dropna(), alpha * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Conditional VaR (Expected Shortfall): the average loss in
    the tail beyond the VaR threshold. Always >= VaR at the same confidence.
    """
    returns = returns.dropna()
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def max_drawdown(cumulative_returns: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Maximum peak-to-trough drawdown of a cumulative wealth series.

    Returns (max_drawdown_fraction, peak_date, trough_date). Drawdown is
    reported as a positive fraction (e.g. 0.34 == a 34% decline from peak).
    """
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1.0
    trough_date = drawdown.idxmin()
    max_dd = -drawdown.loc[trough_date]
    peak_date = cumulative_returns.loc[:trough_date].idxmax()
    return float(max_dd), peak_date, trough_date


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    """Annualized downside deviation below a minimum acceptable return (MAR),
    expressed as a daily return threshold (default 0, i.e. any loss day).
    """
    downside = np.minimum(returns - mar, 0.0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, mar: float = 0.0) -> float:
    """Annualized Sortino ratio: excess return per unit of downside deviation."""
    annualized_return = returns.mean() * TRADING_DAYS_PER_YEAR
    dd = downside_deviation(returns, mar)
    if dd == 0:
        return 0.0
    return float((annualized_return - risk_free_rate) / dd)


def beta_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """OLS beta of `returns` against `benchmark_returns` (aligned on index)."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var = np.var(aligned.iloc[:, 1], ddof=1)
    if var == 0:
        return float("nan")
    return float(cov / var)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def annualized_return(returns: pd.Series) -> float:
    """Compounded annual growth rate (CAGR) from a daily return series."""
    cumulative = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return float(cumulative ** (1 / n_years) - 1)


@dataclass
class RiskReport:
    """A complete headline risk report for a return series."""

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    max_drawdown_peak: pd.Timestamp
    max_drawdown_trough: pd.Timestamp
    downside_deviation: float
    beta: float | None = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["max_drawdown_peak"] = str(d["max_drawdown_peak"].date())
        d["max_drawdown_trough"] = str(d["max_drawdown_trough"].date())
        return d


def build_risk_report(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_returns: pd.Series | None = None,
) -> RiskReport:
    """Compute the full suite of risk metrics for a portfolio return series."""
    returns = returns.dropna()
    cumulative = (1 + returns).cumprod()
    dd, peak, trough = max_drawdown(cumulative)
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    beta = beta_to_benchmark(returns, benchmark_returns) if benchmark_returns is not None else None

    return RiskReport(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=float(sharpe),
        sortino_ratio=sortino_ratio(returns, risk_free_rate),
        var_95=historical_var(returns, 0.95),
        cvar_95=historical_cvar(returns, 0.95),
        var_99=historical_var(returns, 0.99),
        cvar_99=historical_cvar(returns, 0.99),
        max_drawdown=dd,
        max_drawdown_peak=peak,
        max_drawdown_trough=trough,
        downside_deviation=downside_deviation(returns),
        beta=beta,
    )
