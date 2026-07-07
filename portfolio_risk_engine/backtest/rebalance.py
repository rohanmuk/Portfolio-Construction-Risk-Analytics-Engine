"""Buy-and-hold vs. periodically-rebalanced backtests, net of transaction costs.

Buy-and-hold: the portfolio is purchased once at target weights and then
left alone -- weights drift with relative asset performance over time (a
long-run winner ends up overweight). Only one turnover event (the initial
purchase) incurs transaction costs.

Rebalanced: at the end of each period (default quarterly), the portfolio is
traded back to the target weights. Each rebalance incurs a transaction cost
proportional to turnover (sum of absolute weight changes) at a configurable
bps-per-trade assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk_engine.config import DEFAULT_TRANSACTION_COST_BPS, TRADING_DAYS_PER_YEAR
from portfolio_risk_engine.risk.metrics import max_drawdown


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set[pd.Timestamp]:
    """The last available trading date in each calendar period (e.g. quarter).

    Grouped via PeriodIndex rather than `Series.resample(...).last()` to
    avoid the pandas 2.2+ "Q"/offset-alias deprecation churn. Values are
    cast to plain `pd.Timestamp` explicitly -- comparing/hashing a mix of
    `pd.Timestamp` and `numpy.datetime64` can silently fail set-membership
    checks (equal values, different hash) even though `==` reports True.
    """
    periods = pd.PeriodIndex(dates, freq=freq)
    last_date_per_period = pd.Series(dates, index=periods).groupby(level=0).max()
    return {pd.Timestamp(d) for d in last_date_per_period}


@dataclass
class BacktestResult:
    label: str
    portfolio_value: pd.Series  # cumulative value, starts near 1.0
    total_transaction_costs: float  # cumulative drag from trading, as a fraction of NAV
    n_rebalances: int
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def run_backtest(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
    rebalance_freq: str | None = "Q",
    cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    risk_free_rate: float = 0.0,
    label: str = "",
) -> BacktestResult:
    """Simulate a portfolio starting at `target_weights`.

    If `rebalance_freq` is None, this is a pure buy-and-hold backtest (only
    the initial purchase incurs transaction costs). Otherwise weights are
    traded back to target at the end of each pandas offset period
    (e.g. "Q" = quarterly, "M" = monthly, "A" = annually).
    """
    weights = target_weights.reindex(asset_returns.columns).fillna(0.0).values.astype(float)
    weights = weights / weights.sum()
    cost_rate = cost_bps / 10_000.0

    rebalance_dates = _rebalance_dates(asset_returns.index, rebalance_freq) if rebalance_freq else set()

    current_weights = weights.copy()
    value = 1.0
    # Initial purchase from all-cash is a 100% turnover event.
    total_cost_drag = np.abs(weights).sum() * cost_rate
    value *= 1 - total_cost_drag
    n_rebalances = 0

    values = np.empty(len(asset_returns))
    returns_filled = asset_returns.fillna(0.0).values

    for i, date in enumerate(asset_returns.index):
        r = returns_filled[i]
        port_ret = current_weights @ r
        value *= 1 + port_ret
        current_weights = current_weights * (1 + r)
        w_sum = current_weights.sum()
        if w_sum > 0:
            current_weights = current_weights / w_sum

        if date in rebalance_dates:
            turnover = np.abs(weights - current_weights).sum()
            cost = turnover * cost_rate
            value *= 1 - cost
            total_cost_drag += cost
            current_weights = weights.copy()
            n_rebalances += 1

        values[i] = value

    value_series = pd.Series(values, index=asset_returns.index, name=label or "portfolio")

    daily_port_returns = value_series.pct_change().dropna()
    n_years = len(value_series) / TRADING_DAYS_PER_YEAR
    cagr = float(value_series.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol = float(daily_port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0
    dd, _, _ = max_drawdown(value_series)

    return BacktestResult(
        label=label,
        portfolio_value=value_series,
        total_transaction_costs=float(total_cost_drag),
        n_rebalances=n_rebalances,
        cagr=cagr,
        annualized_volatility=vol,
        sharpe_ratio=float(sharpe),
        max_drawdown=dd,
    )


def compare_buy_and_hold_vs_rebalanced(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
    rebalance_freq: str = "Q",
    cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    risk_free_rate: float = 0.0,
) -> dict[str, BacktestResult]:
    """Convenience wrapper running both backtest variants with shared assumptions."""
    buy_hold = run_backtest(
        asset_returns, target_weights, rebalance_freq=None, cost_bps=cost_bps,
        risk_free_rate=risk_free_rate, label="Buy & Hold",
    )
    rebalanced = run_backtest(
        asset_returns, target_weights, rebalance_freq=rebalance_freq, cost_bps=cost_bps,
        risk_free_rate=risk_free_rate, label=f"Rebalanced ({rebalance_freq})",
    )
    return {"buy_and_hold": buy_hold, "rebalanced": rebalanced}
