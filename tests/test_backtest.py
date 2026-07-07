import numpy as np
import pandas as pd
import pytest

from portfolio_risk_engine.backtest.rebalance import _rebalance_dates, run_backtest


def test_rebalance_dates_are_hashable_against_index_timestamps(synthetic_daily_returns):
    """Regression test: the rebalance-date set must contain plain
    pd.Timestamp objects so `date in rebalance_dates` matches while
    iterating a DatetimeIndex. A prior implementation stored raw
    numpy.datetime64 values, which compare equal to Timestamps but hash
    differently -- silently making every `in` check fail and disabling
    rebalancing entirely."""
    dates = synthetic_daily_returns.index
    rebalance_set = _rebalance_dates(dates, "Q")
    assert len(rebalance_set) > 0
    hits = sum(1 for d in dates if d in rebalance_set)
    assert hits == len(rebalance_set)  # every rebalance date must be found by direct iteration


def test_quarterly_rebalancing_actually_triggers(synthetic_daily_returns):
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=synthetic_daily_returns.columns)
    result = run_backtest(synthetic_daily_returns, weights, rebalance_freq="Q", cost_bps=10.0, label="test")
    n_quarters = len(synthetic_daily_returns) // 63  # ~63 trading days/quarter
    assert result.n_rebalances >= n_quarters - 2  # allow for partial first/last quarter


def test_buy_and_hold_has_zero_rebalances(synthetic_daily_returns):
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=synthetic_daily_returns.columns)
    result = run_backtest(synthetic_daily_returns, weights, rebalance_freq=None, cost_bps=10.0, label="buy_hold")
    assert result.n_rebalances == 0


def test_higher_transaction_costs_reduce_terminal_value(synthetic_daily_returns):
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=synthetic_daily_returns.columns)
    cheap = run_backtest(synthetic_daily_returns, weights, rebalance_freq="Q", cost_bps=1.0)
    expensive = run_backtest(synthetic_daily_returns, weights, rebalance_freq="Q", cost_bps=200.0)
    assert expensive.total_transaction_costs > cheap.total_transaction_costs
    assert expensive.portfolio_value.iloc[-1] < cheap.portfolio_value.iloc[-1]


def test_backtest_weights_need_not_presum_to_one(synthetic_daily_returns):
    """run_backtest should normalize weights internally regardless of input scale."""
    unnormalized = pd.Series([1, 1, 1, 1], index=synthetic_daily_returns.columns)
    normalized = pd.Series([0.25, 0.25, 0.25, 0.25], index=synthetic_daily_returns.columns)
    r1 = run_backtest(synthetic_daily_returns, unnormalized, rebalance_freq=None)
    r2 = run_backtest(synthetic_daily_returns, normalized, rebalance_freq=None)
    np.testing.assert_allclose(r1.portfolio_value.values, r2.portfolio_value.values, rtol=1e-9)
