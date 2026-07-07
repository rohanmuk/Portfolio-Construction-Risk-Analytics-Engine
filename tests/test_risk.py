import numpy as np
import pandas as pd
import pytest

from portfolio_risk_engine.risk.metrics import (
    annualized_return,
    annualized_volatility,
    beta_to_benchmark,
    build_risk_report,
    downside_deviation,
    historical_cvar,
    historical_var,
    max_drawdown,
    sortino_ratio,
)


def test_historical_var_known_distribution():
    # 100 evenly spaced returns from -0.10 to +0.09; 5th percentile is well-defined.
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    var_95 = historical_var(returns, confidence=0.95)
    assert var_95 == pytest.approx(-np.percentile(returns, 5))
    assert var_95 > 0  # reported as a positive loss magnitude


def test_cvar_is_at_least_as_large_as_var():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.standard_t(df=3, size=5000) * 0.01)  # fat-tailed
    for conf in (0.95, 0.99):
        var = historical_var(returns, conf)
        cvar = historical_cvar(returns, conf)
        assert cvar >= var - 1e-12


def test_cvar_higher_confidence_is_more_severe():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.standard_normal(5000) * 0.01)
    cvar_95 = historical_cvar(returns, 0.95)
    cvar_99 = historical_cvar(returns, 0.99)
    assert cvar_99 >= cvar_95


def test_max_drawdown_on_known_path():
    # Wealth path: 1.0 -> 1.2 (peak) -> 0.6 (trough, -50% from peak) -> 0.9
    prices = pd.Series(
        [1.0, 1.2, 1.0, 0.8, 0.6, 0.7, 0.9],
        index=pd.bdate_range("2020-01-01", periods=7),
    )
    dd, peak_date, trough_date = max_drawdown(prices)
    assert dd == pytest.approx(0.5, abs=1e-9)
    assert peak_date == prices.index[1]
    assert trough_date == prices.index[4]


def test_downside_deviation_ignores_gains():
    returns = pd.Series([0.05, 0.05, 0.05, 0.05])  # all positive
    assert downside_deviation(returns) == 0.0

    mixed = pd.Series([0.05, -0.05, 0.05, -0.05])
    assert downside_deviation(mixed) > 0.0


def test_sortino_ratio_zero_downside_returns_zero():
    returns = pd.Series([0.01, 0.02, 0.01, 0.03])
    assert sortino_ratio(returns) == 0.0


def test_beta_to_benchmark_exact_linear_relationship():
    rng = np.random.default_rng(3)
    benchmark = pd.Series(rng.standard_normal(500) * 0.01)
    asset = 1.5 * benchmark  # exact beta of 1.5, zero idiosyncratic noise
    beta = beta_to_benchmark(asset, benchmark)
    assert beta == pytest.approx(1.5, abs=1e-9)


def test_beta_to_benchmark_handles_misaligned_index():
    idx1 = pd.date_range("2020-01-01", periods=10)
    idx2 = pd.date_range("2020-01-05", periods=10)
    asset = pd.Series(np.arange(10, dtype=float), index=idx1)
    benchmark = pd.Series(np.arange(10, dtype=float), index=idx2)
    beta = beta_to_benchmark(asset, benchmark)
    assert not np.isnan(beta)  # should compute over the overlapping window


def test_annualized_return_matches_cagr_definition():
    # Constant daily return r compounded for exactly 1 year (252 days):
    # CAGR should equal (1+r)^252 - 1's annualized rate, i.e. just (1+r)^252 - 1
    # since n_years == 1 exactly.
    r = 0.001
    returns = pd.Series([r] * 252)
    result = annualized_return(returns)
    assert result == pytest.approx((1 + r) ** 252 - 1, rel=1e-9)


def test_build_risk_report_internal_consistency(synthetic_daily_returns):
    port_returns = synthetic_daily_returns.mean(axis=1)  # equal-weight portfolio proxy
    benchmark = synthetic_daily_returns["ASSET_A"]
    report = build_risk_report(port_returns, risk_free_rate=0.02, benchmark_returns=benchmark)

    assert report.cvar_95 >= report.var_95 - 1e-9
    assert report.cvar_99 >= report.var_99 - 1e-9
    assert report.var_99 >= report.var_95 - 1e-9  # 99% tail is further out
    assert 0.0 <= report.max_drawdown <= 1.0
    assert report.annualized_volatility > 0
    assert report.beta is not None and not np.isnan(report.beta)


def test_build_risk_report_without_benchmark_has_no_beta(synthetic_daily_returns):
    port_returns = synthetic_daily_returns.mean(axis=1)
    report = build_risk_report(port_returns, risk_free_rate=0.0)
    assert report.beta is None
