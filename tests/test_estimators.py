import numpy as np
import pandas as pd
import pytest

from portfolio_risk_engine.estimators import (
    CovarianceMethod,
    annualize_mean_returns,
    annualize_volatility,
    compute_daily_returns,
    compute_market_statistics,
    ledoit_wolf_covariance,
    portfolio_return,
    portfolio_sharpe_ratio,
    portfolio_volatility,
    sample_covariance,
)


def test_compute_daily_returns_matches_pct_change(synthetic_prices):
    returns = compute_daily_returns(synthetic_prices)
    expected = synthetic_prices.pct_change().dropna(how="all")
    pd.testing.assert_frame_equal(returns, expected)


def test_annualized_stats_scale_by_trading_days(synthetic_daily_returns):
    mean_ret = annualize_mean_returns(synthetic_daily_returns)
    vol = annualize_volatility(synthetic_daily_returns)
    np.testing.assert_allclose(mean_ret.values, synthetic_daily_returns.mean().values * 252)
    np.testing.assert_allclose(vol.values, synthetic_daily_returns.std(ddof=1).values * np.sqrt(252))


def test_sample_covariance_is_symmetric_psd(synthetic_daily_returns):
    cov = sample_covariance(synthetic_daily_returns)
    np.testing.assert_allclose(cov.values, cov.values.T)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert (eigenvalues >= -1e-10).all()


def test_ledoit_wolf_shrinkage_intensity_in_unit_interval(synthetic_daily_returns):
    cov, shrinkage = ledoit_wolf_covariance(synthetic_daily_returns)
    assert 0.0 <= shrinkage <= 1.0
    np.testing.assert_allclose(cov.values, cov.values.T)


def test_ledoit_wolf_shrinks_off_diagonals_toward_zero(synthetic_daily_returns):
    """Ledoit-Wolf shrinks toward a scaled identity target, so (holding the
    diagonal roughly fixed) off-diagonal covariance terms should shrink in
    magnitude relative to the noisy sample estimate."""
    sample_cov = sample_covariance(synthetic_daily_returns)
    lw_cov, shrinkage = ledoit_wolf_covariance(synthetic_daily_returns)
    assert shrinkage > 0
    sample_offdiag = np.abs(sample_cov.values[np.triu_indices(4, k=1)]).sum()
    lw_offdiag = np.abs(lw_cov.values[np.triu_indices(4, k=1)]).sum()
    assert lw_offdiag < sample_offdiag


def test_compute_market_statistics_returns_both_methods(synthetic_prices):
    sample_stats = compute_market_statistics(synthetic_prices, method=CovarianceMethod.SAMPLE)
    lw_stats = compute_market_statistics(synthetic_prices, method=CovarianceMethod.LEDOIT_WOLF)
    assert sample_stats.shrinkage_intensity is None
    assert lw_stats.shrinkage_intensity is not None
    assert list(sample_stats.mean_returns.index) == list(synthetic_prices.columns)


def test_portfolio_helpers_basic_arithmetic():
    weights = np.array([0.5, 0.5])
    mean_returns = pd.Series([0.10, 0.20], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["A", "B"], columns=["A", "B"])

    assert portfolio_return(weights, mean_returns) == pytest.approx(0.15)
    assert portfolio_volatility(weights, cov) == pytest.approx(np.sqrt(0.5**2 * 0.04 + 0.5**2 * 0.01))

    sharpe = portfolio_sharpe_ratio(weights, mean_returns, cov, risk_free_rate=0.05)
    expected_sharpe = (0.15 - 0.05) / portfolio_volatility(weights, cov)
    assert sharpe == pytest.approx(expected_sharpe)


def test_portfolio_sharpe_ratio_zero_vol_returns_zero():
    weights = np.array([1.0])
    mean_returns = pd.Series([0.10], index=["A"])
    cov = pd.DataFrame([[0.0]], index=["A"], columns=["A"])
    assert portfolio_sharpe_ratio(weights, mean_returns, cov) == 0.0
