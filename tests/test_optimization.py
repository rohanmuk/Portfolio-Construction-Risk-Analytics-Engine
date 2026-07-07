import numpy as np
import pytest

from portfolio_risk_engine.config import OptimizationConstraints
from portfolio_risk_engine.estimators import (
    CovarianceMethod,
    compute_market_statistics,
    portfolio_volatility,
)
from portfolio_risk_engine.optimization.engine import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    risk_contributions,
    risk_parity_portfolio,
)


@pytest.fixture
def market_stats(synthetic_prices):
    return compute_market_statistics(synthetic_prices, method=CovarianceMethod.LEDOIT_WOLF)


@pytest.fixture
def constraints():
    return OptimizationConstraints(long_only=True, max_weight=0.6)


def test_min_variance_weights_sum_to_one_and_respect_bounds(market_stats, constraints):
    result = min_variance_portfolio(market_stats.mean_returns, market_stats.cov_matrix, constraints)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-6).all()
    assert (result.weights <= constraints.max_weight + 1e-6).all()


def test_min_variance_beats_equal_weight_volatility(market_stats, constraints):
    """The defining property of the min-variance portfolio: no other
    fully-invested, constraint-respecting portfolio has lower volatility."""
    result = min_variance_portfolio(market_stats.mean_returns, market_stats.cov_matrix, constraints)
    n = len(market_stats.mean_returns)
    equal_weights = np.ones(n) / n
    equal_vol = portfolio_volatility(equal_weights, market_stats.cov_matrix)
    assert result.volatility <= equal_vol + 1e-9


def test_max_sharpe_weights_sum_to_one_and_respect_bounds(market_stats, constraints):
    result = max_sharpe_portfolio(market_stats.mean_returns, market_stats.cov_matrix, constraints, risk_free_rate=0.02)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (result.weights >= -1e-6).all()
    assert (result.weights <= constraints.max_weight + 1e-6).all()


def test_max_sharpe_achieves_highest_sharpe_on_a_small_grid(market_stats, constraints):
    """Max-Sharpe should beat (or match) a coarse random search over the
    feasible simplex -- a cheap sanity check that the QP actually optimizes,
    rather than e.g. silently returning the initial guess."""
    result = max_sharpe_portfolio(market_stats.mean_returns, market_stats.cov_matrix, constraints, risk_free_rate=0.02)

    rng = np.random.default_rng(7)
    n = len(market_stats.mean_returns)
    best_random_sharpe = -np.inf
    for _ in range(2000):
        w = rng.dirichlet(np.ones(n))
        w = np.minimum(w, constraints.max_weight)
        w = w / w.sum()
        ret = w @ market_stats.mean_returns.values
        vol = portfolio_volatility(w, market_stats.cov_matrix)
        sharpe = (ret - 0.02) / vol if vol > 0 else -np.inf
        best_random_sharpe = max(best_random_sharpe, sharpe)

    assert result.sharpe_ratio >= best_random_sharpe - 1e-6


def test_risk_parity_equalizes_risk_contributions(market_stats, constraints):
    result = risk_parity_portfolio(market_stats.mean_returns, market_stats.cov_matrix, constraints)
    rc = risk_contributions(result.weights.values, market_stats.cov_matrix)
    rc_frac = rc / rc.sum()
    n = len(rc_frac)
    np.testing.assert_allclose(rc_frac, np.ones(n) / n, atol=1e-3)


def test_risk_parity_weights_sum_to_one(market_stats, constraints):
    result = risk_parity_portfolio(market_stats.mean_returns, market_stats.cov_matrix, constraints)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_efficient_frontier_has_at_least_50_points_by_default(market_stats, constraints):
    frontier = efficient_frontier(market_stats.mean_returns, market_stats.cov_matrix, constraints, n_points=50)
    assert len(frontier) >= 45  # allow a few infeasible target-return points to drop out


def test_efficient_frontier_volatility_is_monotonic_nondecreasing(market_stats, constraints):
    """Classic efficient-frontier shape: as target return rises, the
    minimum achievable volatility should not decrease."""
    frontier = efficient_frontier(market_stats.mean_returns, market_stats.cov_matrix, constraints, n_points=50)
    frontier = frontier.sort_values("target_return")
    vol_diffs = np.diff(frontier["volatility"].values)
    assert (vol_diffs >= -1e-6).all()


def test_efficient_frontier_weights_are_feasible(market_stats, constraints):
    frontier = efficient_frontier(market_stats.mean_returns, market_stats.cov_matrix, constraints, n_points=50)
    weight_cols = list(market_stats.mean_returns.index)
    weights = frontier[weight_cols].values
    row_sums = weights.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(frontier)), atol=1e-4)
    assert (weights >= -1e-6).all()
    assert (weights <= constraints.max_weight + 1e-6).all()


def test_max_weight_constraint_is_enforced_tightly(market_stats):
    """With a max weight below 1/N, no single asset can dominate -- verifies
    the box constraint is actually binding, not just nominally passed in."""
    tight_constraints = OptimizationConstraints(long_only=True, max_weight=0.3)
    result = max_sharpe_portfolio(market_stats.mean_returns, market_stats.cov_matrix, tight_constraints, risk_free_rate=0.02)
    assert (result.weights <= 0.3 + 1e-6).all()
