"""Portfolio optimization engine.

Implements four classic constructions on top of annualized mean returns and
a covariance matrix (see `portfolio_risk_engine.estimators`):

- Minimum variance: a convex QP, solved with cvxpy.
- Maximum Sharpe ratio: fractional (non-convex) in its natural form, so we
  use the standard Cornuejols-Tutuncu change of variables (y = kappa * w)
  that turns it into a convex QP when the feasible region is a cone (true
  for long-only + proportional weight caps). Falls back to scipy SLSQP on
  the raw weights if that transformation is infeasible.
- Efficient frontier: convex QP (minimize variance for a target return),
  swept across a grid of target returns.
- Risk parity / equal risk contribution: non-convex in general once
  per-asset weight caps are added, solved via scipy SLSQP minimizing the
  dispersion of risk contributions.

All functions accept a `mean_returns` Series and `cov_matrix` DataFrame
(annualized) and an `OptimizationConstraints` and return weights aligned to
the same asset ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portfolio_risk_engine.config import OptimizationConstraints
from portfolio_risk_engine.estimators import (
    portfolio_return,
    portfolio_sharpe_ratio,
    portfolio_volatility,
)

_JITTER = 1e-10


@dataclass
class PortfolioResult:
    """A single portfolio allocation and its headline statistics."""

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    label: str = ""

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            **self.weights.to_dict(),
        }


def _regularized(cov_matrix: pd.DataFrame) -> np.ndarray:
    n = cov_matrix.shape[0]
    return cov_matrix.values + _JITTER * np.eye(n)


def _build_result(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    label: str,
) -> PortfolioResult:
    weights = np.clip(weights, 0.0 if weights.min() > -1e-8 else weights.min(), None)
    weights = weights / weights.sum()
    w_series = pd.Series(weights, index=mean_returns.index)
    return PortfolioResult(
        weights=w_series,
        expected_return=portfolio_return(weights, mean_returns),
        volatility=portfolio_volatility(weights, cov_matrix),
        sharpe_ratio=portfolio_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate),
        label=label,
    )


def _solve_qp(objective, constraints_list) -> None:
    prob = cp.Problem(objective, constraints_list)
    for solver in (cp.OSQP, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return prob
        except (cp.error.SolverError, cp.error.DCPError):
            continue
    return prob


def min_variance_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: OptimizationConstraints | None = None,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """The global minimum variance portfolio subject to box constraints."""
    constraints = constraints or OptimizationConstraints()
    n = len(mean_returns)
    sigma = _regularized(cov_matrix)
    lo, hi = zip(*constraints.bounds(n))
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    cons = [cp.sum(w) == 1, w >= np.array(lo), w <= np.array(hi)]
    prob = _solve_qp(objective, cons)
    if w.value is None:
        raise RuntimeError(f"Min-variance optimization failed to converge (status={prob.status}).")
    return _build_result(w.value, mean_returns, cov_matrix, risk_free_rate, "Minimum Variance")


def _max_sharpe_scipy(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: OptimizationConstraints,
    risk_free_rate: float,
) -> np.ndarray:
    n = len(mean_returns)
    mu = mean_returns.values
    sigma = _regularized(cov_matrix)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu
        vol = np.sqrt(max(w @ sigma @ w, 1e-16))
        return -(ret - risk_free_rate) / vol

    bounds = constraints.bounds(n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n
    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"Max-Sharpe fallback optimization failed: {result.message}")
    return result.x


def max_sharpe_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: OptimizationConstraints | None = None,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """The tangency (maximum Sharpe ratio) portfolio.

    Solved as a convex QP via the Cornuejols-Tutuncu homogenization
    (y = kappa * w, kappa > 0), which is exact whenever the constraint set is
    a scaled cone -- true here since box bounds are proportional to kappa.
    Falls back to direct scipy SLSQP maximization if that QP is infeasible
    (e.g. all assets have non-positive excess return).
    """
    constraints = constraints or OptimizationConstraints()
    n = len(mean_returns)
    mu = mean_returns.values
    sigma = _regularized(cov_matrix)
    excess = mu - risk_free_rate

    weights = None
    if np.any(excess > 0):
        lo, hi = zip(*constraints.bounds(n))
        lo, hi = np.array(lo), np.array(hi)
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)
        cons = [
            excess @ y == 1,
            cp.sum(y) == kappa,
            kappa >= 1e-8,
            y >= lo * kappa,
            y <= hi * kappa,
        ]
        objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma)))
        prob = _solve_qp(objective, cons)
        if y.value is not None and kappa.value and kappa.value > 1e-8:
            weights = y.value / kappa.value

    if weights is None:
        weights = _max_sharpe_scipy(mean_returns, cov_matrix, constraints, risk_free_rate)

    return _build_result(weights, mean_returns, cov_matrix, risk_free_rate, "Maximum Sharpe Ratio")


def risk_contributions(weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
    """Each asset's contribution to total portfolio variance-based risk."""
    sigma = cov_matrix.values
    port_vol = np.sqrt(max(weights @ sigma @ weights, 1e-16))
    marginal = sigma @ weights / port_vol
    return weights * marginal


def risk_parity_portfolio(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: OptimizationConstraints | None = None,
    risk_free_rate: float = 0.0,
    risk_budget: np.ndarray | None = None,
) -> PortfolioResult:
    """Equal (or budgeted) risk contribution portfolio.

    Minimizes the sum of squared deviations between each asset's fractional
    contribution to portfolio risk and its target risk budget (equal-weight
    budget by default), subject to box constraints and full investment.
    """
    constraints = constraints or OptimizationConstraints()
    n = len(mean_returns)
    sigma = cov_matrix.values
    budget = risk_budget if risk_budget is not None else np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov_matrix)
        rc_frac = rc / rc.sum()
        return float(np.sum((rc_frac - budget) ** 2))

    bounds = [(max(lo, 1e-6), hi) for lo, hi in constraints.bounds(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 2000, "ftol": 1e-14})
    if not result.success:
        raise RuntimeError(f"Risk parity optimization failed: {result.message}")

    return _build_result(result.x, mean_returns, cov_matrix, risk_free_rate, "Risk Parity")


def _max_achievable_return(
    mean_returns: pd.Series, constraints: OptimizationConstraints
) -> float:
    n = len(mean_returns)
    lo, hi = zip(*constraints.bounds(n))
    w = cp.Variable(n)
    objective = cp.Maximize(mean_returns.values @ w)
    cons = [cp.sum(w) == 1, w >= np.array(lo), w <= np.array(hi)]
    prob = cp.Problem(objective, cons)
    prob.solve()
    return float(mean_returns.values @ w.value)


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    constraints: OptimizationConstraints | None = None,
    n_points: int = 50,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Sweep target returns from the min-variance return to the max
    achievable return, solving a minimum-variance QP at each point.

    Returns a DataFrame with one row per frontier point: target_return,
    volatility, sharpe_ratio, and one column per asset weight.
    """
    constraints = constraints or OptimizationConstraints()
    n = len(mean_returns)
    sigma = _regularized(cov_matrix)
    lo, hi = zip(*constraints.bounds(n))
    lo, hi = np.array(lo), np.array(hi)

    min_var = min_variance_portfolio(mean_returns, cov_matrix, constraints, risk_free_rate)
    min_ret = min_var.expected_return
    max_ret = _max_achievable_return(mean_returns, constraints)
    # Guard against a degenerate (near-zero-width) sweep range.
    if max_ret <= min_ret:
        max_ret = min_ret + 1e-6

    target_returns = np.linspace(min_ret, max_ret, n_points)
    rows = []
    for target in target_returns:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
        cons = [
            cp.sum(w) == 1,
            w >= lo,
            w <= hi,
            mean_returns.values @ w >= target,
        ]
        prob = _solve_qp(objective, cons)
        if w.value is None:
            continue
        weights = np.clip(w.value, 0, None)
        weights = weights / weights.sum()
        vol = portfolio_volatility(weights, cov_matrix)
        ret = portfolio_return(weights, mean_returns)
        sharpe = portfolio_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)
        rows.append({"target_return": ret, "volatility": vol, "sharpe_ratio": sharpe,
                      **dict(zip(mean_returns.index, weights))})

    return pd.DataFrame(rows)
