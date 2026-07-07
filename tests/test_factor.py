import numpy as np
import pandas as pd
import pytest

from portfolio_risk_engine.factor.fama_french import factor_regression_table, run_factor_regression


def test_factor_regression_recovers_known_loadings():
    """Construct returns that are an exact linear combination of synthetic
    factors plus tiny noise, and verify OLS recovers the true betas/alpha."""
    rng = np.random.default_rng(5)
    n = 1000
    index = pd.bdate_range("2020-01-01", periods=n)

    mkt_rf = rng.standard_normal(n) * 0.01
    smb = rng.standard_normal(n) * 0.005
    hml = rng.standard_normal(n) * 0.005
    rf = np.full(n, 0.0001)

    factors = pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf}, index=index)

    true_betas = {"Mkt-RF": 1.2, "SMB": 0.3, "HML": -0.4}
    noise = rng.standard_normal(n) * 0.0001
    excess_return = (
        0.0002
        + true_betas["Mkt-RF"] * mkt_rf
        + true_betas["SMB"] * smb
        + true_betas["HML"] * hml
        + noise
    )
    asset_returns = pd.DataFrame({"SYNTH": excess_return + rf}, index=index)

    results = run_factor_regression(asset_returns, factors, model="3F")
    assert len(results) == 1
    r = results[0]

    assert r.betas["Mkt-RF"] == pytest.approx(true_betas["Mkt-RF"], abs=0.05)
    assert r.betas["SMB"] == pytest.approx(true_betas["SMB"], abs=0.15)
    assert r.betas["HML"] == pytest.approx(true_betas["HML"], abs=0.15)
    assert r.r_squared > 0.95

    table = factor_regression_table(results)
    assert "R-squared" in table.columns
    assert table.index.tolist() == ["SYNTH"]
