"""Portfolio Construction & Risk Analytics Engine -- interactive Streamlit app.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_risk_engine.backtest.rebalance import compare_buy_and_hold_vs_rebalanced
from portfolio_risk_engine.config import (
    BENCHMARK_TICKER,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_START_DATE,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_UNIVERSE,
    OptimizationConstraints,
    STRESS_TEST_START_DATE,
)
from portfolio_risk_engine.data.loader import align_and_clean, get_adjusted_close, get_raw_daily_returns
from portfolio_risk_engine.estimators import CovarianceMethod, compute_daily_returns, compute_market_statistics
from portfolio_risk_engine.factor.fama_french import (
    download_fama_french_factors,
    factor_regression_table,
    run_factor_regression,
)
from portfolio_risk_engine.optimization.engine import (
    PortfolioResult,
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    risk_parity_portfolio,
)
from portfolio_risk_engine.reporting.charts import (
    allocation_pie_chart,
    cumulative_return_chart,
    drawdown_chart,
    efficient_frontier_chart,
    monte_carlo_fan_chart,
    stress_test_bar_chart,
)
from portfolio_risk_engine.reporting.pdf_report import generate_portfolio_proposal_pdf
from portfolio_risk_engine.risk.metrics import build_risk_report
from portfolio_risk_engine.risk.stress import portfolio_daily_returns, run_monte_carlo, run_stress_tests

st.set_page_config(
    page_title="Portfolio Construction & Risk Analytics Engine",
    page_icon="\U0001F4C8",
    layout="wide",
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Cached data / computation layer
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading price history...", ttl=3600)
def load_clean_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    raw = get_adjusted_close(list(tickers), start=start)
    return align_and_clean(raw)


@st.cache_data(show_spinner="Downloading extended history for stress tests...", ttl=3600)
def load_stress_returns(tickers: tuple[str, ...]) -> pd.DataFrame:
    return get_raw_daily_returns(list(tickers), start=STRESS_TEST_START_DATE)


@st.cache_data(show_spinner="Downloading Fama-French factor data...", ttl=86400)
def load_ff_factors(model: str) -> pd.DataFrame:
    return download_fama_french_factors(model)


@st.cache_data(show_spinner=False)
def cached_monte_carlo(returns_hash: str, returns: pd.Series, n_paths: int):
    return run_monte_carlo(returns, n_paths=n_paths)


# --------------------------------------------------------------------------
# Sidebar: universe, constraints, assumptions
# --------------------------------------------------------------------------

st.sidebar.title("Portfolio Construction &\nRisk Analytics Engine")
st.sidebar.caption("Educational / research tool. Not investment advice.")

st.sidebar.header("1. Universe")
selected_defaults = st.sidebar.multiselect(
    "Default asset universe",
    options=list(DEFAULT_UNIVERSE.keys()),
    default=list(DEFAULT_UNIVERSE.keys()),
    format_func=lambda t: f"{t} — {DEFAULT_UNIVERSE[t]}",
)
custom_input = st.sidebar.text_input("Add custom tickers (comma-separated)", value="")
custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
tickers = sorted(set(selected_defaults) | set(custom_tickers))
universe_labels = {**DEFAULT_UNIVERSE, **{t: t for t in custom_tickers}}

start_date = st.sidebar.date_input("History start date", value=pd.Timestamp(DEFAULT_START_DATE))
benchmark_ticker = st.sidebar.text_input("Benchmark ticker", value=BENCHMARK_TICKER).strip().upper()

st.sidebar.header("2. Risk Model")
cov_choice = st.sidebar.radio(
    "Covariance estimator",
    ["Ledoit-Wolf Shrinkage", "Sample Covariance"],
    help="Ledoit-Wolf shrinks the noisy sample covariance toward a structured "
         "target, trading a little bias for much lower estimation variance -- "
         "usually more stable optimized weights out-of-sample.",
)
cov_method = CovarianceMethod.LEDOIT_WOLF if "Ledoit" in cov_choice else CovarianceMethod.SAMPLE
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual)", value=0.03, step=0.0025, format="%.4f")

st.sidebar.header("3. Constraints")
long_only = st.sidebar.checkbox("Long-only (no short selling)", value=True)
max_weight = st.sidebar.slider("Max weight per asset", min_value=0.05, max_value=1.0, value=DEFAULT_MAX_WEIGHT, step=0.05)
n_frontier_points = st.sidebar.slider("Efficient frontier resolution (points)", 50, 150, 50, step=10)

st.sidebar.header("4. Recommended Objective")
objective = st.sidebar.selectbox(
    "Highlight & export this portfolio",
    ["Maximum Sharpe Ratio", "Minimum Variance", "Risk Parity"],
)

st.sidebar.header("5. Backtest Assumptions")
rebalance_freq = st.sidebar.selectbox("Rebalance frequency", ["Q", "M", "A"], format_func=lambda f: {"Q": "Quarterly", "M": "Monthly", "A": "Annual"}[f])
cost_bps = st.sidebar.number_input("Transaction cost (bps per trade)", value=DEFAULT_TRANSACTION_COST_BPS, step=1.0)

if len(tickers) < 2:
    st.warning("Select at least two tickers in the sidebar to build a portfolio.")
    st.stop()

constraints = OptimizationConstraints(long_only=long_only, max_weight=max_weight)

# --------------------------------------------------------------------------
# Data & core computation
# --------------------------------------------------------------------------

try:
    all_needed = tuple(sorted(set(tickers) | {benchmark_ticker}))
    prices = load_clean_prices(all_needed, str(start_date))
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to download price data: {exc}")
    st.stop()

missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(f"No usable data for: {', '.join(missing)} -- excluded from analysis.")
    tickers = [t for t in tickers if t in prices.columns]

asset_prices = prices[tickers]
asset_returns = compute_daily_returns(asset_prices)
benchmark_returns = compute_daily_returns(prices[[benchmark_ticker]])[benchmark_ticker] if benchmark_ticker in prices.columns else None

stats = compute_market_statistics(asset_prices, method=cov_method)

portfolios: dict[str, PortfolioResult] = {
    "Maximum Sharpe Ratio": max_sharpe_portfolio(stats.mean_returns, stats.cov_matrix, constraints, risk_free_rate),
    "Minimum Variance": min_variance_portfolio(stats.mean_returns, stats.cov_matrix, constraints, risk_free_rate),
    "Risk Parity": risk_parity_portfolio(stats.mean_returns, stats.cov_matrix, constraints, risk_free_rate),
}
recommended = portfolios[objective]

frontier = efficient_frontier(stats.mean_returns, stats.cov_matrix, constraints, n_points=n_frontier_points, risk_free_rate=risk_free_rate)

st.title("Portfolio Construction & Risk Analytics Engine")
st.caption(
    f"{len(tickers)} assets · {cov_choice} · "
    f"{'Long-only' if long_only else 'Long/short'}, max {max_weight:.0%} per asset · "
    f"History since {prices.index.min().date()}"
)

tab_frontier, tab_risk, tab_factor, tab_stress, tab_backtest, tab_export = st.tabs(
    ["Efficient Frontier", "Risk Report", "Factor Analysis", "Stress & Monte Carlo", "Rebalancing Backtest", "Export PDF"]
)

# --------------------------------------------------------------------------
# Tab 1: Efficient Frontier + Allocation
# --------------------------------------------------------------------------
with tab_frontier:
    col1, col2 = st.columns([1.4, 1])
    with col1:
        asset_points = pd.DataFrame({
            "volatility": np.sqrt(np.diag(stats.cov_matrix.values)),
            "return": stats.mean_returns.values,
        }, index=stats.mean_returns.index)
        highlighted = {name: {"volatility": p.volatility, "expected_return": p.expected_return} for name, p in portfolios.items()}
        st.plotly_chart(efficient_frontier_chart(frontier, highlighted, asset_points), use_container_width=True)
        if stats.shrinkage_intensity is not None:
            st.caption(f"Ledoit-Wolf shrinkage intensity: {stats.shrinkage_intensity:.3f} (0 = pure sample covariance, 1 = pure structured target)")
    with col2:
        st.plotly_chart(allocation_pie_chart(recommended.weights, title=f"{objective} Allocation"), use_container_width=True)

    st.subheader("Candidate Portfolios")
    summary_rows = []
    for name, p in portfolios.items():
        summary_rows.append({
            "Portfolio": name, "Expected Return": p.expected_return, "Volatility": p.volatility, "Sharpe Ratio": p.sharpe_ratio,
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Portfolio")
    st.dataframe(
        summary_df.style.format({"Expected Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}),
        width="stretch",
    )

    with st.expander("Full weights for recommended portfolio"):
        w = recommended.weights[recommended.weights > 1e-4].sort_values(ascending=False)
        w_df = pd.DataFrame({"Weight": w, "Description": [universe_labels.get(t, t) for t in w.index]})
        st.dataframe(w_df.style.format({"Weight": "{:.2%}"}), width="stretch")

# --------------------------------------------------------------------------
# Tab 2: Risk report
# --------------------------------------------------------------------------
with tab_risk:
    port_returns = portfolio_daily_returns(asset_returns, recommended.weights)
    report = build_risk_report(port_returns, risk_free_rate=risk_free_rate, benchmark_returns=benchmark_returns)

    metric_cols = st.columns(4)
    metrics_display = [
        ("Annualized Return", f"{report.annualized_return:.2%}"),
        ("Annualized Volatility", f"{report.annualized_volatility:.2%}"),
        ("Sharpe Ratio", f"{report.sharpe_ratio:.2f}"),
        ("Sortino Ratio", f"{report.sortino_ratio:.2f}"),
        (f"Beta vs {benchmark_ticker}", f"{report.beta:.2f}" if report.beta is not None else "n/a"),
        ("Max Drawdown", f"{report.max_drawdown:.2%}"),
        ("Downside Deviation", f"{report.downside_deviation:.2%}"),
        ("Max DD Window", f"{report.max_drawdown_peak.date()} → {report.max_drawdown_trough.date()}"),
    ]
    for i, (label, value) in enumerate(metrics_display):
        metric_cols[i % 4].metric(label, value)

    var_cols = st.columns(4)
    var_cols[0].metric("95% Daily VaR", f"{report.var_95:.2%}")
    var_cols[1].metric("95% Daily CVaR", f"{report.cvar_95:.2%}")
    var_cols[2].metric("99% Daily VaR", f"{report.var_99:.2%}")
    var_cols[3].metric("99% Daily CVaR", f"{report.cvar_99:.2%}")

    cumulative = (1 + port_returns).cumprod()
    st.plotly_chart(cumulative_return_chart({objective: cumulative}), use_container_width=True)
    st.plotly_chart(drawdown_chart(cumulative), use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3: Factor analysis
# --------------------------------------------------------------------------
with tab_factor:
    ff_model = st.radio("Fama-French model", ["3F", "5F"], horizontal=True, format_func=lambda m: "3-Factor (Mkt-RF, SMB, HML)" if m == "3F" else "5-Factor (adds RMW, CMA)")
    try:
        factors = load_ff_factors(ff_model)
        results = run_factor_regression(asset_returns, factors, model=ff_model)
        table = factor_regression_table(results)
        st.dataframe(
            table.style.format({c: "{:.3f}" for c in table.columns if c != "R-squared"} | {"R-squared": "{:.1%}", "Annualized Alpha": "{:.2%}"}),
            width="stretch",
        )
        st.caption(
            "Alpha is annualized (daily intercept × 252): the average excess return unexplained by "
            "factor exposures. Betas are daily factor loadings. R-squared is the fraction of an asset's "
            "return variance explained by the factor model."
        )

        port_ret_series = portfolio_daily_returns(asset_returns, recommended.weights).to_frame(name=objective)
        port_factor_result = run_factor_regression(port_ret_series, factors, model=ff_model)
        st.subheader(f"Recommended Portfolio ({objective}) Factor Exposure")
        st.dataframe(factor_regression_table(port_factor_result).style.format({c: "{:.3f}" for c in factor_regression_table(port_factor_result).columns if c != "R-squared"} | {"R-squared": "{:.1%}", "Annualized Alpha": "{:.2%}"}), width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Factor regression unavailable: {exc}")

# --------------------------------------------------------------------------
# Tab 4: Stress testing + Monte Carlo
# --------------------------------------------------------------------------
with tab_stress:
    st.subheader("Historical Stress Windows")
    stress_returns = load_stress_returns(tuple(tickers))
    stress_results = run_stress_tests(stress_returns, recommended.weights)
    if stress_results:
        st.plotly_chart(stress_test_bar_chart(stress_results), use_container_width=True)
        stress_df = pd.DataFrame([{
            "Window": r.window_name, "Start": r.start, "End": r.end,
            "Total Return": r.total_return, "Max Drawdown": r.max_drawdown,
            "Trading Days": r.n_trading_days,
            "Excluded (insufficient history)": ", ".join(r.excluded_assets) or "—",
        } for r in stress_results]).set_index("Window")
        st.dataframe(stress_df.style.format({"Total Return": "{:.2%}", "Max Drawdown": "{:.2%}"}), width="stretch")
    else:
        st.info("No stress windows overlap the available price history for this universe.")

    st.subheader("Monte Carlo Simulation (Bootstrapped, 10,000 Paths)")
    port_returns_full = portfolio_daily_returns(asset_returns, recommended.weights)
    mc_results = run_monte_carlo(port_returns_full)
    mc_cols = st.columns(len(mc_results))
    for col, (horizon, res) in zip(mc_cols, mc_results.items()):
        with col:
            st.plotly_chart(monte_carlo_fan_chart(res.paths, f"{horizon}-Year Horizon"), use_container_width=True)
            st.metric(f"{horizon}Y Median Outcome", f"{res.median:.1%}")
            st.metric(f"{horizon}Y Prob. of Loss", f"{res.prob_of_loss:.1%}")
            st.caption(f"5th–95th percentile: {res.percentile_5:.1%} to {res.percentile_95:.1%}")

# --------------------------------------------------------------------------
# Tab 5: Rebalancing backtest
# --------------------------------------------------------------------------
with tab_backtest:
    bt = compare_buy_and_hold_vs_rebalanced(
        asset_returns, recommended.weights, rebalance_freq=rebalance_freq, cost_bps=cost_bps, risk_free_rate=risk_free_rate,
    )
    st.plotly_chart(
        cumulative_return_chart(
            {r.label: r.portfolio_value for r in bt.values()},
            title=f"Buy & Hold vs. {({'Q':'Quarterly','M':'Monthly','A':'Annual'})[rebalance_freq]} Rebalancing (net of {cost_bps:.0f}bps/trade)",
        ),
        use_container_width=True,
    )
    bt_df = pd.DataFrame([{
        "Strategy": r.label, "CAGR": r.cagr, "Volatility": r.annualized_volatility, "Sharpe": r.sharpe_ratio,
        "Max Drawdown": r.max_drawdown, "Total Txn Cost Drag": r.total_transaction_costs, "# Rebalances": r.n_rebalances,
    } for r in bt.values()]).set_index("Strategy")
    st.dataframe(
        bt_df.style.format({"CAGR": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}", "Total Txn Cost Drag": "{:.3%}"}),
        width="stretch",
    )

# --------------------------------------------------------------------------
# Tab 6: Export PDF
# --------------------------------------------------------------------------
with tab_export:
    st.write(f"Generate a one-page client-ready PDF summarizing the **{objective}** portfolio.")
    client_name = st.text_input("Client name (optional, for the proposal header)", value="")
    if st.button("Generate PDF Proposal", type="primary"):
        with st.spinner("Building PDF..."):
            port_returns_pdf = portfolio_daily_returns(asset_returns, recommended.weights)
            report_pdf = build_risk_report(port_returns_pdf, risk_free_rate=risk_free_rate, benchmark_returns=benchmark_returns)
            stress_returns_pdf = load_stress_returns(tuple(tickers))
            stress_results_pdf = run_stress_tests(stress_returns_pdf, recommended.weights)
            out_path = OUTPUT_DIR / "portfolio_proposal.pdf"
            generate_portfolio_proposal_pdf(
                str(out_path), objective, recommended.weights, report_pdf, universe_labels,
                frontier=frontier,
                current_point={"volatility": recommended.volatility, "expected_return": recommended.expected_return},
                stress_results=stress_results_pdf, benchmark_name=benchmark_ticker,
                risk_free_rate=risk_free_rate, client_name=client_name or None,
            )
        with open(out_path, "rb") as f:
            st.download_button("Download Portfolio Proposal (PDF)", f, file_name="portfolio_proposal.pdf", mime="application/pdf")
        st.success("PDF generated.")

st.divider()
st.caption(
    "**Disclaimer:** This tool is for educational and research purposes only. It does not constitute "
    "investment advice, a recommendation, or an offer/solicitation to buy or sell any security. All "
    "figures are derived from historical data and modeling assumptions that may not hold in the future. "
    "Consult a licensed financial professional before making investment decisions."
)
