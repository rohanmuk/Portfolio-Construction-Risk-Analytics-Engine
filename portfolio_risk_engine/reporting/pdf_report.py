"""One-page PDF "portfolio proposal" -- an advisor-style client handout.

Built with matplotlib (GridSpec on a US-Letter figure) rather than a
separate PDF/HTML templating stack, so the only dependency is matplotlib
(already required for charting) and the whole report renders from a single
`savefig` call.
"""

from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Deliberately does NOT call matplotlib.use("Agg") here: fig.savefig(...) works
# under any backend, and forcing one globally at import time would silently
# break plt.show() for every OTHER consumer of this process (e.g. a notebook
# or Streamlit session that imports this module and then tries to render its
# own interactive matplotlib figures).

_NAVY = "#1f4e79"
_ACCENT = "#c00000"
_GREY = "#595959"
_PALETTE = ["#1f4e79", "#2e75b6", "#9dc3e6", "#548235", "#a9d18e", "#bf8f00", "#ffd966", "#833c00", "#c55a11", "#7030a0"]


def _fmt_pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


def generate_portfolio_proposal_pdf(
    output_path: str,
    portfolio_label: str,
    weights: pd.Series,
    risk_report,  # RiskReport
    universe_labels: dict[str, str],
    frontier: pd.DataFrame | None = None,
    current_point: dict | None = None,
    stress_results: list | None = None,
    benchmark_name: str = "SPY",
    risk_free_rate: float = 0.0,
    client_name: str | None = None,
    firm_name: str = "Portfolio Construction & Risk Analytics Engine",
) -> str:
    """Render a one-page advisor-style PDF proposal and save it to `output_path`."""
    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(
        nrows=6, ncols=2, figure=fig,
        height_ratios=[0.55, 1.55, 1.15, 0.85, 1.15, 0.55],
        width_ratios=[1.15, 0.85],
        hspace=0.7, wspace=0.55,
        left=0.07, right=0.95, top=0.95, bottom=0.05,
    )

    # --- Header ---
    # NOTE: text-only axes must have xlim/ylim fixed *and* autoscale disabled
    # before any Line2D (e.g. axhline) is added -- otherwise matplotlib
    # autoscales the axes to fit the line alone (Text artists are excluded
    # from autoscaling), which can collapse the y-range to a sliver around
    # the line and shove every text() call miles outside the visible figure.
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(-0.5, 1)
    ax_header.autoscale(False)
    ax_header.text(0, 0.85, firm_name, fontsize=15, fontweight="bold", color=_NAVY, va="top")
    ax_header.text(0, 0.35, "Portfolio Proposal", fontsize=11, color=_GREY, va="top")
    as_of = dt.date.today().isoformat()
    subtitle = f"Recommended Allocation: {portfolio_label}    |    As of {as_of}"
    if client_name:
        subtitle = f"Prepared for: {client_name}    |    " + subtitle
    ax_header.text(0, 0.0, subtitle, fontsize=9, color=_GREY, va="top")
    ax_header.axhline(y=-0.35, xmin=0, xmax=1, color=_NAVY, linewidth=1.5)

    # --- Allocation pie (left) ---
    ax_pie = fig.add_subplot(gs[1, 0])
    w = weights[weights > 1e-4].sort_values(ascending=False)
    labels = [f"{t} ({universe_labels.get(t, t)})" for t in w.index]
    ax_pie.pie(
        w.values, labels=None, autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
        colors=(_PALETTE * 3)[: len(w)], startangle=90,
        wedgeprops=dict(linewidth=0.5, edgecolor="white"),
        textprops=dict(fontsize=7),
    )
    ax_pie.set_title("Recommended Allocation", fontsize=10, fontweight="bold", color=_NAVY)
    ax_pie.legend(
        labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=6, frameon=False,
        labelspacing=0.4, ncol=2, columnspacing=1.0,
    )

    # --- Efficient frontier (right) ---
    ax_frontier = fig.add_subplot(gs[1, 1])
    if frontier is not None and len(frontier) > 0:
        ax_frontier.plot(frontier["volatility"], frontier["target_return"], color=_NAVY, linewidth=1.8)
        if current_point:
            ax_frontier.scatter(
                [current_point["volatility"]], [current_point["expected_return"]],
                color=_ACCENT, s=70, zorder=5, marker="*",
            )
        ax_frontier.set_xlabel("Volatility", fontsize=7)
        ax_frontier.set_ylabel("Expected Return", fontsize=7)
        ax_frontier.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_frontier.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_frontier.tick_params(labelsize=6)
        ax_frontier.spines[["top", "right"]].set_visible(False)
    ax_frontier.set_title("Efficient Frontier", fontsize=10, fontweight="bold", color=_NAVY)

    # --- Key metrics table ---
    ax_metrics = fig.add_subplot(gs[2, :])
    ax_metrics.axis("off")
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.autoscale(False)
    ax_metrics.set_title("Risk & Return Profile", fontsize=10, fontweight="bold", color=_NAVY, loc="left", pad=10)
    metrics = [
        ("Expected Annual Return", _fmt_pct(risk_report.annualized_return)),
        ("Annual Volatility", _fmt_pct(risk_report.annualized_volatility)),
        ("Sharpe Ratio", f"{risk_report.sharpe_ratio:.2f}"),
        ("Sortino Ratio", f"{risk_report.sortino_ratio:.2f}"),
        (f"Beta vs. {benchmark_name}", f"{risk_report.beta:.2f}" if risk_report.beta is not None else "n/a"),
        ("Max Drawdown", _fmt_pct(risk_report.max_drawdown)),
        ("95% Daily VaR", _fmt_pct(risk_report.var_95)),
        ("95% Daily CVaR", _fmt_pct(risk_report.cvar_95)),
        ("99% Daily VaR", _fmt_pct(risk_report.var_99)),
        ("99% Daily CVaR", _fmt_pct(risk_report.cvar_99)),
        ("Downside Deviation", _fmt_pct(risk_report.downside_deviation)),
        ("Risk-Free Rate Assumed", _fmt_pct(risk_free_rate)),
    ]
    n_cols = 4
    n_rows = int(np.ceil(len(metrics) / n_cols))
    row_height = 0.95 / n_rows
    for i, (label, value) in enumerate(metrics):
        col = i % n_cols
        row = i // n_cols
        x = col / n_cols
        y = 0.92 - row * row_height
        ax_metrics.text(x, y, label, fontsize=7.5, color=_GREY, va="top")
        ax_metrics.text(x, y - row_height * 0.45, value, fontsize=12, color=_NAVY, fontweight="bold", va="top")

    # --- Stress test bar chart ---
    ax_stress = fig.add_subplot(gs[3, :])
    if stress_results:
        names = [r.window_name for r in stress_results]
        values = [r.total_return for r in stress_results]
        colors = [_ACCENT if v < 0 else "#548235" for v in values]
        bars = ax_stress.bar(names, values, color=colors, width=0.5)
        # Extra headroom so the value label clears the bar end without
        # colliding with the axis line / category tick labels below it.
        y_min, y_max = min(0, *values), max(0, *values)
        margin = (y_max - y_min) * 0.3 or 0.05
        ax_stress.set_ylim(y_min - margin, y_max + margin)
        for b, v in zip(bars, values):
            offset = margin * 0.15
            ax_stress.text(b.get_x() + b.get_width() / 2, v + offset if v >= 0 else v - offset,
                            _fmt_pct(v, 0), ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax_stress.axhline(0, color="black", linewidth=0.6)
        ax_stress.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_stress.tick_params(labelsize=7)
        ax_stress.spines[["top", "right"]].set_visible(False)
    ax_stress.set_title("Historical Stress Scenarios", fontsize=10, fontweight="bold", color=_NAVY, loc="left")

    # --- Holdings table ---
    ax_holdings = fig.add_subplot(gs[4, :])
    ax_holdings.axis("off")
    ax_holdings.set_xlim(0, 1)
    ax_holdings.set_ylim(0, 1)
    ax_holdings.autoscale(False)
    ax_holdings.set_title("Portfolio Holdings", fontsize=10, fontweight="bold", color=_NAVY, loc="left", pad=10)
    top_holdings = w.head(10)
    col_x = [0.0, 0.5]
    half = int(np.ceil(len(top_holdings) / 2))
    for i, (ticker, weight) in enumerate(top_holdings.items()):
        col = 0 if i < half else 1
        row = i if col == 0 else i - half
        y = 0.85 - row * 0.18
        name = universe_labels.get(ticker, ticker)
        ax_holdings.text(col_x[col], y, f"{ticker}", fontsize=8, fontweight="bold", color=_NAVY, va="top")
        ax_holdings.text(col_x[col] + 0.08, y, f"{name}", fontsize=7.5, color=_GREY, va="top")
        ax_holdings.text(col_x[col] + 0.42, y, _fmt_pct(weight), fontsize=8, color=_NAVY, va="top", ha="right")

    # --- Footer / disclaimer ---
    ax_footer = fig.add_subplot(gs[5, :])
    ax_footer.axis("off")
    ax_footer.set_xlim(0, 1)
    ax_footer.set_ylim(0, 1)
    ax_footer.autoscale(False)
    ax_footer.axhline(y=1.0, xmin=0, xmax=1, color="#bfbfbf", linewidth=0.8)
    disclaimer = (
        "For educational and research purposes only. This is not investment advice, a recommendation, "
        "or an offer/solicitation to buy or sell any security. Past performance does not guarantee future "
        "results. Estimates are based on historical data and modeling assumptions that may not hold going "
        "forward. Consult a licensed financial professional before making investment decisions."
    )
    ax_footer.text(0, 0.65, disclaimer, fontsize=6, color=_GREY, va="top", wrap=True)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    return output_path
