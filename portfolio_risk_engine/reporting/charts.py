"""Plotly chart builders for the Streamlit app (interactive, theme-consistent)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_TEMPLATE = "plotly_white"
_COLORWAY = [
    "#1f4e79", "#2e75b6", "#9dc3e6", "#548235", "#a9d18e",
    "#bf8f00", "#ffd966", "#833c00", "#c55a11", "#7030a0",
]


def efficient_frontier_chart(
    frontier: pd.DataFrame,
    highlighted: dict[str, dict] | None = None,
    asset_points: pd.DataFrame | None = None,
) -> go.Figure:
    """Frontier scatter (volatility vs. return) with optional highlighted
    portfolios (e.g. {"Max Sharpe": {"volatility": .., "expected_return": ..}})
    and individual asset risk/return points for context.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frontier["volatility"],
            y=frontier["target_return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color=_COLORWAY[1], width=3),
            customdata=frontier["sharpe_ratio"],
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.2f}<extra></extra>",
        )
    )

    if asset_points is not None:
        fig.add_trace(
            go.Scatter(
                x=asset_points["volatility"],
                y=asset_points["return"],
                mode="markers+text",
                text=asset_points.index,
                textposition="top center",
                marker=dict(size=7, color="#999999"),
                name="Individual Assets",
                hovertemplate="%{text}<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
            )
        )

    if highlighted:
        markers = {"Maximum Sharpe Ratio": "star", "Minimum Variance": "diamond", "Risk Parity": "square"}
        colors = {"Maximum Sharpe Ratio": "#c00000", "Minimum Variance": "#548235", "Risk Parity": "#bf8f00"}
        for name, point in highlighted.items():
            fig.add_trace(
                go.Scatter(
                    x=[point["volatility"]],
                    y=[point["expected_return"]],
                    mode="markers",
                    marker=dict(size=16, symbol=markers.get(name, "circle"), color=colors.get(name, "#000")),
                    name=name,
                    hovertemplate=f"{name}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
                )
            )

    fig.update_layout(
        template=_TEMPLATE,
        xaxis=dict(title="Annualized Volatility", tickformat=".0%"),
        yaxis=dict(title="Annualized Expected Return", tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
        height=480,
    )
    return fig


def allocation_pie_chart(weights: pd.Series, title: str = "Recommended Allocation", threshold: float = 0.01) -> go.Figure:
    """Pie chart of portfolio weights; assets below `threshold` are grouped as 'Other'."""
    weights = weights[weights > 1e-6].sort_values(ascending=False)
    small = weights[weights < threshold]
    shown = weights[weights >= threshold]
    if len(small) > 0:
        shown["Other"] = small.sum()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=shown.index,
                values=shown.values,
                hole=0.45,
                marker=dict(colors=_COLORWAY * 3),
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(template=_TEMPLATE, title=title, margin=dict(t=50, b=20, l=20, r=20), height=420)
    return fig


def cumulative_return_chart(series_dict: dict[str, pd.Series], title: str = "Cumulative Growth of $1") -> go.Figure:
    fig = go.Figure()
    for i, (name, series) in enumerate(series_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.values, mode="lines", name=name,
                line=dict(color=_COLORWAY[i % len(_COLORWAY)], width=2),
            )
        )
    fig.update_layout(
        template=_TEMPLATE, title=title,
        yaxis=dict(title="Growth of $1"), xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20), height=420,
    )
    return fig


def drawdown_chart(cumulative: pd.Series, title: str = "Drawdown") -> go.Figure:
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values, mode="lines", fill="tozeroy",
            line=dict(color="#c00000", width=1.5), name="Drawdown",
        )
    )
    fig.update_layout(
        template=_TEMPLATE, title=title,
        yaxis=dict(title="Drawdown", tickformat=".0%"), margin=dict(t=50, b=40, l=60, r=20), height=300,
    )
    return fig


def monte_carlo_fan_chart(mc_paths: np.ndarray, horizon_label: str) -> go.Figure:
    """Histogram of simulated cumulative return outcomes for one horizon."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=mc_paths, nbinsx=60, marker=dict(color=_COLORWAY[1]), name=horizon_label,
        )
    )
    p5, p50, p95 = np.percentile(mc_paths, [5, 50, 95])
    for val, label, color in [(p5, "5th pct", "#c00000"), (p50, "Median", "#000000"), (p95, "95th pct", "#548235")]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top")
    fig.update_layout(
        template=_TEMPLATE, title=f"Monte Carlo Outcome Distribution ({horizon_label})",
        xaxis=dict(title="Cumulative Return", tickformat=".0%"), yaxis=dict(title="Simulated Paths"),
        margin=dict(t=60, b=40, l=60, r=20), height=380, showlegend=False,
    )
    return fig


def stress_test_bar_chart(stress_results: list, title: str = "Historical Stress Test: Total Return") -> go.Figure:
    names = [r.window_name for r in stress_results]
    values = [r.total_return for r in stress_results]
    colors = ["#c00000" if v < 0 else "#548235" for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors, text=[f"{v:.1%}" for v in values], textposition="outside"))
    fig.update_layout(
        template=_TEMPLATE, title=title,
        yaxis=dict(title="Total Return", tickformat=".0%"), margin=dict(t=50, b=80, l=60, r=20), height=380,
    )
    return fig
