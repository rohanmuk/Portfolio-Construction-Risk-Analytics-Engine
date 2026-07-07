"""Default configuration for the Portfolio Construction & Risk Analytics Engine."""

from __future__ import annotations

from dataclasses import dataclass, field

# Default 18-asset multi-asset universe: US equity, international equity,
# bonds, REITs, and commodities. Tickers are liquid ETFs with long histories
# so a 10+ year backtest is possible.
DEFAULT_UNIVERSE: dict[str, str] = {
    # US Equity
    "SPY": "US Large Cap (S&P 500)",
    "IWM": "US Small Cap (Russell 2000)",
    "QQQ": "US Large Cap Growth (Nasdaq 100)",
    "VTV": "US Large Cap Value",
    # International Equity
    "EFA": "Developed ex-US Equity",
    "VWO": "Emerging Markets Equity",
    "VGK": "European Equity",
    "EWJ": "Japan Equity",
    # Bonds
    "AGG": "US Aggregate Bond",
    "TLT": "US Long-Term Treasury",
    "IEF": "US 7-10Y Treasury",
    "LQD": "US Investment Grade Corporate Bond",
    "HYG": "US High Yield Corporate Bond",
    "BNDX": "International Bond (hedged)",
    # REITs
    "VNQ": "US REITs",
    "RWX": "International REITs",
    # Commodities
    "GLD": "Gold",
    "DBC": "Broad Commodities",
}

BENCHMARK_TICKER = "SPY"
RISK_FREE_TICKER = "^IRX"  # 13-week T-bill discount rate, annualized %

DEFAULT_START_DATE = "2013-01-01"
# Stress testing needs to reach back through the 2008 GFC. Some universe
# ETFs (e.g. BNDX, inception 2013) postdate this window entirely -- the
# stress-test module excludes and reweights around such assets per-window
# rather than forcing the whole analysis panel to start this early.
STRESS_TEST_START_DATE = "2005-01-01"
TRADING_DAYS_PER_YEAR = 252

DEFAULT_MAX_WEIGHT = 0.35  # per-asset cap for constrained optimizations
DEFAULT_MIN_WEIGHT = 0.0  # long-only floor

VAR_CONFIDENCE_LEVELS = (0.95, 0.99)

STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008 Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "March 2020 COVID Crash": ("2020-02-19", "2020-03-23"),
    "2022 Rate-Hike Drawdown": ("2022-01-03", "2022-10-14"),
}

FAMA_FRENCH_3F_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FAMA_FRENCH_5F_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

DEFAULT_TRANSACTION_COST_BPS = 10.0
MONTE_CARLO_PATHS = 10_000
MONTE_CARLO_HORIZONS_YEARS = (1, 5)


@dataclass
class OptimizationConstraints:
    """Configurable box constraints applied to portfolio weights."""

    long_only: bool = True
    max_weight: float = DEFAULT_MAX_WEIGHT
    min_weight: float = DEFAULT_MIN_WEIGHT

    def bounds(self, n_assets: int) -> list[tuple[float, float]]:
        lower = self.min_weight if self.long_only else -self.max_weight
        return [(lower, self.max_weight)] * n_assets
