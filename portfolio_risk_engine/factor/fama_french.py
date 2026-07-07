"""Fama-French factor regression: download Ken French's daily factor data and
regress asset excess returns against it to report factor loadings and R^2.

The 3-factor model (Mkt-RF, SMB, HML) and 5-factor model (adds RMW, CMA) are
both supported. Factor files are downloaded once from Ken French's data
library and cached locally as parquet/CSV alongside the price cache.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import statsmodels.api as sm

from portfolio_risk_engine.config import FAMA_FRENCH_3F_URL, FAMA_FRENCH_5F_URL

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FACTOR_COLUMNS = {
    "3F": ["Mkt-RF", "SMB", "HML", "RF"],
    "5F": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
}


def _parse_ff_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Parse a Ken French daily-factor CSV (extracted from the zip).

    The file has a text preamble, a header row, daily data rows keyed by an
    8-digit YYYYMMDD integer, then trailing copyright/notes text. We locate
    the data block by scanning for the first and last rows whose first field
    is an 8-digit date.
    """
    text = raw_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    data_start = None
    for i, line in enumerate(lines):
        first_field = line.split(",")[0].strip()
        if len(first_field) == 8 and first_field.isdigit():
            data_start = i
            break
    if data_start is None:
        raise ValueError("Could not locate start of data block in Fama-French CSV.")

    data_lines = []
    for line in lines[data_start:]:
        first_field = line.split(",")[0].strip()
        if len(first_field) == 8 and first_field.isdigit():
            data_lines.append(line)
        else:
            break

    header_line = lines[data_start - 1]
    columns = ["Date"] + [c.strip() for c in header_line.split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None, names=columns)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date")
    # Values are in percent (e.g. 0.05 == 0.05%); convert to decimal returns.
    return df.astype(float) / 100.0


def download_fama_french_factors(model: str = "3F", force_refresh: bool = False) -> pd.DataFrame:
    """Download (or load from cache) daily Fama-French factor returns.

    `model` is "3F" (Mkt-RF, SMB, HML, RF) or "5F" (adds RMW, CMA).
    """
    model = model.upper()
    if model not in _FACTOR_COLUMNS:
        raise ValueError(f"model must be one of {list(_FACTOR_COLUMNS)}, got {model!r}")

    cache_path = CACHE_DIR / f"fama_french_{model}.parquet"
    if cache_path.exists() and not force_refresh:
        logger.info("Using cached Fama-French %s factor data.", model)
        return pd.read_parquet(cache_path)

    url = FAMA_FRENCH_3F_URL if model == "3F" else FAMA_FRENCH_5F_URL
    logger.info("Downloading Fama-French %s factors from %s", model, url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw_bytes = zf.read(csv_name)

    factors = _parse_ff_csv(raw_bytes)
    factors.to_parquet(cache_path)
    return factors


@dataclass
class FactorRegressionResult:
    asset: str
    alpha: float  # annualized intercept (unexplained excess return)
    betas: dict  # factor name -> loading
    r_squared: float
    n_obs: int


def run_factor_regression(
    asset_daily_returns: pd.DataFrame,
    factors: pd.DataFrame,
    model: str = "3F",
) -> list[FactorRegressionResult]:
    """Regress each asset's excess daily return on the Fama-French factors.

    excess_return_i = alpha_i + sum_k(beta_ik * factor_k) + epsilon_i

    `alpha` is annualized (daily intercept * 252); betas are the raw daily
    factor loadings (dimensionless, comparable across assets/horizons).
    """
    factor_names = [c for c in _FACTOR_COLUMNS[model.upper()] if c != "RF"]
    aligned = asset_daily_returns.join(factors, how="inner").dropna()

    results = []
    for asset in asset_daily_returns.columns:
        excess = aligned[asset] - aligned["RF"]
        X = sm.add_constant(aligned[factor_names])
        model_fit = sm.OLS(excess, X).fit()
        results.append(
            FactorRegressionResult(
                asset=asset,
                alpha=float(model_fit.params["const"] * 252),
                betas={f: float(model_fit.params[f]) for f in factor_names},
                r_squared=float(model_fit.rsquared),
                n_obs=int(model_fit.nobs),
            )
        )
    return results


def factor_regression_table(results: list[FactorRegressionResult]) -> pd.DataFrame:
    """Tidy the list of FactorRegressionResult into a single display DataFrame."""
    rows = []
    for r in results:
        row = {"Asset": r.asset, "Annualized Alpha": r.alpha, **r.betas, "R-squared": r.r_squared}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Asset")
