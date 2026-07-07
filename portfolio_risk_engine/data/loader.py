"""Price data acquisition with local on-disk caching.

Downloads adjusted daily close prices via yfinance and caches them as
parquet (falling back to CSV if pyarrow is unavailable) so repeated runs of
the app/notebooks don't re-hit the network unless the cache is stale or
missing tickers.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_PARQUET_PATH = CACHE_DIR / "adj_close.parquet"
_CSV_PATH = CACHE_DIR / "adj_close.csv"
_META_PATH = CACHE_DIR / "adj_close.meta.txt"


def _read_cache() -> pd.DataFrame | None:
    if _PARQUET_PATH.exists():
        try:
            return pd.read_parquet(_PARQUET_PATH)
        except Exception:  # pragma: no cover - pyarrow missing/corrupt cache
            logger.warning("Failed to read parquet cache, trying CSV.")
    if _CSV_PATH.exists():
        return pd.read_csv(_CSV_PATH, index_col=0, parse_dates=True)
    return None


def _write_cache(prices: pd.DataFrame) -> None:
    try:
        prices.to_parquet(_PARQUET_PATH)
    except Exception:  # pragma: no cover - pyarrow missing
        logger.warning("Parquet engine unavailable, falling back to CSV cache.")
    prices.to_csv(_CSV_PATH)


def _cache_is_fresh(max_age_days: int = 1) -> bool:
    if not _META_PATH.exists():
        return False
    last_run = dt.datetime.fromisoformat(_META_PATH.read_text().strip())
    return (dt.datetime.now() - last_run) < dt.timedelta(days=max_age_days)


def get_adjusted_close(
    tickers: list[str],
    start: str,
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted daily close prices, one column per ticker.

    Uses a local parquet/CSV cache keyed on the full requested price panel.
    If the cache exists, is fresh (<1 day old), and already contains every
    requested ticker with data at least as far back as `start`, it is reused
    as-is. Otherwise the full ticker set is (re)downloaded via yfinance so the
    cache stays a single consistent panel.
    """
    end = end or dt.date.today().isoformat()
    tickers = sorted(set(tickers))

    if not force_refresh:
        cached = _read_cache()
        if (
            cached is not None
            and set(tickers).issubset(cached.columns)
            and cached.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=5)
            and _cache_is_fresh()
        ):
            logger.info("Using cached price data (%s tickers).", len(tickers))
            return cached.loc[cached.index >= pd.Timestamp(start), tickers]

    logger.info("Downloading %s tickers from yfinance (%s -> %s).", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # adjusts OHLC for splits/dividends; 'Close' is the adjusted price
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for tickers={tickers}. "
            "Check ticker symbols and network connectivity."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.levels[0]})
    else:
        # Single ticker request collapses to a flat column index.
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index().dropna(how="all")
    missing = set(tickers) - set(prices.columns)
    if missing:
        logger.warning("No data returned for tickers: %s", sorted(missing))

    _write_cache(prices)
    _META_PATH.write_text(dt.datetime.now().isoformat())

    return prices


def align_and_clean(prices: pd.DataFrame, min_history_frac: float = 0.5) -> pd.DataFrame:
    """Drop assets with insufficient history and forward-fill small gaps.

    Assets with less than `min_history_frac` of the panel's total observed
    date range are dropped (e.g. an ETF that launched partway through the
    window), then any remaining internal gaps are forward-filled (holidays /
    listing mismatches across exchanges) and leading NaNs trimmed.
    """
    threshold = int(len(prices) * min_history_frac)
    kept = prices.dropna(axis=1, thresh=threshold)
    dropped = set(prices.columns) - set(kept.columns)
    if dropped:
        logger.warning("Dropping tickers with insufficient history: %s", sorted(dropped))
    cleaned = kept.ffill().dropna(how="any")
    return cleaned


def get_raw_daily_returns(
    tickers: list[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Per-asset daily returns from an *unaligned* price panel.

    Unlike `align_and_clean`, this does not drop rows/columns to force a
    jointly-complete panel -- each column is forward-filled independently
    (safe: never borrows from other tickers) and pct_change'd, so tickers
    that IPO'd partway through `start`..`end` simply have leading NaNs.
    Intended for stress-window replay, where we want the longest possible
    history per asset and handle missing assets window-by-window instead of
    truncating the whole universe to the latest common inception date.
    """
    prices = get_adjusted_close(tickers, start=start, end=end)
    return prices.ffill().pct_change()
