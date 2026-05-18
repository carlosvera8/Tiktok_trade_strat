import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATA_CACHE_DIR


def fetch_ohlcv(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker, caching to disk."""
    cache_path = DATA_CACHE_DIR / f"{ticker}_{start}_{end}.csv"

    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamps"
    df.dropna(inplace=True)

    df.to_csv(cache_path)
    return df


def fetch_all(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers in parallel, returning a dict keyed by ticker."""
    result = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
        futures = {executor.submit(fetch_ohlcv, ticker, start, end): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result[ticker] = future.result()
                print(f"  Loaded {ticker}: {len(result[ticker])} rows")
            except Exception as e:
                print(f"  WARNING: Could not fetch {ticker}: {e}")
    return result
