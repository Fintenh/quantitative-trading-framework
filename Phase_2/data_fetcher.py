"""
data_fetcher.py - Data downloading and preprocessing for Phase 2.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_price_data(tickers, lookback_days=252, buffer_days=400):
    """
    Download closing prices for a list of tickers.

    Args:
        tickers: List of Yahoo Finance symbols
        lookback_days: Number of trading days to return
        buffer_days: Calendar days to download (ensures enough data)

    Returns:
        DataFrame of closing prices, dates as index
    """
    end = datetime.now()
    start = end - timedelta(days=buffer_days)

    print(f"Downloading {len(tickers)} assets from {start.date()} to {end.date()}")

    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]

    # Handle single ticker case
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Clean column names
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Trim to exact lookback
    data = data.dropna(how='all').tail(lookback_days)

    print(f"  Downloaded: {len(data)} trading days, {len(data.columns)} assets")
    return data


def fetch_spy_data(start_date="2020-01-01", end_date=None):
    """Download SPY benchmark data and normalise to 100."""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
    spy = spy / spy.iloc[0] * 100  # Normalise to 100

    print(f"  SPY: {len(spy)} days, normalised to 100")
    return spy


def calculate_returns(price_data):
    """Calculate daily returns from price data."""
    return price_data.pct_change().dropna()


def calculate_annualised_stats(returns):
    """
    Calculate annualised expected returns and covariance matrix.

    Returns:
        tuple: (expected_returns, covariance_matrix, volatilities)
    """
    exp_ret = returns.mean() * 252
    cov = returns.cov() * 252
    vols = np.sqrt(np.diag(cov))
    return exp_ret, cov, vols