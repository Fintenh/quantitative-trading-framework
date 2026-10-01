"""
final_performance_tracker.py - Tracks portfolio performance from the injection date onwards.

Uses previous-day closing prices for all updates. Implements rolling optimisation,
Kelly cash allocation with sign-change detection, take-profit vs SPY, time/drift
rebalancing, and cash interest. No look-ahead bias.

Portfolio initialisation:
- Original_18 / Optimal_Standard: compute optimal weights on Day 0.
- Optimal_Ethical: reads the state from portfolio_log.csv to stay in sync with the live engine.
- ML_Strategy: generates weights up to the injection date (cached).

Universe-agnostic: reads ALL asset values from the log, not just TICKERS.

Data fetching uses a persistent rolling cache (logs/price_history.csv). Only
successful, complete tickers are cached; failed tickers are retried next run.
"""

import os
import sys
import time

# --- Setup paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, script_dir)
sys.path.insert(0, parent_dir)
for phase in ["Phase_2", "Phase_3", "Phase_5"]:
    path = os.path.join(parent_dir, phase)
    if os.path.exists(path):
        sys.path.insert(0, path)

# --- Try to import ML module ---
ML_AVAILABLE = False
try:
    from final_ml_research import generate_ml_weights_for_date
    ML_AVAILABLE = True
    print("ML module loaded successfully.")
except ImportError as e:
    print(f"Warning: Could not import final_ml_research: {e}")
    print("ML strategy will be skipped.")

# --- Standard imports ---
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pickle

from ethical_universe import ORIGINAL_UNIVERSE
from final_config import (
    INITIAL_CAPITAL, RISK_FREE_RATE,
    CASH_INTEREST_RATE, CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY,
    KELLY_BASE_CAP, KELLY_MAX_CAP,
    ROUND_TRIP_COST_PCT,
    TICKERS,
    LOOKBACK_DAYS,
    REBALANCE_MIN_DAYS,
    REBALANCE_MAX_DAYS,
    DRIFT_THRESHOLD,
    RELATIVE_TAKE_PROFIT_PCT,
    KELLY_LOOKBACK,
)
from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO,
    ORIGINAL_PARAMS,
    OPTIMAL_STANDARD_PARAMS,
)

from data_fetcher import calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

# --- Constants ---
UNIVERSE_COLORS = {
    "Original_18": '#1f77b4',
    "Optimal_Standard": '#ff7f0e',
    "Optimal_Ethical": '#2ca02c',
    "ML_Strategy": '#9467bd',
}
SPY_COLOR = '#d62728'

FINAL_ENGINE_DIR = os.path.join(script_dir)
FINAL_LOGS_DIR = os.path.join(FINAL_ENGINE_DIR, "logs")
FINAL_FIGURES_DIR = os.path.join(FINAL_ENGINE_DIR, "figures")
os.makedirs(FINAL_LOGS_DIR, exist_ok=True)
os.makedirs(FINAL_FIGURES_DIR, exist_ok=True)

MAX_CACHE_ROWS = 1500
MIN_CACHE_ROWS = 50
MAX_CACHE_STALENESS_DAYS = 3
SPARSE_FIRST_ROW_FRACTION = 0.5

print(f"Working directory: {os.getcwd()}")
print(f"Logs directory: {FINAL_LOGS_DIR}")
print(f"Figures directory: {FINAL_FIGURES_DIR}")

# ---------------------------------------------------------------------------
# Currency / FX handling
# ---------------------------------------------------------------------------

_FALLBACK_FX_RATES = {
    'USD': 0.79, 'EUR': 0.85, 'DKK': 0.115, 'GBP': 1.0,
    'SEK': 0.075, 'NOK': 0.072, 'CHF': 0.88, 'CAD': 0.58,
    'AUD': 0.52, 'JPY': 0.0052, 'CNY': 0.11, 'HKD': 0.10,
    'SGD': 0.59, 'INR': 0.0095,
}

_fx_series_cache = {}
_ticker_currency_cache = {}


def get_ticker_currency(ticker):
    """Return the native trading currency for a ticker (cached)."""
    if ticker in _ticker_currency_cache:
        return _ticker_currency_cache[ticker]

    try:
        info = yf.Ticker(ticker).info
        ccy = info.get('currency')
        if ccy:
            _ticker_currency_cache[ticker] = ccy
            return ccy
    except Exception:
        pass

    suffix_map = {
        '.L': 'GBP', '.CO': 'DKK', '.MC': 'EUR', '.DE': 'EUR',
        '.PA': 'EUR', '.AS': 'EUR', '.OL': 'NOK', '.ST': 'SEK',
        '.MI': 'EUR', '.LS': 'EUR', '.VI': 'EUR', '.HE': 'EUR',
    }
    for suffix, ccy in suffix_map.items():
        if ticker.endswith(suffix):
            _ticker_currency_cache[ticker] = ccy
            return ccy

    _ticker_currency_cache[ticker] = 'USD'
    return 'USD'


def _ffill_value(series, date):
    """Look up the most recent value at or before `date`."""
    if series is None or len(series) == 0:
        return None
    date_ts = pd.Timestamp(date)
    idx = series.index[series.index <= date_ts]
    if len(idx) == 0:
        return float(series.iloc[0])
    return float(series.loc[idx[-1]])


def prefetch_fx_rates(tickers, start_date, end_date):
    """Pre-fetch FX series (GBP per unit) for every non-GBP currency in `tickers`."""
    currencies = set()
    for t in tickers:
        ccy = get_ticker_currency(t)
        if ccy != 'GBP':
            currencies.add(ccy)

    for ccy in sorted(currencies):
        if ccy in _fx_series_cache:
            continue
        try:
            pair = f"{ccy}GBP=X"
            start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
            end_str = (end_date + timedelta(days=2)).strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)
            data = yf.download(pair, start=start_str, end=end_str,
                               progress=False, auto_adjust=False)
            if len(data) == 0:
                raise ValueError(f"No FX data for {pair}")
            if 'Close' in data.columns:
                series = data['Close']
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
            else:
                series = data.iloc[:, 0]
            series = series.dropna()
            if len(series) == 0:
                raise ValueError(f"Empty FX series for {pair}")
            _fx_series_cache[ccy] = series
            print(f"   FX {ccy}/GBP: {len(series)} points, latest = {float(series.iloc[-1]):.6f}")
        except Exception as e:
            fallback = _FALLBACK_FX_RATES.get(ccy, 1.0)
            flat = pd.Series([fallback], index=[pd.Timestamp(start_date)])
            _fx_series_cache[ccy] = flat
            print(f"   FX {ccy}/GBP: live fetch failed ({e}); using fallback {fallback}")


def get_fx_rate_on_date(currency, date):
    """Return the FX rate (GBP per unit) on or before a given date."""
    if currency == 'GBP':
        return 1.0
    series = _fx_series_cache.get(currency)
    if series is None or len(series) == 0:
        return _FALLBACK_FX_RATES.get(currency, 1.0)
    val = _ffill_value(series, date)
    if val is None or val <= 0:
        return _FALLBACK_FX_RATES.get(currency, 1.0)
    return val


def get_fx_ratio(currency, day):
    """Return FX(day-1) / FX(day-2), mirroring the previous-day-close convention."""
    if currency == 'GBP':
        return 1.0
    series = _fx_series_cache.get(currency)
    if series is None or len(series) == 0:
        return 1.0
    d1 = day - timedelta(days=1)
    d2 = day - timedelta(days=2)
    v1 = _ffill_value(series, d1)
    v2 = _ffill_value(series, d2)
    if v1 is None or v2 is None or v1 <= 0 or v2 <= 0:
        return 1.0
    return v1 / v2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_all_days_between(start_date, end_date):
    """Return every calendar day between two dates, inclusive."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates

def daily_cash_rate(annual):
    """Convert an annual cash rate to a daily compounded rate."""
    return (1 + annual) ** (1 / 365) - 1 if annual > 0 else 0.0

def get_all_shifted_prices(date, shifted_df):
    """Return prices for a date from the shifted calendar."""
    prices = {}
    date_ts = pd.Timestamp(date)
    if date_ts in shifted_df.index:
        row = shifted_df.loc[date_ts]
        for t in shifted_df.columns:
            prices[t] = float(row[t]) if row[t] > 0 else 0.0
    else:
        idx = shifted_df.index[shifted_df.index <= date_ts]
        if len(idx) > 0:
            row = shifted_df.loc[idx[-1]]
            for t in shifted_df.columns:
                prices[t] = float(row[t]) if row[t] > 0 else 0.0
        else:
            for t in shifted_df.columns:
                prices[t] = 0.0
    return prices

def load_ml_weights_for_date(date):
    """Load cached ML weights for a date, or generate and cache them."""
    cache_file = os.path.join(FINAL_LOGS_DIR, f"ml_weights_{date.strftime('%Y-%m-%d')}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            weights = pickle.load(f)
        print(f"   Loaded cached ML weights for {date.strftime('%Y-%m-%d')}")
        return weights

    if not ML_AVAILABLE:
        print("   ML module not available; cannot generate weights.")
        return None

    print(f"   Generating ML weights for {date.strftime('%Y-%m-%d')} (this may take a few minutes)...")
    try:
        weights = generate_ml_weights_for_date(date)
        with open(cache_file, 'wb') as f:
            pickle.dump(weights, f)
        print(f"   Cached ML weights to {cache_file}")
        return weights
    except Exception as e:
        print(f"   Failed to generate ML weights: {e}")
        return None

# ---------------------------------------------------------------------------
# Persistent price history
# ---------------------------------------------------------------------------

def get_price_history_path():
    """Path to the persistent price history file."""
    return os.path.join(FINAL_LOGS_DIR, "price_history.csv")

def load_price_history():
    """Load the persistent price history, or None if missing."""
    path = get_price_history_path()
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if len(df) == 0:
            return None
        df = df[sorted(df.columns)]
        print(f"   Loaded price history ({len(df.columns)} tickers, {len(df)} rows, "
              f"latest: {df.index.max().date()})")

        if len(df) >= 2:
            first_filled = df.iloc[0].notna().sum()
            last_filled = df.iloc[-1].notna().sum()
            if last_filled > 0 and first_filled < last_filled * SPARSE_FIRST_ROW_FRACTION:
                print(f"   NOTE: cache starts sparse ({first_filled}/{len(df.columns)} tickers "
                      f"at {df.index[0].date()}, {last_filled}/{len(df.columns)} at "
                      f"{df.index[-1].date()}). Consider rebuilding the cache.")

        return df
    except Exception as e:
        print(f"   Could not load price history: {e}")
        return None

def save_price_history(df):
    """Save the persistent price history (sorted columns)."""
    path = get_price_history_path()
    try:
        df = df[sorted(df.columns)]
        df.index.name = "Date"
        total_cells = df.shape[0] * df.shape[1]
        filled_cells = int(df.notna().sum().sum())
        fill_pct = (filled_cells / total_cells * 100) if total_cells > 0 else 0.0
        df.to_csv(path)
        print(f"   Saved price history: {len(df.columns)} tickers, {len(df)} rows "
              f"({df.index.min().date()} to {df.index.max().date()}, "
              f"{fill_pct:.1f}% filled)")
    except Exception as e:
        print(f"   Could not save price history: {e}")

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _bulk_fetch_with_fallback(tickers_to_fetch, fetch_start, end_date_exclusive, target_date, min_rows=1):
    """
    Bulk-download a batch, then fall back to individual downloads for any
    that fail or return incomplete data. `min_rows` should be MIN_CACHE_ROWS
    for full-history fetches and 1 for gap-fills.
    """
    if not tickers_to_fetch:
        return {}

    all_data = {}

    try:
        bulk = yf.download(
            tickers_to_fetch,
            start=fetch_start,
            end=end_date_exclusive,
            progress=False,
            group_by='ticker',
            threads=True,
            auto_adjust=False
        )

        if isinstance(bulk.columns, pd.MultiIndex):
            for t in tickers_to_fetch:
                try:
                    if t in bulk.columns.get_level_values(0):
                        series = bulk[t]['Close'].dropna()
                        if len(series) > 0:
                            all_data[t] = series
                except Exception:
                    continue
        else:
            if len(tickers_to_fetch) == 1:
                series = bulk['Close'].dropna() if 'Close' in bulk.columns else bulk.iloc[:, 0].dropna()
                if len(series) > 0:
                    all_data[tickers_to_fetch[0]] = series
    except Exception as e:
        print(f"   Bulk download failed: {e}")

    still_missing = []
    for t in tickers_to_fetch:
        if t not in all_data or len(all_data[t]) == 0:
            still_missing.append(t)
        else:
            ticker_last = all_data[t].index.max().date()
            if (target_date - ticker_last).days > 3:
                still_missing.append(t)
                print(f"   {t}: bulk response only reaches {ticker_last} (target: {target_date})")

    if still_missing:
        print(f"   Fallback: downloading {len(still_missing)} tickers individually...")
        failed_tickers = []

        for t in still_missing:
            success = False
            for attempt in range(3):
                try:
                    data = yf.download(t, start=fetch_start, end=end_date_exclusive,
                                       progress=False, auto_adjust=False)
                    if len(data) == 0:
                        if attempt < 2:
                            time.sleep(1)
                        continue
                    series = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
                    if len(series) > 0 and len(series) >= min_rows:
                        ticker_last = series.index.max().date()
                        if (target_date - ticker_last).days <= 3:
                            all_data[t] = series.dropna()
                            success = True
                            break
                except Exception:
                    if attempt < 2:
                        time.sleep(1)
                    continue

            if not success:
                failed_tickers.append(t)

        if failed_tickers:
            print(f"   WARNING: Failed to download {len(failed_tickers)} tickers: {failed_tickers}")
            print(f"   These will NOT be cached and will be retried next run.")

    valid = {}
    for t, s in all_data.items():
        if len(s) < min_rows:
            continue
        ticker_last = s.index.max().date()
        if (target_date - ticker_last).days > 3:
            continue
        valid[t] = s

    return valid

def fetch_price_data_for_universe(tickers, lookback_days, injection_date):
    """
    Fetch prices using the persistent rolling cache. Only fetches the
    missing data (full history for new tickers, gap-fill for stale ones).
    """
    yesterday = datetime.now().date() - timedelta(days=1)
    end_date_exclusive = yesterday + timedelta(days=1)

    full_history_start = injection_date - timedelta(days=lookback_days * 2)

    history_df = load_price_history()

    if history_df is not None and len(history_df) > 0:
        days_stale = (yesterday - history_df.index.max().date()).days
        if days_stale > MAX_CACHE_STALENESS_DAYS:
            print(f"   WARNING: cache is {days_stale} days stale "
                  f"(latest: {history_df.index.max().date()}, target: {yesterday})")

    new_tickers = []
    stale_tickers = []
    current_tickers = []

    for t in tickers:
        if history_df is None or t not in history_df.columns:
            new_tickers.append(t)
            continue
        series = history_df[t].dropna()
        if len(series) < MIN_CACHE_ROWS:
            new_tickers.append(t)
            continue
        ticker_max = series.index.max().date()
        if ticker_max >= yesterday:
            current_tickers.append(t)
        elif (yesterday - ticker_max).days > 90:
            new_tickers.append(t)
        else:
            stale_tickers.append(t)

    total_to_fetch = len(new_tickers) + len(stale_tickers)

    if total_to_fetch == 0:
        print(f"\nAll {len(tickers)} tickers up to date in cache")
        if history_df is None:
            return pd.DataFrame()
        for t in tickers:
            if t not in history_df.columns:
                history_df[t] = 0.0
        return history_df[sorted(tickers)].copy()

    print(f"\nFetching {total_to_fetch} tickers "
          f"({len(new_tickers)} new/full-history, {len(stale_tickers)} gap-fill, "
          f"{len(current_tickers)} cached)")

    new_data = {}
    if new_tickers:
        print(f"\n   Batch 1: full history for {len(new_tickers)} new tickers from {full_history_start}")
        new_data = _bulk_fetch_with_fallback(
            new_tickers, full_history_start, end_date_exclusive, yesterday,
            min_rows=MIN_CACHE_ROWS,
        )

    stale_data = {}
    if stale_tickers:
        gap_start = min(
            history_df[t].dropna().index.max().date() + timedelta(days=1)
            for t in stale_tickers
        )
        print(f"\n   Batch 2: gap-fill for {len(stale_tickers)} tickers from {gap_start}")
        stale_data = _bulk_fetch_with_fallback(
            stale_tickers, gap_start, end_date_exclusive, yesterday,
            min_rows=1,
        )

    all_fetched = {**new_data, **stale_data}

    if not all_fetched:
        print("   No valid new data to append to cache")
        if history_df is not None:
            for t in tickers:
                if t not in history_df.columns:
                    history_df[t] = 0.0
            return history_df[sorted(tickers)].copy()
        return pd.DataFrame()

    all_dates = pd.DatetimeIndex([])
    for s in all_fetched.values():
        all_dates = all_dates.union(s.index)

    business_days = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='B')
    new_df = pd.DataFrame(index=business_days)
    for t in sorted(all_fetched.keys()):
        new_df[t] = all_fetched[t].reindex(business_days).ffill().fillna(0)

    if history_df is not None:
        combined = history_df.join(new_df, how='outer', rsuffix='_new')
        for col in new_df.columns:
            new_col = f"{col}_new"
            if new_col in combined.columns:
                if col in combined.columns:
                    combined[col] = combined[new_col].combine_first(combined[col])
                else:
                    combined[col] = combined[new_col]
                combined = combined.drop(columns=[new_col])
        history_df = combined
    else:
        history_df = new_df

    history_df = history_df[sorted(history_df.columns)]

    if len(history_df) > MAX_CACHE_ROWS:
        history_df = history_df.iloc[-MAX_CACHE_ROWS:]

    save_price_history(history_df)

    for t in tickers:
        if t not in history_df.columns:
            history_df[t] = 0.0

    return history_df[sorted(tickers)].copy()

def download_spy(start_date, end_date, injection_date):
    """Fetch SPY via the persistent cache and apply the previous-day shift."""
    print("   Downloading SPY...")
    try:
        spy_df = fetch_price_data_for_universe(["SPY"], 400, injection_date)

        if spy_df is not None and "SPY" in spy_df.columns and spy_df["SPY"].sum() > 0:
            spy = spy_df["SPY"].copy()
        else:
            raise ValueError("SPY not available in cache")
    except Exception:
        try:
            end_dt = datetime.now() + timedelta(days=5)
            spy_raw = yf.download("SPY", start=start_date, end=end_dt.strftime("%Y-%m-%d"),
                                  progress=False, auto_adjust=False)["Close"]
            if len(spy_raw) == 0:
                spy_raw = yf.download("^GSPC", start=start_date, end=end_dt.strftime("%Y-%m-%d"),
                                      progress=False, auto_adjust=False)["Close"]
            if len(spy_raw) > 0:
                if isinstance(spy_raw, pd.DataFrame):
                    spy_raw = spy_raw.iloc[:, 0]
                spy = spy_raw
            else:
                dates = pd.date_range(start=start_date, end=end_date, freq='B')
                return pd.Series([100.0] * len(dates), index=dates)
        except Exception as e:
            print(f"   Error: {e}")
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            return pd.Series([100.0] * len(dates), index=dates)

    business_days = pd.date_range(start=spy.index.min(), end=spy.index.max(), freq='B')
    spy = spy.reindex(business_days).ffill().bfill()

    spy_shifted = spy.copy()
    spy_shifted.index = spy.index + pd.Timedelta(days=1)

    today = datetime.now().date()
    spy_shifted = spy_shifted[spy_shifted.index <= pd.Timestamp(today)]
    spy_shifted = spy_shifted[spy_shifted.index >= pd.Timestamp(injection_date)]

    print(f"   SPY: {len(spy_shifted)} days (previous-day close)")
    return spy_shifted

# ---------------------------------------------------------------------------
# Portfolio calculations
# ---------------------------------------------------------------------------

def calculate_drift(current, target):
    """Return the largest absolute drift and the asset that has it."""
    max_drift, max_asset = 0.0, None
    for t in current:
        drift = abs(current.get(t, 0.0) - target.get(t, 0.0))
        if drift > max_drift:
            max_drift, max_asset = drift, t
    return max_drift, max_asset

def calc_cash_allocation(returns_window, risk_free_rate, kelly_lookback,
                         kelly_base_cap, kelly_max_cap,
                         cash_min_volatility, cash_max_volatility):
    """Determine target cash allocation using Kelly (cap) and GARCH (scaling)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            models, _, _ = fit_garch_for_assets(returns_window)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns_window))
            sys.stdout = old_stdout
    except Exception:
        avg_vol = returns_window.std().mean() * np.sqrt(252)

    if len(returns_window) > kelly_lookback:
        kelly_returns = returns_window.iloc[-kelly_lookback:]
    else:
        kelly_returns = returns_window

    try:
        exp_ret, cov, _ = calculate_annualised_stats(kelly_returns)
        opt_results = optimise_portfolios(exp_ret, cov, risk_free_rate)
        weights = opt_results['msr_weights']
        mu = np.sum(exp_ret * weights)
        sigma = np.sqrt(weights.T @ cov @ weights)
        f_star = (mu - risk_free_rate) / (sigma ** 2) if sigma > 0 else 0.0
    except Exception:
        f_star = 0.0

    cash_cap = kelly_base_cap if f_star > 0 else kelly_max_cap

    if avg_vol <= cash_min_volatility:
        cash_pct = 0.0
    elif avg_vol >= cash_max_volatility:
        cash_pct = cash_cap
    else:
        fraction = (avg_vol - cash_min_volatility) / (cash_max_volatility - cash_min_volatility)
        cash_pct = fraction * cash_cap

    return cash_pct, f_star

def optimise_portfolio(returns_window, risk_free_rate, tickers):
    """Return MSR weights for the given tickers, or equal weights on failure."""
    try:
        exp_ret, cov, _ = calculate_annualised_stats(returns_window)
        opt = optimise_portfolios(exp_ret, cov, risk_free_rate)
        return {t: opt['msr_weights'][i] for i, t in enumerate(tickers)}
    except:
        return {t: 1.0/len(tickers) for t in tickers}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_daily_asset_values(name, date, asset_values, cash_pounds, total_value):
    """Log daily asset values for a universe (one row per date)."""
    log_file = os.path.join(FINAL_LOGS_DIR, f"{name.lower()}_daily.csv")
    entry = {'date': date.strftime("%Y-%m-%d"), 'total_value': total_value, 'cash': cash_pounds}
    for ticker, val in asset_values.items():
        entry[ticker] = val if val > 0 else 0.0

    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        if date.strftime("%Y-%m-%d") in existing['date'].values:
            existing.loc[existing['date'] == date.strftime("%Y-%m-%d")] = entry
            existing.to_csv(log_file, index=False)
            return
        df = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(log_file, index=False)
        return
    pd.DataFrame([entry]).to_csv(log_file, index=False)

# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def get_injection_date_from_log():
    """Return the injection date from the first row of portfolio_log.csv, or None."""
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            return pd.to_datetime(df.iloc[0]['date']).date()
    return None

def compute_target_state(tickers, params, returns_data):
    """Compute target weights and cash % from a returns window."""
    lookback_days = params['lookback_days']
    if len(returns_data) > lookback_days:
        returns_window = returns_data.iloc[-lookback_days:]
    else:
        returns_window = returns_data

    if len(returns_window) < lookback_days * 0.5:
        n = len(tickers)
        target_weights = {t: 1.0/n for t in tickers}
        target_cash_pct = 0.0
    else:
        target_weights = optimise_portfolio(returns_window, RISK_FREE_RATE, tickers)
        kelly_lookback = params.get('kelly_lookback', 165)
        target_cash_pct, _ = calc_cash_allocation(
            returns_window, RISK_FREE_RATE, kelly_lookback,
            KELLY_BASE_CAP, KELLY_MAX_CAP,
            CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY
        )
    return target_weights, target_cash_pct

def initialise_state_from_weights(tickers, target_weights, target_cash_pct, total_capital=INITIAL_CAPITAL):
    """Initialise asset values and cash from target weights."""
    asset_values = {t: total_capital * (1 - target_cash_pct) * target_weights.get(t, 0.0) for t in tickers}
    cash = total_capital * target_cash_pct
    total_asset = sum(asset_values.values())
    if abs(total_asset + cash - total_capital) > 1e-6:
        scale = (total_capital - cash) / total_asset if total_asset > 0 else 1.0
        asset_values = {t: v * scale for t, v in asset_values.items()}
    return asset_values, cash

def read_engine_state_all_assets():
    """Read the first row of portfolio_log.csv as the engine's initial state."""
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    if not os.path.exists(log_file):
        return None, None

    df = pd.read_csv(log_file)
    if len(df) == 0:
        return None, None

    first_row = df.iloc[0]
    asset_values = {}

    for col in first_row.index:
        if col.endswith('_value') and col not in ['portfolio_value', 'cash_pounds', 'realised_profit', 'total_wealth']:
            ticker = col.replace('_value', '')
            val = first_row[col]
            if pd.isna(val):
                asset_values[ticker] = 0.0
            else:
                asset_values[ticker] = float(val) if val > 0 else 0.0

    cash = float(first_row['cash_pounds']) if 'cash_pounds' in df.columns else 0.0
    return asset_values, cash

# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def run_universe_tracker(tickers, params, name, spy_shifted=None, ml_weights=None):
    """Run the day-by-day tracker for a single universe."""
    print(f"\n{name}...")
    print(f"   Kelly lookback: {params['kelly_lookback']} days")
    print(f"   Rebalance: {params['rebalance_min_days']}-{params['rebalance_max_days']} days")
    print(f"   Drift threshold: {params['drift_threshold']*100:.1f}%")

    injection_date = get_injection_date_from_log()
    if injection_date is None:
        injection_date = datetime.now().date()
        print(f"   Injection date set to: {injection_date} (no log found)")

    all_assets_from_log = []
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in first_row.index:
                if col.endswith('_value') and col not in ['portfolio_value', 'cash_pounds', 'realised_profit', 'total_wealth']:
                    ticker = col.replace('_value', '')
                    val = first_row[col]
                    if not pd.isna(val) and val > 0:
                        all_assets_from_log.append(ticker)

    all_tickers_to_fetch = sorted(set(tickers + all_assets_from_log))
    print(f"   Fetching prices for {len(all_tickers_to_fetch)} tickers ({len(tickers)} target + {len(all_tickers_to_fetch) - len(tickers)} from log)")

    raw_prices_df = fetch_price_data_for_universe(
        all_tickers_to_fetch,
        params.get('lookback_days', 405),
        injection_date
    )

    if raw_prices_df is None or len(raw_prices_df) == 0:
        print("   No price data. Skipping.")
        return None, None

    prefetch_fx_rates(all_tickers_to_fetch,
                      start_date=raw_prices_df.index.min(),
                      end_date=datetime.now().date())

    unshifted_df = raw_prices_df.dropna(how='all')
    business_days = pd.date_range(start=unshifted_df.index.min(), end=unshifted_df.index.max(), freq='B')
    unshifted_df = unshifted_df.reindex(business_days).ffill().bfill()

    if ml_weights is None and name == "ML_Strategy":
        ml_weights = load_ml_weights_for_date(injection_date)
        if ml_weights is None:
            print("   ML weights not available; skipping ML Strategy.")
            return None, None

    if ml_weights is not None:
        total_w = sum(ml_weights.values())
        if total_w > 0:
            norm_weights = {t: w / total_w for t, w in ml_weights.items()}
        else:
            norm_weights = {t: 0.0 for t in tickers}
        target_weights = norm_weights
        target_cash_pct = 0.0
        asset_values, cash = initialise_state_from_weights(tickers, target_weights, target_cash_pct)
        total = sum(asset_values.values()) + cash
        print(f"   ML state: Total = £{total:.2f}, Cash = £{cash:.2f}")
    else:
        if name == "Optimal_Ethical":
            print("   Optimal_Ethical: Using engine's logged state (in sync with live engine)")
            asset_values, cash = read_engine_state_all_assets()
            if asset_values is None:
                print("   No log found; falling back to computed target.")
                returns_data = unshifted_df.pct_change().dropna()
                returns_data = returns_data[returns_data.index <= pd.Timestamp(injection_date)]
                target_weights, target_cash_pct = compute_target_state(tickers, params, returns_data)
                asset_values, cash = initialise_state_from_weights(tickers, target_weights, target_cash_pct)
            else:
                total = sum(asset_values.values()) + cash
                if abs(total - 100.0) > 0.01:
                    asset_total = sum(asset_values.values())
                    if asset_total > 0:
                        scale = (100.0 - cash) / asset_total
                        for t in asset_values:
                            asset_values[t] *= scale
                for t in tickers:
                    if t not in asset_values:
                        asset_values[t] = 0.0
                print(f"   Engine state: {len(asset_values)} assets, Total = £{sum(asset_values.values()) + cash:.2f}, Cash = £{cash:.2f}")
        else:
            print(f"   {name}: Computing target weights from scratch on injection date")
            returns_data = unshifted_df.pct_change().dropna()
            returns_data = returns_data[returns_data.index <= pd.Timestamp(injection_date)]
            if len(returns_data) < 50:
                print(f"   Not enough returns data up to {injection_date}, using all available.")
            target_weights, target_cash_pct = compute_target_state(tickers, params, returns_data)
            print(f"   Target cash: {target_cash_pct*100:.1f}%")
            asset_values, cash = initialise_state_from_weights(tickers, target_weights, target_cash_pct)
            total = sum(asset_values.values()) + cash
            print(f"   Target state: Total = £{total:.2f}, Cash = £{cash:.2f}")

    total = sum(asset_values.values()) + cash
    if abs(total - 100.0) > 0.01:
        asset_total = sum(asset_values.values())
        if asset_total > 0:
            scale = (100.0 - cash) / asset_total
            for t in asset_values:
                asset_values[t] *= scale
        else:
            for t in tickers:
                asset_values[t] = (100.0 - cash) / len(tickers)
        total = sum(asset_values.values()) + cash
        print(f"   Normalised to £{total:.2f} (Cash: £{cash:.2f})")

    print(f"   Initial: £{total:.2f} (Cash: £{cash:.2f})")

    shifted_df = unshifted_df.copy()
    shifted_df.index = shifted_df.index + pd.Timedelta(days=1)

    calendar_index = pd.date_range(start=shifted_df.index.min(), end=shifted_df.index.max(), freq='D')
    shifted_calendar_df = shifted_df.reindex(calendar_index).ffill()

    today = datetime.now().date()
    shifted_calendar_df = shifted_calendar_df[shifted_calendar_df.index <= pd.Timestamp(today)]

    all_days = get_all_days_between(injection_date, today)

    dates, values = [], []
    current_values = asset_values.copy()
    current_cash = cash
    daily_rate = daily_cash_rate(CASH_INTEREST_RATE)

    prev_f_star = None
    last_rebalance = injection_date
    kelly_sign_changes = 0

    lookback_days = params.get('lookback_days', 405)
    rebalance_min = params.get('rebalance_min_days', 110)
    rebalance_max = params.get('rebalance_max_days', 130)
    drift_threshold = params.get('drift_threshold', 0.085)
    kelly_lookback = params.get('kelly_lookback', 165)
    take_profit_pct = params.get('take_profit_pct', 0.0)

    total = sum(current_values.values()) + current_cash
    dates.append(injection_date)
    values.append(total)
    log_daily_asset_values(name, injection_date, current_values, current_cash, total)

    print(f"   Tracking from {injection_date} to {today} ({len(all_days)} days)")

    spy_start_value = 100.0
    if spy_shifted is not None and len(spy_shifted) > 0:
        if injection_date in spy_shifted.index:
            spy_start_value = float(spy_shifted.loc[injection_date])
        else:
            spy_start_value = float(spy_shifted.iloc[0])

    for idx, day in enumerate(all_days):
        if idx == 0:
            continue

        current_cash *= (1 + daily_rate)

        today_prices = get_all_shifted_prices(day, shifted_calendar_df)
        prev_day = day - timedelta(days=1)
        yesterday_prices = get_all_shifted_prices(prev_day, shifted_calendar_df)

        for t in current_values:
            if current_values.get(t, 0) > 0:
                today_price = today_prices.get(t, 0.0)
                yesterday_price = yesterday_prices.get(t, 0.0)
                if yesterday_price > 0 and today_price > 0:
                    price_ratio = today_price / yesterday_price
                    ccy = get_ticker_currency(t)
                    if ccy != 'GBP':
                        price_ratio *= get_fx_ratio(ccy, day)
                    current_values[t] *= price_ratio

        total = sum(current_values.values()) + current_cash
        current_weights = {t: current_values[t] / total if total > 0 else 0.0 for t in current_values}

        if ml_weights is not None:
            days_since = (day - last_rebalance).days
            if days_since >= 120:
                asset_weight_sum = sum(target_weights.values())
                if asset_weight_sum >= 1:
                    target_cash_pct = 0.0
                    scaled_target = {t: w / asset_weight_sum for t, w in target_weights.items()}
                else:
                    target_cash_pct = 1.0 - asset_weight_sum
                    if asset_weight_sum > 0:
                        scaled_target = {t: w * (1 - target_cash_pct) / asset_weight_sum for t, w in target_weights.items()}
                    else:
                        scaled_target = {t: 0.0 for t in tickers}

                target_cash = total * target_cash_pct

                trades = {}
                total_trade_value = 0.0
                for t in tickers:
                    current_val = current_values.get(t, 0.0)
                    target_val = total * scaled_target.get(t, 0.0)
                    diff = target_val - current_val
                    if abs(diff) > 0.01:
                        trades[t] = {'action': 'BUY' if diff > 0 else 'SELL', 'amount': abs(diff)}
                        total_trade_value += abs(diff)
                cash_diff = target_cash - current_cash
                if abs(cash_diff) > 0.01:
                    trades['CASH'] = {'action': 'BUY' if cash_diff > 0 else 'SELL', 'amount': abs(cash_diff)}
                    total_trade_value += abs(cash_diff)

                cost = total_trade_value * ROUND_TRIP_COST_PCT
                current_cash -= cost
                total -= cost

                for t in current_values:
                    current_values[t] = 0.0
                for t in tickers:
                    current_values[t] = total * scaled_target.get(t, 0.0)
                current_cash = total * target_cash_pct
                total = sum(current_values.values()) + current_cash
                last_rebalance = day
                print(f"   ML Rebalance on {day}: cost £{cost:.4f}")

            dates.append(day)
            values.append(total)
            log_daily_asset_values(name, day, current_values, current_cash, total)
            continue

        shifted_business = shifted_calendar_df.resample('B').last().ffill()
        returns_all = shifted_business[tickers].pct_change().dropna()
        returns_all = returns_all[returns_all.index <= pd.Timestamp(day)]

        if len(returns_all) > lookback_days:
            returns_window = returns_all.iloc[-lookback_days:]
        else:
            returns_window = returns_all

        if len(returns_window) > lookback_days * 0.5:
            target_weights = optimise_portfolio(returns_window, RISK_FREE_RATE, tickers)
            target_cash_pct, f_star = calc_cash_allocation(
                returns_window, RISK_FREE_RATE, kelly_lookback,
                KELLY_BASE_CAP, KELLY_MAX_CAP,
                CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY
            )

            force_rebalance = False
            if prev_f_star is not None and (prev_f_star > 0) != (f_star > 0):
                kelly_sign_changes += 1
                force_rebalance = True
            prev_f_star = f_star

            adjusted_target = {t: w * (1 - target_cash_pct) for t, w in target_weights.items()}
            days_since = (day - last_rebalance).days
            drift, asset = calculate_drift(current_weights, adjusted_target)

            rebalance_needed = False
            reason = ""

            if take_profit_pct > 0 and spy_shifted is not None:
                if day in spy_shifted.index:
                    spy_val = float(spy_shifted.loc[day])
                    spy_return = spy_val / spy_start_value if spy_start_value > 0 else 1.0
                    port_return = total / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 1.0
                    if port_return > spy_return * (1 + take_profit_pct):
                        new_total = spy_return * INITIAL_CAPITAL
                        profit = total - new_total
                        if profit > 0:
                            print(f"   TAKE-PROFIT on {day}: £{profit:.2f} "
                                  f"(portfolio {port_return*100:.1f}% vs SPY {spy_return*100:.1f}%)")
                            scale = new_total / total if total > 0 else 1.0
                            for t in current_values:
                                current_values[t] *= scale
                            current_cash *= scale
                            total = new_total
                            rebalance_needed = True
                            reason = "Take-profit"

            if sum(current_weights.values()) < 0.01 and not rebalance_needed:
                rebalance_needed = True
                reason = "First run"
            elif days_since >= rebalance_max and not rebalance_needed:
                rebalance_needed = True
                reason = f"Time-based: {days_since} days"
            elif days_since >= rebalance_min and drift > drift_threshold and not rebalance_needed:
                rebalance_needed = True
                reason = f"Drift: {asset} {drift*100:.2f}%"
            elif force_rebalance and not rebalance_needed:
                rebalance_needed = True
                reason = "Kelly sign change"

            if rebalance_needed:
                target_cash = total * target_cash_pct
                new_values = {}
                for t in current_values:
                    new_values[t] = 0.0
                for t in tickers:
                    new_values[t] = (total - target_cash) * adjusted_target.get(t, 0.0)

                total_trade_value = 0.0
                for t in current_values:
                    diff = abs(current_values.get(t, 0.0) - new_values.get(t, 0.0))
                    if diff > 0.01:
                        total_trade_value += diff
                cash_diff = abs(current_cash - target_cash)
                if cash_diff > 0.01:
                    total_trade_value += cash_diff

                cost = total_trade_value * ROUND_TRIP_COST_PCT
                if cost > 0:
                    current_cash -= cost
                    total -= cost
                    target_cash = total * target_cash_pct
                    for t in current_values:
                        new_values[t] = 0.0
                    for t in tickers:
                        new_values[t] = (total - target_cash) * adjusted_target.get(t, 0.0)
                    print(f"   Rebalance on {day}: cost £{cost:.4f}")

                if sum(new_values.values()) > 0:
                    current_values = new_values
                    current_cash = target_cash
                    total = sum(current_values.values()) + current_cash
                    last_rebalance = day

        dates.append(day)
        values.append(total)
        log_daily_asset_values(name, day, current_values, current_cash, total)

    print(f"   {name} complete: {len(values)} days, {kelly_sign_changes} Kelly sign changes")
    return pd.Series(values, index=pd.DatetimeIndex(dates)), injection_date

# ---------------------------------------------------------------------------
# Normalisation and alignment
# ---------------------------------------------------------------------------

def normalise(series, injection_date):
    """Normalise a series so it starts at INITIAL_CAPITAL on the injection date."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    crop = series[series.index >= pd.Timestamp(injection_date)]
    if len(crop) == 0:
        return pd.Series(dtype=float)
    first = float(crop.iloc[0])
    return (crop / first) * INITIAL_CAPITAL if first > 0 else pd.Series(dtype=float)

def align_series(data_dict):
    """Align all series to a common daily date range."""
    all_dates = pd.DatetimeIndex([])
    for s in data_dict.values():
        if s is not None and len(s) > 0:
            all_dates = all_dates.union(s.index)
    if len(all_dates) == 0:
        return data_dict
    full_range = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='D')
    aligned = {}
    for name, s in data_dict.items():
        if s is not None and len(s) > 0:
            aligned[name] = s.reindex(full_range).ffill().bfill()
        else:
            aligned[name] = pd.Series(dtype=float)
    return aligned

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_performance(data, title, filename, log_scale=False):
    """Plot performance curves for all universes."""
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='series')

    fig, ax = plt.subplots(figsize=(14, 8))

    for name, series in data.items():
        if series is not None and len(series) > 0:
            color = SPY_COLOR if name == "SPY" else UNIVERSE_COLORS.get(name, '#888')
            label = 'SPY (Benchmark)' if name == "SPY" else f'{name} - Active Strategy'
            if log_scale:
                ax.semilogy(series.index, series, color=color, linewidth=3 if name == "SPY" else 2,
                           label=label, marker='o', markersize=6)
            else:
                ax.plot(series.index, series, color=color, linewidth=3 if name == "SPY" else 2,
                       label=label, marker='o', markersize=6)

    all_data = [v for s in data.values() if s is not None for v in s.values]
    if all_data:
        min_d = min([s.index.min() for s in data.values() if s is not None and len(s) > 0])
        max_d = max([s.index.max() for s in data.values() if s is not None and len(s) > 0])
        pad = max((max_d - min_d).days * 0.05, 2)
        ax.set_xlim(min_d - pd.Timedelta(days=pad), max_d + pd.Timedelta(days=pad))
        min_v, max_v = min(all_data), max(all_data)
        if not log_scale:
            pad_y = max((max_v - min_v) * 0.1, 5)
            ax.set_ylim(min_v - pad_y, max_v + pad_y)
        else:
            positive = [v for v in all_data if v > 0]
            if positive:
                ax.set_ylim(min(positive) * 0.9, max(all_data) * 1.1)

    ax.axhline(INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (£)' + (' - Log Scale' if log_scale else ''))
    ax.set_title(title + (' (Log Scale)' if log_scale else ''))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    y_off = 0.02
    for name, series in data.items():
        if series is not None and len(series) > 0:
            color = SPY_COLOR if name == "SPY" else UNIVERSE_COLORS.get(name, '#888')
            val = float(series.iloc[-1])
            date_str = series.index[-1].strftime('%d/%m/%Y')
            ax.text(0.02, y_off, f"{name}: £{val:,.2f} ({date_str})",
                   transform=ax.transAxes, fontsize=9, color=color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.85))
            y_off += 0.045

    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {filename}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FINAL PERFORMANCE TRACKER (hybrid initialisation + ML date-aware + universe-agnostic)")
    print("=" * 70)

    for name in ["original_18", "optimal_standard", "optimal_ethical", "ml_strategy"]:
        log_file = os.path.join(FINAL_LOGS_DIR, f"{name.lower()}_daily.csv")
        if os.path.exists(log_file):
            os.remove(log_file)

    injection_date = get_injection_date_from_log()
    if injection_date is None:
        injection_date = datetime.now().date()
        print(f"Injection date set to: {injection_date} (no log found)")

    print("\nDownloading SPY...")
    backtest_start = injection_date - timedelta(days=30)
    spy_shifted = download_spy(backtest_start.strftime("%Y-%m-%d"),
                               datetime.now().strftime("%Y-%m-%d"),
                               injection_date)

    universes = {
        "Original_18": {
            'tickers': ORIGINAL_UNIVERSE,
            'params': ORIGINAL_PARAMS,
        },
        "Optimal_Standard": {
            'tickers': OPTIMAL_STANDARD_PORTFOLIO,
            'params': OPTIMAL_STANDARD_PARAMS,
        },
        "Optimal_Ethical": {
            'tickers': TICKERS,
            'params': {
                'lookback_days': LOOKBACK_DAYS,
                'rebalance_min_days': REBALANCE_MIN_DAYS,
                'rebalance_max_days': REBALANCE_MAX_DAYS,
                'drift_threshold': DRIFT_THRESHOLD,
                'take_profit_pct': RELATIVE_TAKE_PROFIT_PCT,
                'kelly_lookback': KELLY_LOOKBACK,
            }
        },
    }

    if ML_AVAILABLE:
        from final_universe import FINAL_POOL
        ml_params = {
            'lookback_days': 405,
            'rebalance_min_days': 120,
            'rebalance_max_days': 120,
            'drift_threshold': 1.0,
            'take_profit_pct': 0.0,
            'kelly_lookback': 165,
        }
        universes["ML_Strategy"] = {
            'tickers': FINAL_POOL,
            'params': ml_params,
            'ml_weights': None,
        }
        print(f"\nML Strategy will be generated at injection date.")
    else:
        print("\nML Strategy skipped (final_ml_research.py not found).")

    all_data = {}

    for name, config in universes.items():
        print(f"\n{'='*60}\nRUNNING: {name}\n{'='*60}")
        ml_weights_for_universe = config.get('ml_weights', None)
        tickers = config['tickers']
        params = config['params']

        series, inj_date = run_universe_tracker(
            tickers,
            params,
            name,
            spy_shifted,
            ml_weights=ml_weights_for_universe
        )

        if series is not None:
            all_data[name] = normalise(series, inj_date)

    if not all_data:
        print("No data generated. Exiting.")
        return

    all_data["SPY"] = normalise(spy_shifted, injection_date)

    print("\nAligning series...")
    all_data = align_series(all_data)

    today = datetime.now().date()
    combined_df = pd.DataFrame()
    for name, series in all_data.items():
        if series is not None and len(series) > 0:
            combined_df[name] = series[series.index <= pd.Timestamp(today)]

    csv_path = os.path.join(FINAL_LOGS_DIR, "daily_portfolio_values.csv")
    combined_df.to_csv(csv_path, float_format='%.2f')
    print(f"\nDaily values saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("FINAL VALUES")
    print("=" * 70)
    for name, series in combined_df.items():
        if series is not None and len(series) > 0:
            val = float(series.iloc[-1])
            ret = (val / INITIAL_CAPITAL - 1) * 100
            print(f"   {name}: £{val:.2f} ({ret:.1f}%)")

    print("\nGenerating plots...")
    title = f'Performance Since {injection_date.strftime("%d/%m/%Y")}\n(previous-day close)'
    plot_dict = combined_df.to_dict(orient='series')
    plot_performance(plot_dict, title, "final_performance_tracker.png", log_scale=False)
    plot_performance(plot_dict, title, "final_performance_tracker_log.png", log_scale=True)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()