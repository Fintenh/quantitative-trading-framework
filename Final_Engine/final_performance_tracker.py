"""
final_performance_tracker.py
----------------------------
Tracks portfolio performance from the injection date onwards.
Uses previous-day closing prices for all asset updates.

Implements:
- Rolling optimisation lookback (configurable)
- Kelly cash allocation with sign-change detection
- Take-profit relative to SPY
- Time and drift based rebalancing
- Cash interest
- No look-ahead bias (manual date shift)
- Crops output to today's date
- Reads the initial state directly from portfolio_log.csv rather than
  recomputing it, so the tracker always starts from the same state as
  the trading engine
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for phase in ["Phase_2", "Phase_3", "Phase_5"]:
    path = os.path.join(parent_dir, phase)
    if os.path.exists(path):
        sys.path.insert(0, path)

# --- Imports ---
from ethical_universe import ORIGINAL_UNIVERSE
from final_config import (
    INITIAL_CAPITAL, RISK_FREE_RATE,
    CASH_INTEREST_RATE, CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION, KELLY_BASE_CAP, KELLY_MAX_CAP,
)
from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO, OPTIMAL_ETHICAL_PORTFOLIO,
    ORIGINAL_PARAMS, OPTIMAL_STANDARD_PARAMS, OPTIMAL_ETHICAL_PARAMS,
)

from data_fetcher import calculate_returns, calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

# --- Constants ---
UNIVERSE_COLORS = {"Original_18": '#1f77b4', "Optimal_Standard": '#ff7f0e', "Optimal_Ethical": '#2ca02c'}
SPY_COLOR = '#d62728'

FINAL_ENGINE_DIR = os.path.join(parent_dir, "Final_Engine")
FINAL_LOGS_DIR = os.path.join(FINAL_ENGINE_DIR, "logs")
FINAL_FIGURES_DIR = os.path.join(FINAL_ENGINE_DIR, "figures")
os.makedirs(FINAL_LOGS_DIR, exist_ok=True)
os.makedirs(FINAL_FIGURES_DIR, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Logs directory: {FINAL_LOGS_DIR}")
print(f"Figures directory: {FINAL_FIGURES_DIR}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_all_days_between(start_date, end_date):
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def daily_cash_rate(annual):
    return (1 + annual) ** (1 / 252) - 1 if annual > 0 else 0.0


def get_all_shifted_prices(date, shifted_df):
    """Get all prices from the shifted calendar DataFrame for a given date."""
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


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_price_data_for_universe(tickers, lookback_days, injection_date):
    """Fetch closing prices for a specific universe."""
    end_date = datetime.now()
    end_date_for_yf = end_date + timedelta(days=1)
    start_date = min(end_date - timedelta(days=lookback_days * 2),
                     datetime.combine(injection_date, datetime.min.time()) - timedelta(days=lookback_days * 2))

    print(f"\nDownloading data for {len(tickers)} tickers...")
    print(f"   Period: {start_date.date()} to {end_date.date()}")

    all_data = {}
    for t in tickers:
        try:
            data = yf.download(t, start=start_date, end=end_date_for_yf, progress=False)
            if len(data) == 0:
                all_data[t] = pd.Series(dtype=float)
                continue
            series = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
            if len(series) > 0:
                all_data[t] = series
            else:
                all_data[t] = pd.Series(dtype=float)
        except:
            all_data[t] = pd.Series(dtype=float)

    valid_series = {k: v for k, v in all_data.items() if len(v) > 0}
    if not valid_series:
        return pd.DataFrame()

    # Create aligned DataFrame
    all_dates = pd.DatetimeIndex([])
    for series in valid_series.values():
        all_dates = all_dates.union(series.index)

    business_days = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='B')
    df = pd.DataFrame(index=business_days)

    for ticker, series in valid_series.items():
        df[ticker] = series.reindex(business_days).ffill().bfill().fillna(0)

    for t in tickers:
        if t not in df.columns:
            df[t] = 0.0

    print(f"   Aligned to {len(df)} business days")
    return df


def download_spy(start_date, end_date, injection_date):
    """Download SPY and shift to previous-day close."""
    print("   Downloading SPY...")
    try:
        end_dt = datetime.now() + timedelta(days=5)
        spy = yf.download("SPY", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)["Close"]
        if len(spy) == 0:
            spy = yf.download("^GSPC", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)["Close"]

        if len(spy) > 0:
            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]

            business_days = pd.date_range(start=spy.index.min(), end=spy.index.max(), freq='B')
            spy = spy.reindex(business_days).ffill().bfill()

            # Shift index forward by 1 day (previous-day close)
            spy_shifted = spy.copy()
            spy_shifted.index = spy.index + pd.Timedelta(days=1)

            # Crop to today and injection date
            today = datetime.now().date()
            spy_shifted = spy_shifted[spy_shifted.index <= pd.Timestamp(today)]
            spy_shifted = spy_shifted[spy_shifted.index >= pd.Timestamp(injection_date)]

            print(f"   SPY: {len(spy_shifted)} days (previous-day close)")
            return spy_shifted

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)
    except Exception as e:
        print(f"   Error: {e}")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)


# ---------------------------------------------------------------------------
# Portfolio calculations
# ---------------------------------------------------------------------------

def calculate_drift(current, target):
    max_drift, max_asset = 0.0, None
    for t in current:
        drift = abs(current.get(t, 0.0) - target.get(t, 0.0))
        if drift > max_drift:
            max_drift, max_asset = drift, t
    return max_drift, max_asset


def calc_cash_allocation(returns_window, risk_free_rate, kelly_lookback,
                         kelly_base_cap, kelly_max_cap,
                         cash_min_volatility, cash_max_volatility, cash_max_allocation):
    """
    Calculate cash allocation using GARCH (on returns_window) and Kelly (on the last kelly_lookback days).

    Args:
        returns_window: DataFrame of returns (already limited to the optimisation lookback)
        kelly_lookback: number of days to use for the Kelly f* calculation
    """
    # --- GARCH volatility (using the full window) ---
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            models, _, _ = fit_garch_for_assets(returns_window)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns_window))
            sys.stdout = old_stdout
    except Exception:
        avg_vol = returns_window.std().mean() * np.sqrt(252)

    # --- Kelly f* (using the last kelly_lookback days) ---
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

    # Kelly cash cap
    cash_cap = kelly_base_cap if f_star > 0 else kelly_max_cap

    # Scale cash by GARCH volatility
    if avg_vol <= cash_min_volatility:
        cash_pct = 0.0
    elif avg_vol >= cash_max_volatility:
        cash_pct = cash_cap
    else:
        fraction = (avg_vol - cash_min_volatility) / (cash_max_volatility - cash_min_volatility)
        cash_pct = fraction * cash_cap

    return cash_pct, f_star


def optimise_portfolio(returns_window, risk_free_rate, tickers):
    """Optimise portfolio using the given returns window."""
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
# Injection state (read directly from the log file)
# ---------------------------------------------------------------------------

def get_injection_state_for_universe(tickers, params, unshifted_df):
    """
    Read the first entry from portfolio_log.csv and use it as the starting
    state. This ensures the performance tracker starts from the exact same
    state as the trading engine, rather than recomputing an optimisation
    that might drift from what the engine actually logged.
    """
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    if not os.path.exists(log_file):
        print("   No log file found. Cannot determine injection date.")
        return None, None, None

    df = pd.read_csv(log_file)
    if len(df) == 0:
        print("   Log file is empty.")
        return None, None, None

    # Take the first row (injection/rebalanced state)
    first_row = df.iloc[0]
    injection_date = pd.to_datetime(first_row['date']).date()
    print(f"   Injection date from log: {injection_date}")

    # Extract asset values from the first row
    asset_values = {}
    for t in tickers:
        col = f'{t}_value'
        if col in df.columns:
            asset_values[t] = float(first_row[col]) if first_row[col] > 0 else 0.0
        else:
            asset_values[t] = 0.0

    # Extract cash
    if 'cash_pounds' in df.columns:
        cash = float(first_row['cash_pounds'])
    else:
        cash = 0.0

    # Ensure all tickers are present
    for t in tickers:
        if t not in asset_values:
            asset_values[t] = 0.0

    print(f"   Starting state from log: Total = £{sum(asset_values.values()) + cash:.2f}, Cash = £{cash:.2f}")
    return injection_date, asset_values, cash


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

def run_universe_tracker(tickers, params, name, raw_prices_df, spy_shifted=None):
    print(f"\n{name}...")
    print(f"   Kelly lookback: {params['kelly_lookback']} days")
    print(f"   Rebalance: {params['rebalance_min_days']}-{params['rebalance_max_days']} days")
    print(f"   Drift threshold: {params['drift_threshold']*100:.1f}%")

    unshifted_df = raw_prices_df.dropna(how='all')
    business_days = pd.date_range(start=unshifted_df.index.min(), end=unshifted_df.index.max(), freq='B')
    unshifted_df = unshifted_df.reindex(business_days).ffill().bfill()

    injection_date, asset_values, cash = get_injection_state_for_universe(tickers, params, unshifted_df)
    if injection_date is None:
        return None, None

    total = sum(asset_values.values()) + cash
    print(f"   Initial: £{total:.2f} (Cash: £{cash:.2f})")

    # Create shifted calendar version
    prices_aligned = raw_prices_df.reindex(business_days).ffill().bfill()

    # Manual date shift: shift index forward by 1 day
    shifted_df = prices_aligned.copy()
    shifted_df.index = shifted_df.index + pd.Timedelta(days=1)

    # Reindex to full calendar and crop to today
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

    lookback_days = params['lookback_days']
    rebalance_min = params['rebalance_min_days']
    rebalance_max = params['rebalance_max_days']
    drift_threshold = params['drift_threshold']
    kelly_lookback = params['kelly_lookback']
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

        # Apply price changes
        for t in tickers:
            if current_values.get(t, 0) > 0:
                today_price = today_prices.get(t, 0.0)
                yesterday_price = yesterday_prices.get(t, 0.0)
                if yesterday_price > 0 and today_price > 0:
                    current_values[t] *= (today_price / yesterday_price)

        total = sum(current_values.values()) + current_cash
        current_weights = {t: current_values[t] / total if total > 0 else 0.0 for t in tickers}

        # Get returns up to this day (shifted)
        shifted_business = shifted_calendar_df.resample('B').last().ffill()
        returns_all = shifted_business[tickers].pct_change().dropna()
        returns_all = returns_all[returns_all.index <= pd.Timestamp(day)]

        # --- Use rolling lookback windows ---
        if len(returns_all) > lookback_days:
            returns_window = returns_all.iloc[-lookback_days:]
        else:
            returns_window = returns_all

        if len(returns_window) > lookback_days * 0.5:
            # Optimise with rolling window
            target_weights = optimise_portfolio(returns_window, RISK_FREE_RATE, tickers)
            target_cash_pct, f_star = calc_cash_allocation(
                returns_window, RISK_FREE_RATE, kelly_lookback,
                KELLY_BASE_CAP, KELLY_MAX_CAP,
                CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION
            )

            # Kelly sign change detection
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

            # Take-profit check
            if take_profit_pct > 0 and spy_shifted is not None:
                if day in spy_shifted.index:
                    spy_val = float(spy_shifted.loc[day])
                    spy_normalized = (spy_val / spy_start_value) * 100.0 if spy_start_value > 0 else 100.0
                    target_spy = 100.0 * (1 + take_profit_pct)
                    port_normalized = (total / INITIAL_CAPITAL) * 100.0
                    if port_normalized > target_spy:
                        profit = total - (spy_normalized / 100.0 * INITIAL_CAPITAL)
                        if profit > 0:
                            print(f"   TAKE-PROFIT on {day}: £{profit:.2f}")
                            total = (spy_normalized / 100.0) * INITIAL_CAPITAL
                            scale = total / sum(current_values.values()) if sum(current_values.values()) > 0 else 1.0
                            for t in current_values:
                                current_values[t] *= scale
                            current_cash *= scale
                            rebalance_needed = True
                            reason = "Take-profit"

            # Rebalance conditions
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
                for t in tickers:
                    new_values[t] = (total - target_cash) * adjusted_target.get(t, 0.0)

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

def plot_performance(data, injection_date, title, filename, log_scale=False):
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

    # Add final value annotations
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
    print("FINAL PERFORMANCE TRACKER (with rolling lookbacks)")
    print("=" * 70)

    # Clear old daily logs
    for name in ["original_18", "optimal_standard", "optimal_ethical"]:
        log_file = os.path.join(FINAL_LOGS_DIR, f"{name.lower()}_daily.csv")
        if os.path.exists(log_file):
            os.remove(log_file)

    # Get injection date
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    first_injection_date = None
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            first_injection_date = pd.to_datetime(df.iloc[0]['date']).date()
    if first_injection_date is None:
        first_injection_date = datetime.now().date()

    # Download SPY
    print("\nDownloading SPY...")
    backtest_start = first_injection_date - timedelta(days=30)
    spy_shifted = download_spy(backtest_start.strftime("%Y-%m-%d"),
                               datetime.now().strftime("%Y-%m-%d"),
                               first_injection_date)

    # Universes configuration
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
            'tickers': OPTIMAL_ETHICAL_PORTFOLIO,
            'params': OPTIMAL_ETHICAL_PARAMS,
        },
    }

    all_data = {}
    injection_dates = {}

    for name, config in universes.items():
        print(f"\n{'='*60}\nRUNNING: {name}\n{'='*60}")

        raw_prices = fetch_price_data_for_universe(
            config['tickers'],
            config['params']['lookback_days'],
            first_injection_date
        )

        if raw_prices is None or len(raw_prices) == 0:
            print(f"   No price data for {name}. Skipping.")
            continue

        series, inj_date = run_universe_tracker(
            config['tickers'],
            config['params'],
            name,
            raw_prices,
            spy_shifted
        )

        if series is not None:
            all_data[name] = normalise(series, inj_date)
            injection_dates[name] = inj_date

    if not all_data:
        print("No data generated. Exiting.")
        return

    all_data["SPY"] = normalise(spy_shifted, first_injection_date)

    # Align and crop to today
    print("\nAligning series...")
    all_data = align_series(all_data)

    today = datetime.now().date()
    combined_df = pd.DataFrame()
    for name, series in all_data.items():
        if series is not None and len(series) > 0:
            combined_df[name] = series[series.index <= pd.Timestamp(today)]

    # Save CSV
    csv_path = os.path.join(FINAL_LOGS_DIR, "daily_portfolio_values.csv")
    combined_df.to_csv(csv_path, float_format='%.2f')
    print(f"\nDaily values saved to: {csv_path}")

    # Print final values
    print("\n" + "=" * 70)
    print("FINAL VALUES")
    print("=" * 70)
    for name, series in combined_df.items():
        if series is not None and len(series) > 0:
            val = float(series.iloc[-1])
            ret = (val / INITIAL_CAPITAL - 1) * 100
            print(f"   {name}: £{val:.2f} ({ret:.1f}%)")

    # Generate plots
    print("\nGenerating plots...")
    title = f'Performance Since {first_injection_date.strftime("%d/%m/%Y")}\n(previous-day close)'

    plot_dict = combined_df.to_dict(orient='series')
    plot_performance(plot_dict, first_injection_date, title, "final_performance_tracker.png", log_scale=False)
    plot_performance(plot_dict, first_injection_date, title, "final_performance_tracker_log.png", log_scale=True)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()