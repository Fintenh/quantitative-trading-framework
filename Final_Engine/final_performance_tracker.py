"""
final_performance_tracker.py
----------------------------
Tracks portfolio performance from the injection date onwards.
Uses previous-day closing prices for all asset updates.
Initial weights are computed from the injection day's close (unshifted).
SPY is also shifted to previous-day close.
The same manual date shift is applied to all portfolio assets.
Returns are applied every day (including weekends) because the
shifted price on Saturday is Friday's close, so the Saturday update
reflects the Friday move.
Includes weekends in plots via forward-filling.
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
    TICKERS, INITIAL_CAPITAL, RISK_FREE_RATE, TRANSACTION_COST_PCT,
    CASH_INTEREST_RATE, CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION, KELLY_BASE_CAP, KELLY_MAX_CAP, LOG_DIR,
    LOOKBACK_DAYS,
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
# Price functions
# ---------------------------------------------------------------------------

def fetch_price_data_for_universe(tickers, lookback_days):
    """
    Fetch closing prices for a specific universe only.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)
    all_data = {}

    print(f"\nDownloading data for {len(tickers)} tickers...")

    for t in tickers:
        try:
            data = yf.download(t, start=start_date, end=end_date, progress=False)
            if len(data) == 0:
                print(f"   {t}: No data returned")
                all_data[t] = pd.Series(dtype=float)
                continue
            if 'Close' in data.columns:
                series = data['Close']
                if len(series) > 0:
                    all_data[t] = series
                    print(f"   {t}: {len(series)} days of data")
                    continue
            if len(data.columns) > 0:
                series = data.iloc[:, 0]
                if len(series) > 0:
                    all_data[t] = series
                    print(f"   {t}: {len(series)} days of data")
                    continue
            print(f"   {t}: No usable data")
            all_data[t] = pd.Series(dtype=float)
        except Exception as e:
            print(f"   {t}: Error - {e}")
            all_data[t] = pd.Series(dtype=float)

    valid_series = {k: v for k, v in all_data.items() if len(v) > 0}
    if len(valid_series) == 0:
        print("No valid data found for any ticker")
        return pd.DataFrame()

    common_index = None
    for series in valid_series.values():
        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)
    if common_index is None or len(common_index) == 0:
        common_index = next(iter(valid_series.values())).index

    df = pd.DataFrame()
    for ticker, series in valid_series.items():
        aligned = series.reindex(common_index).ffill().bfill().fillna(0)
        df[ticker] = aligned

    df = df.ffill().bfill()

    if len(df) > 0:
        business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(business_days).ffill().bfill().fillna(0)

    for t in tickers:
        if t not in df.columns:
            df[t] = 0.0

    print(f"\nData summary for universe:")
    print(f"   Aligned to {len(df)} business days")
    print(f"   Columns: {list(df.columns)}")

    return df


def get_all_shifted_prices(date, shifted_df):
    """Get all prices from shifted DataFrame for a given date."""
    prices = {}
    date_ts = pd.Timestamp(date)
    if date_ts in shifted_df.index:
        row = shifted_df.loc[date_ts]
        for t in shifted_df.columns:
            val = row[t]
            prices[t] = float(val) if val > 0 else 0.0
    else:
        idx = shifted_df.index[shifted_df.index <= date_ts]
        if len(idx) > 0:
            latest = idx[-1]
            row = shifted_df.loc[latest]
            for t in shifted_df.columns:
                val = row[t]
                prices[t] = float(val) if val > 0 else 0.0
        else:
            for t in shifted_df.columns:
                prices[t] = 0.0
    return prices


def get_all_days_between(start_date, end_date):
    """Get all calendar days between two dates (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def daily_cash_rate(annual):
    return (1 + annual) ** (1 / 252) - 1 if annual > 0 else 0.0


# ---------------------------------------------------------------------------
# Rebalance functions
# ---------------------------------------------------------------------------

def calculate_drift(current, target):
    max_drift, max_asset = 0.0, None
    for t in current:
        drift = abs(current.get(t, 0.0) - target.get(t, 0.0))
        if drift > max_drift:
            max_drift, max_asset = drift, t
    return max_drift, max_asset


def calc_cash_allocation(returns, risk_free_rate, kelly_lookback, kelly_base_cap, kelly_max_cap,
                         cash_min_volatility, cash_max_volatility, cash_max_allocation):
    try:
        exp_ret, cov, _ = calculate_annualised_stats(returns)
        opt_results = optimise_portfolios(exp_ret, cov, risk_free_rate)
        weights = opt_results['msr_weights']
        mu = np.sum(exp_ret * weights)
        sigma = np.sqrt(weights.T @ cov @ weights)
        f_star = (mu - risk_free_rate) / (sigma ** 2) if sigma > 0 else 0.0

        cash_cap = kelly_base_cap if f_star > 0 else kelly_max_cap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            models, _, _ = fit_garch_for_assets(returns)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns))
            sys.stdout = old_stdout

    except Exception:
        avg_vol = returns.std().mean() * np.sqrt(252)
        cash_cap = cash_max_allocation
        f_star = 0.0

    if avg_vol <= cash_min_volatility:
        return 0.0, f_star
    elif avg_vol >= cash_max_volatility:
        return cash_cap, f_star
    else:
        fraction = (avg_vol - cash_min_volatility) / (cash_max_volatility - cash_min_volatility)
        return fraction * cash_cap, f_star


def optimise_portfolio(returns, risk_free_rate, tickers):
    try:
        exp_ret, cov, _ = calculate_annualised_stats(returns)
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
        else:
            df = pd.concat([existing, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(log_file, index=False)
            return
    pd.DataFrame([entry]).to_csv(log_file, index=False)


# ---------------------------------------------------------------------------
# Injection state (unshifted)
# ---------------------------------------------------------------------------

def get_injection_state_for_universe(tickers, params, unshifted_df):
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    if not os.path.exists(log_file):
        print("   No log file found. Cannot determine injection date.")
        return None, None, None

    df = pd.read_csv(log_file)
    if len(df) == 0:
        print("   Log file is empty.")
        return None, None, None

    injection_date = pd.to_datetime(df.iloc[0]['date']).date()
    print(f"   Injection date: {injection_date}")

    returns = unshifted_df[tickers].pct_change().dropna()
    returns = returns[returns.index <= pd.Timestamp(injection_date)]

    if len(returns) < params['lookback_days'] * 0.5:
        print(f"   Insufficient data ({len(returns)} days). Using equal weights.")
        n = len(tickers)
        weights = {t: 1.0/n for t in tickers}
    else:
        exp_ret, cov, _ = calculate_annualised_stats(returns)
        opt = optimise_portfolios(exp_ret, cov, RISK_FREE_RATE)
        weights = {t: opt['msr_weights'][i] for i, t in enumerate(tickers)}

    target_cash_pct, _ = calc_cash_allocation(
        returns, RISK_FREE_RATE, params['kelly_lookback'],
        KELLY_BASE_CAP, KELLY_MAX_CAP,
        CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION
    )

    cash = INITIAL_CAPITAL * target_cash_pct
    asset_values = {t: (INITIAL_CAPITAL - cash) * weights.get(t, 0.0) for t in tickers}

    return injection_date, asset_values, cash


# ---------------------------------------------------------------------------
# Tracker with manual date shift (same as SPY)
# ---------------------------------------------------------------------------

def run_universe_tracker(tickers, params, name, raw_prices_df, spy_shifted=None):
    """
    Track portfolio with full rebalancing logic.
    Uses the same manual date shift as SPY:
      - raw price on date D is shifted to date D+1
      - reindexed to calendar (including weekends) with forward fill
      - returns are computed daily using shifted prices
    """
    print(f"\n{name}...")
    print(f"   Kelly lookback: {params['kelly_lookback']} days")
    print(f"   Rebalance: {params['rebalance_min_days']}-{params['rebalance_max_days']} days")
    print(f"   Drift threshold: {params['drift_threshold']*100:.1f}%")
    take_profit_pct = params.get('take_profit_pct', 0.0)
    if take_profit_pct > 0:
        print(f"   Take-Profit: {take_profit_pct*100:.1f}%")

    # 1. Compute initial state using unshifted data.
    # We need the raw (unshifted) prices for the initial optimisation.
    unshifted_df = raw_prices_df.dropna(how='all')
    business_days = pd.date_range(start=unshifted_df.index.min(), end=unshifted_df.index.max(), freq='B')
    unshifted_df = unshifted_df.reindex(business_days).ffill().bfill()

    injection_date, asset_values, cash = get_injection_state_for_universe(tickers, params, unshifted_df)
    if injection_date is None:
        print(f"   Failed to get injection state for {name}")
        return None, None

    total = sum(asset_values.values()) + cash
    print(f"\n   Injection weights for {name} (Total: £{total:.2f}, Cash: £{cash:.2f}):")
    sorted_weights = sorted([(t, v/total*100) for t, v in asset_values.items() if v > 0],
                           key=lambda x: x[1], reverse=True)
    for t, w in sorted_weights[:8]:
        print(f"      {t}: {w:.2f}%")

    # 2. Create shifted calendar version (same as SPY).
    # Align raw prices to business days
    business_days = pd.date_range(start=raw_prices_df.index.min(), end=raw_prices_df.index.max(), freq='B')
    prices_aligned = raw_prices_df.reindex(business_days).ffill().bfill()

    # Manual date shift: shift index forward by 1 day
    shifted_df = prices_aligned.copy()
    new_index = prices_aligned.index + pd.Timedelta(days=1)
    shifted_df.index = new_index

    # Reindex to full calendar (including weekends) with forward fill
    calendar_index = pd.date_range(start=shifted_df.index.min(), end=shifted_df.index.max(), freq='D')
    shifted_calendar_df = shifted_df.reindex(calendar_index).ffill()
    print(f"\n   Shifted calendar data: {len(shifted_calendar_df)} days")

    # 3. Iterate over all days (including weekends).
    today = datetime.now().date()
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

    total = sum(current_values.values()) + current_cash
    dates.append(injection_date)
    values.append(total)
    log_daily_asset_values(name, injection_date, current_values, current_cash, total)

    print(f"\n   Tracking from {injection_date} to {today} ({len(all_days)} days)")
    print(f"   Using previous-day closing prices (manual shift, same as SPY)")

    # SPY initial value for take-profit
    spy_start_value = 100.0
    if spy_shifted is not None and len(spy_shifted) > 0:
        if injection_date in spy_shifted.index:
            spy_start_value = float(spy_shifted.loc[injection_date])
        else:
            spy_start_value = float(spy_shifted.iloc[0])

    for idx, day in enumerate(all_days):
        if idx == 0:
            continue

        # Cash interest every day
        current_cash *= (1 + daily_rate)

        # Apply price changes every day using shifted calendar data
        today_prices = get_all_shifted_prices(day, shifted_calendar_df)
        prev_day = day - timedelta(days=1)
        yesterday_prices = get_all_shifted_prices(prev_day, shifted_calendar_df)

        # Update each asset
        for t in tickers:
            if current_values.get(t, 0) > 0:
                today_price = today_prices.get(t, 0.0)
                yesterday_price = yesterday_prices.get(t, 0.0)
                if yesterday_price > 0 and today_price > 0:
                    change = (today_price / yesterday_price) - 1
                    current_values[t] *= (1 + change)

        total = sum(current_values.values()) + current_cash
        current_weights = {t: current_values[t] / total if total > 0 else 0.0 for t in tickers}

        # Rebalance check (using shifted business-day data).
        # For GARCH and rebalancing we need returns from the shifted business-day data.
        shifted_business = shifted_calendar_df.resample('B').last().ffill()
        returns = shifted_business[tickers].pct_change().dropna()
        returns = returns[returns.index <= pd.Timestamp(day)]

        if len(returns) > lookback_days * 0.5:
            target_weights = optimise_portfolio(returns, RISK_FREE_RATE, tickers)
            target_cash_pct, f_star = calc_cash_allocation(
                returns, RISK_FREE_RATE, kelly_lookback,
                KELLY_BASE_CAP, KELLY_MAX_CAP,
                CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION
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
                            print(f"\n   TAKE-PROFIT on {day.strftime('%Y-%m-%d')}")
                            print(f"      Portfolio: £{total:.2f}, SPY: £{spy_normalized:.2f}")
                            print(f"      Profit locked: £{profit:.2f}")
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
                    current_weights = {t: current_values[t] / total if total > 0 else 0.0 for t in tickers}
                    last_rebalance = day
                    print(f"   Rebalanced on {day.strftime('%Y-%m-%d')}: {reason}")
                    print(f"      Cash: {target_cash_pct*100:.1f}%, f*: {f_star:.3f}")

        # Log this day
        dates.append(day)
        values.append(total)
        log_daily_asset_values(name, day, current_values, current_cash, total)

    print(f"\n   {name} complete: {len(values)} days")
    print(f"   Kelly sign changes: {kelly_sign_changes}")

    return pd.Series(values, index=pd.DatetimeIndex(dates)), injection_date


# ---------------------------------------------------------------------------
# SPY download
# ---------------------------------------------------------------------------

def download_spy(start_date, end_date, injection_date):
    print("   Downloading SPY...")
    try:
        end_dt = pd.Timestamp(end_date) + timedelta(days=5)
        spy = yf.download("SPY", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)["Close"]
        if len(spy) == 0:
            spy = yf.download("^GSPC", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)["Close"]
        if len(spy) > 0:
            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]

            business_days = pd.date_range(start=spy.index.min(), end=spy.index.max(), freq='B')
            spy = spy.reindex(business_days).ffill().bfill()

            injection_ts = pd.Timestamp(injection_date)
            raw_start = injection_ts - pd.Timedelta(days=1)
            raw_spy = spy[spy.index >= raw_start]

            print("\n   Raw SPY prices (before normalisation and shift):")
            print("   " + "-" * 50)
            for date, price in raw_spy.items():
                print(f"      {date.strftime('%Y-%m-%d')}: ${price:.2f}")
            print("   " + "-" * 50)

            spy_shifted = spy.copy()
            new_index = spy.index + pd.Timedelta(days=1)
            spy_shifted.index = new_index
            spy_shifted = spy_shifted[spy_shifted.index >= injection_ts]

            print("\n   SPY prices after shift (using previous-day close, from injection date):")
            print("   " + "-" * 50)
            for date, price in spy_shifted.items():
                print(f"      {date.strftime('%Y-%m-%d')}: ${price:.2f}")
            print("   " + "-" * 50)

            print(f"\n   SPY: {len(spy_shifted)} days (using previous-day close)")
            return spy_shifted
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)
    except Exception as e:
        print(f"   Error: {e}")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)


# ---------------------------------------------------------------------------
# Helpers
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
    min_date = all_dates.min()
    max_date = all_dates.max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='D')
    aligned = {}
    for name, s in data_dict.items():
        if s is not None and len(s) > 0:
            aligned[name] = s.reindex(full_range, method='ffill')
            aligned[name] = aligned[name].ffill()
            aligned[name] = aligned[name].bfill()
            if aligned[name].isna().any():
                first = aligned[name].first_valid_index()
                if first:
                    aligned[name] = aligned[name].fillna(aligned[name].loc[first])
        else:
            aligned[name] = pd.Series(dtype=float)
    return aligned


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_performance(data, injection_date, title, filename, log_scale=False):
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, series in data.items():
        if series is not None and len(series) > 0:
            color = SPY_COLOR if name == "SPY" else UNIVERSE_COLORS.get(name, '#888')
            label = 'SPY (Benchmark)' if name == "SPY" else f'{name} - No Rules'
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
    y_label = 'Portfolio Value (£)' + (' - Log Scale' if log_scale else '')
    ax.set_ylabel(y_label)
    ax.set_title(title + (' (Log Scale)' if log_scale else ''))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    y_off = 0.02
    for name, series in data.items():
        if series is not None and len(series) > 0:
            color = SPY_COLOR if name == "SPY" else UNIVERSE_COLORS.get(name, '#888')
            label = name.replace('_', ' ')
            val = float(series.iloc[-1])
            date_str = series.index[-1].strftime('%d/%m/%Y')
            ax.text(0.02, y_off, f"{label}: £{val:,.2f} ({date_str})",
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
    print("FINAL PERFORMANCE TRACKER (Daily Asset Value CSVs)")
    print("=" * 70)
    print(f"Kelly caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")
    print("=" * 70)

    # Delete old daily CSV files for a clean slate
    for name in ["original_18", "optimal_standard", "optimal_ethical"]:
        log_file = os.path.join(FINAL_LOGS_DIR, f"{name.lower()}_daily.csv")
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"   Removed old {name}_daily.csv")

    # Get injection date from the log
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    first_injection_date = None
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            first_injection_date = pd.to_datetime(df.iloc[0]['date']).date()
    if first_injection_date is None:
        first_injection_date = datetime.now().date()

    # Download and shift SPY
    print("\nDownloading SPY (will shift to previous-day close)...")
    backtest_start = first_injection_date - timedelta(days=30)
    spy_shifted = download_spy(backtest_start.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"), first_injection_date)
    print(f"\n   SPY first date (shifted): {spy_shifted.index[0] if len(spy_shifted) > 0 else 'N/A'}")
    print(f"   SPY last date (shifted): {spy_shifted.index[-1] if len(spy_shifted) > 0 else 'N/A'}")
    print(f"   Injection date: {first_injection_date}")

    universes = {
        "Original_18": (ORIGINAL_UNIVERSE, ORIGINAL_PARAMS),
        "Optimal_Standard": (OPTIMAL_STANDARD_PORTFOLIO, OPTIMAL_STANDARD_PARAMS),
        "Optimal_Ethical": (OPTIMAL_ETHICAL_PORTFOLIO, OPTIMAL_ETHICAL_PARAMS),
    }

    print("\nKelly Lookback Values:")
    for name, (_, params) in universes.items():
        print(f"   {name}: {params['kelly_lookback']} days")
    print("=" * 70)

    all_data = {}
    injection_dates = {}

    for name, (tickers, params) in universes.items():
        print(f"\n{'='*60}\nRUNNING: {name}\n{'='*60}")

        # Fetch price data for this universe only
        raw_prices = fetch_price_data_for_universe(tickers, LOOKBACK_DAYS)
        if raw_prices is None or len(raw_prices) == 0 or len(raw_prices.columns) == 0:
            print(f"   No price data for {name}. Skipping.")
            continue

        # Run the tracker with the raw (unshifted) prices - it will shift internally
        series, inj_date = run_universe_tracker(tickers, params, name, raw_prices, spy_shifted)
        if series is not None:
            all_data[name] = normalise(series, inj_date)
            injection_dates[name] = inj_date

    if len(all_data) == 0:
        print("No data generated. Exiting.")
        return

    all_data["SPY"] = normalise(spy_shifted, first_injection_date)

    print("\nAligning series (daily, including weekends)...")
    all_data = align_series(all_data)
    if all_data:
        first = next(iter(all_data.values()))
        print(f"   Aligned to {len(first)} days")

    print("\n" + "=" * 70)
    print("DAILY PORTFOLIO VALUES (previous-day close)")
    print("=" * 70)

    combined_df = pd.DataFrame()
    for name, series in all_data.items():
        if series is not None and len(series) > 0:
            combined_df[name] = series

    print("\n" + combined_df.to_string(float_format=lambda x: f'{x:.2f}'))

    csv_path = os.path.join(FINAL_LOGS_DIR, "daily_portfolio_values.csv")
    combined_df.to_csv(csv_path, float_format='%.2f')
    print(f"\nDaily portfolio values saved to: {csv_path}")

    print("\n" + "=" * 70)
    print("FINAL VALUES")
    print("=" * 70)
    for name, series in all_data.items():
        if series is not None and len(series) > 0:
            val = float(series.iloc[-1])
            ret = (val / INITIAL_CAPITAL - 1) * 100
            print(f"   {name}: £{val:.2f} ({ret:.1f}%)")

    print("\nGenerating plots...")
    title = f'No Rules Portfolio Performance Since Investment ({first_injection_date.strftime("%d/%m/%Y")})\n(using previous-day close)'
    plot_performance(all_data, first_injection_date, title, "final_performance_tracker.png", log_scale=False)
    plot_performance(all_data, first_injection_date, title, "final_performance_tracker_log.png", log_scale=True)

    print("\n" + "=" * 70)
    print("Daily asset value CSVs saved to:")
    for name in universes.keys():
        print(f"   {name}: {FINAL_LOGS_DIR}/{name.lower()}_daily.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()