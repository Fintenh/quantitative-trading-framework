"""
final_trading_engine.py - Final engine for the Optimal_Ethical 15-asset portfolio.
Features: GARCH volatility, Kelly sign change detection, take-profit, rebalancing.
Uses previous-day closing prices for everything (price on date D is the close of D-1).
Cash interest is applied daily (including weekends).
Logging always creates entries (no deduplication).
GARCH model selection output is silenced.

Dividend handling:
- Dividends are checked ONLY when a rebalance is triggered
- Uses yfinance to fetch actual dividend history since the last rebalance
- Shares are calculated from the log (asset value / price at last rebalance)
- Dividends are added to cash before rebalancing
- The existing rebalance logic automatically reinvests dividends into assets
- No separate CSV files or manual updates required

Universe-agnostic:
- The engine reads ALL asset values from the log, not just those in TICKERS
- When TICKERS changes, old assets are kept until the next rebalance
- At rebalance, old assets are sold and new ones are bought according to target weights

FIXES (2026-09-10):
1. fetch_price_data_with_forward_fill uses the UNION of all trading dates.
2. main() fetches prices for ALL assets held in the log, not just TICKERS.
3. Take-profit: trigger when portfolio return since injection beats SPY return
   since injection by take_profit_pct, then reduce the active portfolio to
   SPY's equivalent value and lock the difference as realised profit.
4. Transaction cost is deducted from the portfolio total on rebalance.
5. On first run, weights are fractions of total portfolio value, not GBP values.
6. get_last_weights() includes cash in the denominator.
7. main() processes each day individually, interleaving price updates, per-day
   returns, per-day rebalance checks, and per-day logging - matching the
   behaviour of final_performance_tracker.py. Each calendar day is logged
   exactly once, after any rebalance that occurred on that day.
8. optimise_portfolio() helper restored (was omitted in the previous rewrite).
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# --- Paths & imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for phase in ["Phase_2", "Phase_3"]:
    sys.path.insert(0, os.path.join(parent_dir, phase))

from final_config import *
from data_fetcher import calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios, generate_portfolio_summary
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def get_log_path(filename):
    """Return the full path to a file in the logs directory."""
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, filename)


def load_state(filename, default=None, parser=None):
    """Load a scalar state value from a text file, with optional parsing."""
    try:
        with open(get_log_path(filename), 'r') as f:
            return parser(f.read().strip()) if parser else f.read().strip()
    except:
        return default


def save_state(filename, value):
    """Save a scalar state value to a text file."""
    with open(get_log_path(filename), 'w') as f:
        f.write(str(value))


def load_float(filename):
    """Load a float from a state file, returning None if missing or unparseable."""
    return load_state(filename, None, float)


def save_float(filename, value):
    """Save a float to a state file."""
    save_state(filename, value)


def load_date(filename):
    """Load a date from a state file. Accepts either YYYY-MM-DD or YYYY-MM-DD HH:MM:SS."""
    filepath = get_log_path(filename)
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            try:
                return datetime.strptime(content, "%Y-%m-%d").date()
            except ValueError:
                return datetime.strptime(content, "%Y-%m-%d %H:%M:%S").date()
    except:
        return None


def get_last_log_row():
    """Return the last row of the portfolio log as a pandas Series, or None."""
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    if df is not None and len(df) > 0:
        return df.iloc[-1]
    return None


def get_last_log_value():
    """Return the last logged portfolio value, or None if no log exists."""
    row = get_last_log_row()
    return float(row['portfolio_value']) if row is not None else None


def get_injection_date_from_log():
    """Return the injection date from the first row of the portfolio log, or None."""
    log_file = get_log_path(PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            return pd.to_datetime(df.iloc[0]['date']).date()
    return None


def get_last_weights():
    """Get the last logged portfolio weights for ALL assets in the log."""
    asset_values = get_last_asset_values()
    if asset_values:
        total_assets = sum(asset_values.values())
        total = total_assets + get_last_cash_pounds()
        if total > 0:
            return {t: v / total for t, v in asset_values.items()}
    return {}


def get_last_asset_values():
    """Get the last logged asset values for ALL assets from the log file."""
    row = get_last_log_row()
    if row is not None:
        values = {}
        for col in row.index:
            if col.endswith('_value') and col not in ['portfolio_value', 'cash_pounds', 'realised_profit', 'total_wealth']:
                ticker = col.replace('_value', '')
                val = row[col]
                try:
                    if pd.isna(val):
                        values[ticker] = 0.0
                    else:
                        values[ticker] = float(val) if val > 0 else 0.0
                except:
                    values[ticker] = 0.0
        return values
    return {}


def get_last_cash_pounds():
    """Return the last logged cash balance in pounds."""
    row = get_last_log_row()
    return float(row['cash_pounds']) if row is not None else 0.0


def get_last_realised():
    """Return the last logged realised profit (from take-profit events)."""
    row = get_last_log_row()
    return float(row['realised_profit']) if row is not None else 0.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_entry(filename, entry):
    """Append a single row to a CSV log file (creating it if needed)."""
    log_path = get_log_path(filename)
    df = pd.DataFrame([entry])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(log_path, index=False)


def log_portfolio(date, value, weights, cash_pounds, realised, total_wealth):
    """Log portfolio state. Logs ALL assets present in the weights dict."""
    entry_full = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': value,
        'cash_pounds': cash_pounds,
        'realised_profit': realised,
        'total_wealth': total_wealth,
    }
    for ticker, weight in weights.items():
        entry_full[f'{ticker}_value'] = value * weight if value > 0 else 0.0
    log_entry(PORTFOLIO_LOG_FILE, entry_full)

    entry_rounded = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': round(value, 2) if value else 0.0,
        'cash_pounds': round(cash_pounds, 2) if cash_pounds else 0.0,
        'realised_profit': round(realised, 2) if realised else 0.0,
        'total_wealth': round(total_wealth, 2) if total_wealth else 0.0,
    }
    for ticker, weight in weights.items():
        entry_rounded[f'{ticker}_value'] = round(value * weight, 2) if value else 0.0
    log_entry('portfolio_log_rounded.csv', entry_rounded)


def log_rebalance_event(date, target, cash_pounds, value, cost, reason):
    """Log a rebalance event. The target dict contains target weights for TICKERS."""
    entry_full = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'cash_pounds': cash_pounds,
        'portfolio_value': value,
        'total_cost': cost,
        'rebalance_reason': reason,
    }
    for ticker, weight in target.items():
        entry_full[f'{ticker}_target_value'] = value * weight
    log_entry(REBALANCE_LOG_FILE, entry_full)

    entry_rounded = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'cash_pounds': round(cash_pounds, 2),
        'portfolio_value': round(value, 2),
        'total_cost': round(cost, 4),
        'rebalance_reason': reason,
    }
    for ticker, weight in target.items():
        entry_rounded[f'{ticker}_target_value'] = round(value * weight, 2)
    log_entry('rebalance_log_rounded.csv', entry_rounded)


def log_kelly_sign_change(date, prev, curr):
    """Log a change in the sign of the Kelly fraction (f*)."""
    log_entry('kelly_sign_changes.csv', {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'prev_f_star': prev,
        'new_f_star': curr,
        'sign_change': 'positive->negative' if prev > 0 and curr <= 0 else 'negative->positive'
    })


def log_take_profit_event(date, profit, new_active, total_realised):
    """Log a take-profit event where profit is withdrawn from the active portfolio."""
    log_entry(TAKE_PROFIT_LOG_FILE, {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'profit_withdrawn': profit,
        'new_active_value': new_active,
        'total_realised_profit': total_realised
    })


def log_dividend_event(date, total_dividends, ticker_breakdown):
    """Log dividend income received since the last rebalance (for audit)."""
    log_entry('dividend_log.csv', {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'total_dividend_amount': total_dividends,
        'ticker_breakdown': str(ticker_breakdown),
        'rebalance_triggered': True
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def daily_cash_rate(annual_rate):
    """Convert an annual cash rate to a daily compounded rate."""
    return (1 + annual_rate) ** (1 / 365) - 1 if annual_rate > 0 else 0.0


def get_spy_equivalent_value(injection_date, initial=INITIAL_CAPITAL):
    """Return what £initial invested in SPY at the injection date would be worth now."""
    try:
        end = datetime.now() + timedelta(days=1)
        start = injection_date - timedelta(days=5)
        spy = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), progress=False)
        if len(spy) == 0:
            return initial
        close = spy['Close'] if 'Close' in spy.columns else spy.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if len(close) == 0:
            return initial
        inj_rows = close.index[close.index.date >= injection_date]
        if len(inj_rows) == 0:
            return initial
        inj_price = float(close.loc[inj_rows[0]])
        current_price = float(close.iloc[-1])
        if inj_price <= 0:
            return initial
        return (current_price / inj_price) * initial
    except Exception:
        return initial


def fetch_price_data_with_forward_fill(tickers, lookback_days):
    """Fetch closing prices with per-ticker forward-fill (union of dates)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)
    all_data = {}

    print("\nDownloading data for each ticker...")
    for t in tickers:
        try:
            data = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if len(data) == 0:
                all_data[t] = pd.Series(dtype=float)
                continue
            if 'Close' in data.columns:
                series = data['Close']
                if len(series) > 0:
                    all_data[t] = series
                    continue
            if len(data.columns) > 0:
                series = data.iloc[:, 0]
                if len(series) > 0:
                    all_data[t] = series
                    continue
            all_data[t] = pd.Series(dtype=float)
        except:
            all_data[t] = pd.Series(dtype=float)

    valid_series = {k: v for k, v in all_data.items() if len(v) > 0}
    if not valid_series:
        return pd.DataFrame()

    all_dates = pd.DatetimeIndex([])
    for series in valid_series.values():
        all_dates = all_dates.union(series.index)

    if len(all_dates) == 0:
        return pd.DataFrame()

    business_days = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='B')
    df = pd.DataFrame(index=business_days)

    for ticker, series in valid_series.items():
        df[ticker] = series.reindex(business_days).ffill().bfill().fillna(0)

    for t in tickers:
        if t not in df.columns:
            df[t] = 0.0

    print(f"\nData summary: {len(df)} business days, {len(df.columns)} tickers")
    return df


# ---------------------------------------------------------------------------
# Price functions
# ---------------------------------------------------------------------------

def get_all_shifted_prices(date, shifted_calendar_df):
    """Return a dict of prices for a given date from the shifted calendar."""
    prices = {}
    date_ts = pd.Timestamp(date)
    if date_ts in shifted_calendar_df.index:
        row = shifted_calendar_df.loc[date_ts]
        for t in shifted_calendar_df.columns:
            prices[t] = float(row[t]) if row[t] > 0 else 0.0
    else:
        idx = shifted_calendar_df.index[shifted_calendar_df.index <= date_ts]
        if len(idx) > 0:
            row = shifted_calendar_df.loc[idx[-1]]
            for t in shifted_calendar_df.columns:
                prices[t] = float(row[t]) if row[t] > 0 else 0.0
        else:
            for t in shifted_calendar_df.columns:
                prices[t] = 0.0
    return prices


def get_all_days_between(start_date, end_date):
    """Return a list of every calendar day between two dates, inclusive."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def build_shifted_calendar(prices_df):
    """Build a shifted calendar (date D -> close of D-1, calendar-reindexed)."""
    business_days = pd.date_range(start=prices_df.index.min(), end=prices_df.index.max(), freq='B')
    prices_aligned = prices_df.reindex(business_days).ffill().bfill()
    shifted = prices_aligned.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    calendar_index = pd.date_range(start=shifted.index.min(), end=shifted.index.max(), freq='D')
    shifted_calendar = shifted.reindex(calendar_index).ffill()
    return shifted_calendar


# ---------------------------------------------------------------------------
# Dividend helpers
# ---------------------------------------------------------------------------

def convert_to_gbp(amount, from_currency):
    """Convert an amount to GBP using approximate fixed rates."""
    rates = {
        'USD': 0.79, 'EUR': 0.85, 'DKK': 0.115, 'GBP': 1.0,
        'SEK': 0.075, 'NOK': 0.072, 'CHF': 0.88, 'CAD': 0.58,
        'AUD': 0.52, 'JPY': 0.0052, 'CNY': 0.11, 'HKD': 0.10,
        'SGD': 0.59, 'INR': 0.0095,
    }
    rate = rates.get(from_currency, 1.0)
    return amount * rate


def get_all_asset_values_at_date(date):
    """Get ALL asset values from the log at (or before) a given date."""
    log_file = get_log_path(PORTFOLIO_LOG_FILE)
    if not os.path.exists(log_file):
        return None
    df = pd.read_csv(log_file)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df_filtered = df[df['date'] <= date]
    if df_filtered.empty:
        return None
    row = df_filtered.iloc[-1]
    values = {}
    for col in row.index:
        if col.endswith('_value') and col not in ['portfolio_value', 'cash_pounds', 'realised_profit', 'total_wealth']:
            ticker = col.replace('_value', '')
            val = row[col]
            if pd.isna(val):
                values[ticker] = 0.0
            else:
                values[ticker] = float(val) if val > 0 else 0.0
    return values


def get_dividends_since(ticker, date_from, date_to, shares):
    """Get total dividends paid for a ticker between two dates."""
    if shares <= 0:
        return 0.0
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        if dividends.empty:
            return 0.0
        info = stock.info
        currency = info.get('currency', 'USD')
        total = 0.0
        for ex_date, div_per_share in dividends.items():
            ex_date_dt = ex_date.date()
            if date_from < ex_date_dt <= date_to:
                amount = shares * float(div_per_share)
                if currency != 'GBP':
                    amount = convert_to_gbp(amount, currency)
                total += amount
        return total
    except Exception:
        return 0.0


def get_all_dividends_since(date_from, date_to, asset_values_at_start, prices_at_start):
    """Compute total dividends across all assets between two dates."""
    total = 0.0
    breakdown = {}
    for ticker, value in asset_values_at_start.items():
        if value <= 0:
            continue
        price = prices_at_start.get(ticker, 0.0)
        if price <= 0:
            continue
        shares = value / price
        div_amount = get_dividends_since(ticker, date_from, date_to, shares)
        if div_amount > 0:
            breakdown[ticker] = div_amount
            total += div_amount
    return total, breakdown


# ---------------------------------------------------------------------------
# Rebalance helpers
# ---------------------------------------------------------------------------

def calculate_drift(current, target):
    """Return the largest absolute drift and the asset that has it."""
    max_drift, max_asset = 0.0, None
    for t in current:
        drift = abs(current.get(t, 0.0) - target.get(t, 0.0))
        if drift > max_drift:
            max_drift, max_asset = drift, t
    return max_drift, max_asset


def generate_orders(value, current_weights, target_weights, cash_pounds, target_cash_pounds):
    """Generate buy/sell orders and total cost."""
    orders = {}
    total_trade = 0.0
    cash_diff = target_cash_pounds - cash_pounds
    if abs(cash_diff) > 0.01:
        orders['CASH'] = {'action': 'BUY' if cash_diff > 0 else 'SELL', 'amount': abs(cash_diff)}
        total_trade += abs(cash_diff)

    all_assets = set(current_weights.keys()) | set(target_weights.keys())
    for ticker in all_assets:
        current_val = value * current_weights.get(ticker, 0.0)
        target_val = value * target_weights.get(ticker, 0.0)
        diff = target_val - current_val
        if abs(diff) > 0.01:
            orders[ticker] = {'action': 'BUY' if diff > 0 else 'SELL', 'amount': abs(diff)}
            total_trade += abs(diff)

    return orders, total_trade * ROUND_TRIP_COST_PCT


# ---------------------------------------------------------------------------
# Portfolio optimisation wrapper
# ---------------------------------------------------------------------------

def optimise_portfolio(returns_window, risk_free_rate, tickers):
    """
    Return a dict of MSR weights for the given tickers, or equal weights
    if the optimisation fails.
    """
    try:
        exp_ret, cov, _ = calculate_annualised_stats(returns_window)
        opt = optimise_portfolios(exp_ret, cov, risk_free_rate)
        return {t: opt['msr_weights'][i] for i, t in enumerate(tickers)}
    except Exception:
        return {t: 1.0/len(tickers) for t in tickers}


# ---------------------------------------------------------------------------
# GARCH wrappers
# ---------------------------------------------------------------------------

def calc_cash_allocation(returns, risk_free_rate, kelly_lookback, kelly_base_cap, kelly_max_cap,
                         cash_min_volatility, cash_max_volatility, cash_max_allocation):
    """Determine the target cash allocation using Kelly (cap) and GARCH (scaling)."""
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


# ---------------------------------------------------------------------------
# Initial state (used on the very first run only)
# ---------------------------------------------------------------------------

def compute_initial_state(tickers, unshifted_df, injection_date, price_date=None):
    """Compute initial portfolio state for the very first run."""
    if price_date is None:
        available_dates = unshifted_df.index[unshifted_df.index < pd.Timestamp(injection_date)]
        if len(available_dates) > 0:
            price_date = available_dates[-1].date()
        else:
            price_date = unshifted_df.index[0].date()

    returns = unshifted_df[tickers].pct_change().dropna()
    returns = returns[returns.index <= pd.Timestamp(price_date)]

    if len(returns) > LOOKBACK_DAYS:
        returns_window = returns.iloc[-LOOKBACK_DAYS:]
    else:
        returns_window = returns

    if len(returns_window) < LOOKBACK_DAYS * 0.5:
        n = len(tickers)
        weights = {t: 1.0/n for t in tickers}
    else:
        exp_ret, cov, _ = calculate_annualised_stats(returns_window)
        opt = optimise_portfolios(exp_ret, cov, RISK_FREE_RATE)
        weights = {t: opt['msr_weights'][i] for i, t in enumerate(tickers)}

    target_cash_pct, _ = calc_cash_allocation(
        returns_window, RISK_FREE_RATE, KELLY_LOOKBACK,
        KELLY_BASE_CAP, KELLY_MAX_CAP,
        CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION
    )

    cash = INITIAL_CAPITAL * target_cash_pct
    asset_values = {t: (INITIAL_CAPITAL - cash) * weights.get(t, 0.0) for t in tickers}
    for t in asset_values:
        if asset_values[t] < 0:
            asset_values[t] = 0.0
    return injection_date, asset_values, cash, price_date


# ---------------------------------------------------------------------------
# Per-day processing
# ---------------------------------------------------------------------------

def process_day(day, current_values, current_cash, realised, last_rebalance,
                prev_f, shifted_calendar, injection_date, current_time, is_first_day=False):
    """
    Process a single calendar day:
    1. Apply cash interest
    2. Apply price changes
    3. Check take-profit
    4. Compute returns up to this day
    5. Recompute target weights and Kelly
    6. Check rebalance conditions
    7. Execute if needed (with cost deduction)
    8. Log the day once

    Returns updated (current_values, current_cash, realised, last_rebalance, prev_f).
    """
    # --- 1. Cash interest ---
    current_cash *= (1 + daily_cash_rate(CASH_INTEREST_RATE))

    # --- 2. Price changes (using shifted calendar) ---
    today_prices = get_all_shifted_prices(day, shifted_calendar)
    prev_day = day - timedelta(days=1)
    yesterday_prices = get_all_shifted_prices(prev_day, shifted_calendar)

    for ticker, val in current_values.items():
        if val > 0:
            tp = today_prices.get(ticker, 0.0)
            yp = yesterday_prices.get(ticker, 0.0)
            if yp > 0 and tp > 0:
                current_values[ticker] *= (tp / yp)

    value = sum(current_values.values()) + current_cash
    current_weights = {t: current_values[t] / value if value > 0 else 0.0 for t in current_values}

    # --- 3. Take-profit check ---
    take_profit_triggered = False
    if RELATIVE_TAKE_PROFIT_PCT > 0:
        spy_equiv = get_spy_equivalent_value(injection_date, INITIAL_CAPITAL)
        spy_return = spy_equiv / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 1.0
        port_return = value / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 1.0
        if port_return > spy_return * (1 + RELATIVE_TAKE_PROFIT_PCT):
            new_total = spy_return * INITIAL_CAPITAL
            profit = value - new_total
            if profit > 0:
                print(f"   TAKE-PROFIT on {day}: £{profit:.2f} "
                      f"(portfolio {port_return*100:.1f}% vs SPY {spy_return*100:.1f}%)")
                realised += profit
                scale = new_total / value if value > 0 else 1.0
                for t in current_values:
                    current_values[t] *= scale
                current_cash *= scale
                value = new_total
                current_weights = {t: current_values[t] / value if value > 0 else 0.0 for t in current_values}
                take_profit_triggered = True
                log_take_profit_event(day, profit, value, realised)

    # --- 4. Compute returns up to this day (shifted) ---
    shifted_business = shifted_calendar.resample('B').last().ffill()
    returns_all = shifted_business[TICKERS].pct_change().dropna()
    returns_all = returns_all[returns_all.index <= pd.Timestamp(day)]

    if len(returns_all) > LOOKBACK_DAYS:
        returns_window = returns_all.iloc[-LOOKBACK_DAYS:]
    else:
        returns_window = returns_all

    if len(returns_window) <= LOOKBACK_DAYS * 0.5:
        # Not enough data yet; just log the day
        weights = {t: current_values[t] / value if value > 0 else 0.0 for t in current_values}
        log_portfolio(datetime.combine(day, current_time), value, weights, current_cash, realised, value + realised)
        print(f"   Logged {day.strftime('%Y-%m-%d')}: £{value:.4f} (insufficient history)")
        return current_values, current_cash, realised, last_rebalance, prev_f

    # --- 5. Target weights and Kelly ---
    target_weights = optimise_portfolio(returns_window, RISK_FREE_RATE, TICKERS)
    target_cash_pct, f_star = calc_cash_allocation(
        returns_window, RISK_FREE_RATE, KELLY_LOOKBACK,
        KELLY_BASE_CAP, KELLY_MAX_CAP,
        CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION
    )

    # --- 6. Kelly sign change ---
    force_rebalance = False
    if prev_f is not None and (prev_f > 0) != (f_star > 0):
        print(f"   Kelly sign change on {day}: {prev_f:.4f} -> {f_star:.4f}")
        log_kelly_sign_change(day, prev_f, f_star)
        force_rebalance = True
    prev_f = f_star

    adjusted_target = {t: w * (1 - target_cash_pct) for t, w in target_weights.items()}
    target_cash_pounds = value * target_cash_pct

    days_since = (day - last_rebalance).days if last_rebalance else 0
    drift, asset = calculate_drift(current_weights, adjusted_target)

    # --- 7. Rebalance decision ---
    if take_profit_triggered:
        rebalance_needed = True
        reason = "Take-profit"
    elif is_first_day or last_rebalance is None:
        rebalance_needed = True
        reason = "First run"
    elif force_rebalance:
        rebalance_needed = True
        reason = "Kelly sign change"
    elif days_since >= REBALANCE_MAX_DAYS:
        rebalance_needed = True
        reason = f"Time-based: {days_since} days"
    elif days_since >= REBALANCE_MIN_DAYS and drift > DRIFT_THRESHOLD:
        rebalance_needed = True
        reason = f"Drift: {asset} {drift*100:.2f}%"
    else:
        rebalance_needed = False
        reason = ""

    # --- 8. Execute rebalance if needed ---
    if rebalance_needed:
        # Dividend check (only if there is a previous rebalance to measure from)
        if last_rebalance is not None:
            asset_values_at_rebalance = get_all_asset_values_at_date(last_rebalance)
            if asset_values_at_rebalance is not None:
                prices_at_rebalance = {}
                for ticker in asset_values_at_rebalance:
                    price = get_all_shifted_prices(last_rebalance, shifted_calendar).get(ticker, 0.0)
                    if price > 0:
                        prices_at_rebalance[ticker] = price
                dividend_total, dividend_breakdown = get_all_dividends_since(
                    last_rebalance, day, asset_values_at_rebalance, prices_at_rebalance
                )
                if dividend_total > 0:
                    value += dividend_total
                    current_cash += dividend_total
                    target_cash_pounds = value * target_cash_pct
                    log_dividend_event(day, dividend_total, dividend_breakdown)

        orders, cost = generate_orders(value, current_weights, adjusted_target, current_cash, target_cash_pounds)

        # Log the rebalance event BEFORE applying cost so the log reflects the pre-cost value
        log_rebalance_event(day, adjusted_target, target_cash_pounds, value, cost, reason)
        save_state(LAST_REBALANCE_FILE, day.strftime("%Y-%m-%d"))

        # Deduct the transaction cost from the portfolio total
        value -= cost
        target_cash_pounds = value * target_cash_pct

        # Rebuild asset values from the target allocation
        new_values = {}
        for t in current_values:
            new_values[t] = 0.0
        for t in TICKERS:
            new_values[t] = value * adjusted_target.get(t, 0.0)
        current_values = new_values
        current_cash = target_cash_pounds
        value = sum(current_values.values()) + current_cash
        last_rebalance = day

    # --- 9. Log daily state (post-rebalance) ---
    weights = {t: current_values[t] / value if value > 0 else 0.0 for t in current_values}
    log_portfolio(datetime.combine(day, current_time), value, weights, current_cash, realised, value + realised)
    print(f"   Logged {day.strftime('%Y-%m-%d')}: £{value:.4f}")

    return current_values, current_cash, realised, last_rebalance, prev_f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.now()
    today = now.date()
    current_time = now.time()

    last_update = load_state(LAST_UPDATE_DATE_FILE, None, lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    if last_update == today:
        print(f"Already updated today ({today}). Skipping.")
        return

    print("=" * 60)
    print(f"FINAL TRADING ENGINE: OPTIMAL ETHICAL 15")
    print(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {len(TICKERS)}, Lookback: {LOOKBACK_DAYS} days")
    print(f"Rebalance: {REBALANCE_MIN_DAYS}-{REBALANCE_MAX_DAYS} days")
    print(f"Drift: {DRIFT_THRESHOLD*100:.1f}%, Take-Profit: {RELATIVE_TAKE_PROFIT_PCT*100:.0f}%")
    print(f"Kelly Lookback: {KELLY_LOOKBACK} days")
    print(f"Kelly Caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")
    print("=" * 60)

    # --- Step 1: data ---
    print("\nSTEP 1: Fetching Data (Closing Prices Only)")

    all_assets_from_log = []
    log_file = get_log_path(PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        if len(log_df) > 0:
            first_row = log_df.iloc[0]
            for col in first_row.index:
                if col.endswith('_value') and col not in [
                    'portfolio_value', 'cash_pounds',
                    'realised_profit', 'total_wealth'
                ]:
                    ticker = col.replace('_value', '')
                    val = first_row[col]
                    if not pd.isna(val) and val > 0:
                        all_assets_from_log.append(ticker)

    all_tickers_to_fetch = list(set(TICKERS + all_assets_from_log))
    print(f"   Fetching prices for {len(all_tickers_to_fetch)} tickers "
          f"({len(TICKERS)} target, {len(all_assets_from_log)} from log)")

    prices_df = fetch_price_data_with_forward_fill(all_tickers_to_fetch, LOOKBACK_DAYS)

    if prices_df is None or len(prices_df) == 0:
        print("No price data. Using existing portfolio values.")
        value = get_last_log_value() or INITIAL_CAPITAL
        weights = get_last_weights()
        cash_pounds = get_last_cash_pounds()
        realised = get_last_realised()
        log_portfolio(now, value, weights, cash_pounds, realised, value + realised)
        return

    print("\nBuilding shifted calendar (date D -> previous day's close)...")
    shifted_calendar = build_shifted_calendar(prices_df)
    print(f"   Shifted calendar data: {len(shifted_calendar)} days")

    unshifted_df = prices_df.dropna(how='all')
    business_days = pd.date_range(start=unshifted_df.index.min(), end=unshifted_df.index.max(), freq='B')
    unshifted_df = unshifted_df.reindex(business_days).ffill().bfill()

    # --- Step 2: determine starting state ---
    value = get_last_log_value()
    is_first = value is None

    if is_first:
        injection_date = today
        print(f"\nSTEP 2: First run - injection date {injection_date}")
        injection_date, asset_values, cash, price_date = compute_initial_state(
            TICKERS, unshifted_df, injection_date, price_date=None
        )
        current_values = asset_values.copy()
        current_cash = cash
        realised = 0.0
        prev_f = None
        last_rebalance = None
        start_day = injection_date
        save_state('injection_date.txt', injection_date.strftime("%Y-%m-%d"))
    else:
        print(f"\nSTEP 2: Loading state from log")
        current_values = get_last_asset_values()
        current_cash = get_last_cash_pounds()
        realised = get_last_realised()
        injection_date = get_injection_date_from_log() or today
        prev_f = load_float('kelly_state.txt')
        last_rebalance = load_date(LAST_REBALANCE_FILE)

        last_date = load_date(LAST_UPDATE_DATE_FILE)
        if last_date is None:
            last_date = today - timedelta(days=1)
            while last_date.weekday() >= 5:
                last_date -= timedelta(days=1)

        start_day = last_date + timedelta(days=1)
        print(f"   Resuming from {start_day} to {today}")

    # --- Step 3: process each day individually ---
    print(f"\nSTEP 3: Processing days")
    all_days = get_all_days_between(start_day, today)

    for idx, day in enumerate(all_days):
        is_first_day = is_first and idx == 0
        current_values, current_cash, realised, last_rebalance, prev_f = process_day(
            day, current_values, current_cash, realised, last_rebalance,
            prev_f, shifted_calendar, injection_date, current_time,
            is_first_day=is_first_day
        )

    # --- Step 4: save state ---
    if prev_f is not None:
        save_float('kelly_state.txt', prev_f)
    save_state(LAST_UPDATE_DATE_FILE, today.strftime("%Y-%m-%d"))

    # --- Summary ---
    value = sum(current_values.values()) + current_cash
    total_wealth = value + realised
    print("\n" + "=" * 60)
    print("FINAL ENGINE COMPLETE")
    print(f"Active: £{value:.2f}, Cash: £{current_cash:.2f}")
    print(f"Realised: £{realised:.2f}, Total: £{total_wealth:.2f}")
    print(f"Return: {(total_wealth / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()