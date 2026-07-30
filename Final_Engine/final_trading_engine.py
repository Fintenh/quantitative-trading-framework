"""
final_trading_engine.py - Final Engine for Optimal_Ethical 15-asset portfolio.
Features: GARCH volatility, Kelly sign change detection, take-profit, rebalancing.
Uses CLOSING prices ONLY for everything. No intraday prices.
FIXED: Each day gets its own distinct closing price from Yahoo Finance.
"""

import os
import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# === PATHS & IMPORTS ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for phase in ["Phase_2", "Phase_3"]:
    sys.path.insert(0, os.path.join(parent_dir, phase))

from final_config import *
from data_fetcher import calculate_returns, calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios, generate_portfolio_summary
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

# === STATE MANAGEMENT ===

def get_log_path(filename):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, filename)

def load_state(filename, default=None, parser=None):
    try:
        with open(get_log_path(filename), 'r') as f:
            return parser(f.read().strip()) if parser else f.read().strip()
    except:
        return default

def save_state(filename, value):
    with open(get_log_path(filename), 'w') as f:
        f.write(str(value))

def load_float(filename):
    return load_state(filename, None, float)

def save_float(filename, value):
    save_state(filename, value)

def load_date(filename):
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
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    if df is not None and len(df) > 0:
        return df.iloc[-1]
    return None

def get_last_log_value():
    row = get_last_log_row()
    return float(row['portfolio_value']) if row is not None else None

def get_last_weights():
    row = get_last_log_row()
    if row is not None:
        portfolio_value = float(row['portfolio_value'])
        weights = {}
        for t in TICKERS:
            val = float(row.get(f'{t}_value', 0))
            weights[t] = val / portfolio_value if portfolio_value > 0 else 0.0
        return weights
    return {t: 0.0 for t in TICKERS}

def get_last_asset_values():
    """Get the actual asset values from the last log row."""
    row = get_last_log_row()
    if row is not None:
        return {t: float(row.get(f'{t}_value', 0)) for t in TICKERS}
    return {t: 0.0 for t in TICKERS}

def get_last_cash_pounds():
    row = get_last_log_row()
    return float(row['cash_pounds']) if row is not None else 0.0

def get_last_cash_percentage():
    row = get_last_log_row()
    if row is not None:
        portfolio_value = float(row['portfolio_value'])
        cash_pounds = float(row['cash_pounds'])
        return cash_pounds / portfolio_value if portfolio_value > 0 else 0.0
    return 0.0

def get_last_realised():
    row = get_last_log_row()
    return float(row['realised_profit']) if row is not None else 0.0

# === LOGGING ===

def log_entry(filename, entry, dedup_keys=None):
    log_path = get_log_path(filename)
    df = pd.DataFrame([entry])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        if dedup_keys and len(existing) > 0:
            last = existing.iloc[-1]
            try:
                if all(abs(float(last.get(k, 0)) - float(entry.get(k, 0))) < 0.0001 for k in dedup_keys):
                    return
            except:
                pass
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(log_path, index=False)

def log_portfolio(date, value, weights, cash_pounds, realised, total_wealth):
    """Log portfolio state with the actual date and time."""
    entry_full = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': value,
        'cash_pounds': cash_pounds,
        'realised_profit': realised,
        'total_wealth': total_wealth,
    }
    for t in TICKERS:
        entry_full[f'{t}_value'] = value * weights.get(t, 0.0)
    
    log_entry(PORTFOLIO_LOG_FILE, entry_full, ['portfolio_value', 'cash_pounds', 'realised_profit'])
    
    entry_rounded = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': round(value, 2),
        'cash_pounds': round(cash_pounds, 2),
        'realised_profit': round(realised, 2),
        'total_wealth': round(total_wealth, 2),
    }
    for t in TICKERS:
        entry_rounded[f'{t}_value'] = round(value * weights.get(t, 0.0), 2)
    
    log_entry('portfolio_log_rounded.csv', entry_rounded, ['portfolio_value', 'cash_pounds', 'realised_profit'])

def log_rebalance_event(date, target, cash_pounds, value, cost, reason):
    entry_full = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'cash_pounds': cash_pounds,
        'portfolio_value': value,
        'total_cost': cost,
        'rebalance_reason': reason,
    }
    for t in TICKERS:
        entry_full[f'{t}_target_value'] = value * target.get(t, 0.0)
    
    log_entry(REBALANCE_LOG_FILE, entry_full, ['cash_pounds', 'portfolio_value', 'rebalance_reason'])
    
    entry_rounded = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'cash_pounds': round(cash_pounds, 2),
        'portfolio_value': round(value, 2),
        'total_cost': round(cost, 2),
        'rebalance_reason': reason,
    }
    for t in TICKERS:
        entry_rounded[f'{t}_target_value'] = round(value * target.get(t, 0.0), 2)
    
    log_entry('rebalance_log_rounded.csv', entry_rounded, ['cash_pounds', 'portfolio_value', 'rebalance_reason'])

def log_kelly_sign_change(date, prev, curr):
    log_entry('kelly_sign_changes.csv', {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'prev_f_star': prev,
        'new_f_star': curr,
        'sign_change': 'positive→negative' if prev > 0 and curr <= 0 else 'negative→positive'
    })

def log_take_profit_event(date, profit, new_active, total_realised):
    log_entry(TAKE_PROFIT_LOG_FILE, {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'profit_withdrawn': profit,
        'new_active_value': new_active,
        'total_realised_profit': total_realised
    })

# === HELPERS ===

def daily_cash_rate(annual_rate):
    return (1 + annual_rate) ** (1 / 252) - 1 if annual_rate > 0 else 0.0

def get_spy_value(initial=100.0):
    """Get SPY value using ONLY closing prices. No intraday."""
    try:
        spy = yf.download("SPY", period="5d", progress=False)["Close"]
        if len(spy) == 0:
            return initial
        
        current_price = float(spy.iloc[-1])
        
        start = yf.download("SPY", start="2020-01-02", end="2020-01-03", progress=False)["Close"]
        if len(start) == 0:
            start = yf.download("SPY", start="2020-01-06", end="2020-01-07", progress=False)["Close"]
        if len(start) == 0:
            return initial
        
        start_price = float(start.iloc[0])
        return (current_price / start_price) * initial
    except:
        return initial

def fetch_price_data_with_forward_fill(tickers, lookback_days):
    """Fetch CLOSING prices only with forward-fill for holidays."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)
    all_data = {}
    
    print("\n📥 Downloading data for each ticker...")
    
    for t in tickers:
        try:
            data = yf.download(t, start=start_date, end=end_date, progress=False)
            if len(data) == 0:
                print(f"   ⚠️ {t}: No data returned")
                all_data[t] = pd.Series(dtype=float)
                continue
            if 'Close' in data.columns:
                series = data['Close']
                if len(series) > 0:
                    all_data[t] = series
                    print(f"   ✅ {t}: {len(series)} days of data")
                    continue
            if len(data.columns) > 0:
                series = data.iloc[:, 0]
                if len(series) > 0:
                    all_data[t] = series
                    print(f"   ✅ {t}: {len(series)} days of data")
                    continue
            print(f"   ⚠️ {t}: No usable data")
            all_data[t] = pd.Series(dtype=float)
        except Exception as e:
            print(f"   ⚠️ {t}: Error - {e}")
            all_data[t] = pd.Series(dtype=float)
    
    valid_series = {k: v for k, v in all_data.items() if len(v) > 0}
    if len(valid_series) == 0:
        print("❌ No valid data found for any ticker")
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
    
    print(f"\n📊 Data summary:")
    print(f"   ✅ Aligned to {len(df)} business days")
    print(f"   📊 Columns: {list(df.columns)}")
    
    return df

# === PRICE FUNCTIONS ===

def get_closing_price_for_date(ticker, date, prices_df):
    """
    Get the CLOSING price for a specific ticker on a specific date.
    Downloads from Yahoo Finance for that exact date.
    """
    try:
        # Download data for that specific date
        start = date - timedelta(days=1)
        end = date + timedelta(days=1)
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if len(data) == 0:
            return 0.0
        
        if "Close" in data.columns:
            closes = data["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            
            # Try exact match
            target_ts = pd.Timestamp(date)
            if target_ts in closes.index:
                val = closes.loc[target_ts]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if val > 0:
                    return float(val)
            
            # Try closest date <= target date
            idx = closes.index[closes.index <= target_ts]
            if len(idx) > 0:
                val = closes.loc[idx[-1]]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if val > 0:
                    return float(val)
        
        return 0.0
    except:
        return 0.0

def get_closing_prices_for_date(date, tickers):
    """Get CLOSING prices for all tickers on a specific date."""
    prices = {}
    for t in tickers:
        prices[t] = get_closing_price_for_date(t, date, None)
    return prices

def get_previous_closing_price(ticker, date, prices_df):
    """Get the previous day's closing price from the DataFrame."""
    target_ts = pd.Timestamp(date)
    if ticker in prices_df.columns and len(prices_df) > 0:
        idx = prices_df.index[prices_df.index < target_ts]
        if len(idx) > 0:
            val = prices_df[ticker].loc[idx[-1]]
            if val > 0:
                return float(val)
    return 0.0

def get_trading_days_between(start_date, end_date):
    """Get all business days between two dates (inclusive)."""
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates

def update_portfolio_between_dates(prices_df, asset_values, cash_pounds, from_date, to_date, daily_rate):
    """
    Update portfolio from from_date to to_date using CLOSING prices.
    For each day, gets the closing price from Yahoo Finance.
    Simple: multiply each asset value by its price change.
    Cash earns interest daily.
    Returns a list of (date, asset_values, cash_pounds, weights) for each day.
    """
    trading_days = get_trading_days_between(from_date + timedelta(days=1), to_date)
    
    if not trading_days:
        total = sum(asset_values.values()) + cash_pounds
        weights = {t: asset_values[t] / total if total > 0 else 0.0 for t in TICKERS}
        return [(to_date, asset_values, cash_pounds, weights)]
    
    print(f"\n📅 Updating from {from_date} to {to_date} ({len(trading_days)} trading days)")
    
    results = []
    current_values = asset_values.copy()
    current_cash = cash_pounds
    
    for day in trading_days:
        print(f"\n   📊 Processing {day.strftime('%Y-%m-%d')}:")
        
        # Get TODAY's closing prices from Yahoo Finance
        today_prices = get_closing_prices_for_date(day, TICKERS)
        
        # Get YESTERDAY's closing prices (from the DataFrame or previous day)
        prev_day = day - timedelta(days=1)
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)
        
        # Apply price changes to each asset
        for t in TICKERS:
            if current_values[t] > 0:
                # Get today's price (from Yahoo Finance)
                today_price = today_prices.get(t, 0.0)
                
                # If today's price is 0, try the DataFrame
                if today_price == 0:
                    today_price = get_closing_price_for_date(t, day, prices_df)
                
                # Get yesterday's price
                yesterday_price = get_previous_closing_price(t, prev_day, prices_df)
                
                if yesterday_price > 0 and today_price > 0:
                    return_pct = (today_price / yesterday_price) - 1
                    current_values[t] *= (1 + return_pct)
                    if abs(return_pct) > 0.0001:
                        print(f"      {t}: {return_pct*100:+.2f}% (${yesterday_price:.2f} → ${today_price:.2f})")
                    else:
                        print(f"      {t}: {return_pct*100:+.2f}% (no significant change)")
                else:
                    print(f"      {t}: No valid prices (yesterday: {yesterday_price}, today: {today_price})")
        
        # Apply cash interest
        current_cash *= (1 + daily_rate)
        print(f"      Cash: £{current_cash:.4f} (interest: £{current_cash * daily_rate:.6f})")
        
        # Calculate totals and weights for this day
        day_total = sum(current_values.values()) + current_cash
        day_weights = {t: current_values[t] / day_total if day_total > 0 else 0.0 for t in TICKERS}
        
        # Store result for this day
        results.append((day, current_values.copy(), current_cash, day_weights))
        
        # Also log a summary
        print(f"      Total: £{day_total:.4f}")
    
    return results

# ============================================================================
# REBALANCE FUNCTIONS
# ============================================================================

def calculate_drift(current, target):
    max_drift, max_asset = 0.0, None
    for t in current:
        drift = abs(current.get(t, 0.0) - target.get(t, 0.0))
        if drift > max_drift:
            max_drift, max_asset = drift, t
    return max_drift, max_asset

def should_rebalance(date, last_date, current, target):
    if last_date is None:
        return True, "First run", 0.0, None
    
    days = (date - last_date).days
    drift, asset = calculate_drift(current, target)
    
    if sum(current.values()) < 0.01:
        return True, "First run", drift, asset
    if days < REBALANCE_MIN_DAYS:
        return False, f"Min days ({days}/{REBALANCE_MIN_DAYS})", drift, asset
    if days >= REBALANCE_MAX_DAYS:
        return True, f"Time-based: {days} days", drift, asset
    if drift > DRIFT_THRESHOLD:
        return True, f"Drift: {asset} {drift*100:.2f}%", drift, asset
    return False, f"Waiting ({days} days, drift {drift*100:.2f}%)", drift, asset

def generate_orders(value, current_weights, target_weights, cash_pounds, target_cash_pounds):
    orders = {}
    total_trade = 0.0
    
    cash_diff = target_cash_pounds - cash_pounds
    if abs(cash_diff) > 0.01:
        orders['CASH'] = {'action': 'BUY' if cash_diff > 0 else 'SELL', 'amount': abs(cash_diff)}
        total_trade += abs(cash_diff)
    
    for t in TICKERS:
        current_val = value * current_weights.get(t, 0.0)
        target_val = value * target_weights.get(t, 0.0)
        diff = target_val - current_val
        if abs(diff) > 0.01:
            orders[t] = {'action': 'BUY' if diff > 0 else 'SELL', 'amount': abs(diff)}
            total_trade += abs(diff)
    
    return orders, total_trade * ROUND_TRIP_COST_PCT

# === MAIN ===

def main():
    now = datetime.now()
    today = now.date()
    current_time = now.time()
    
    # Skip if already updated today
    last_update = load_state(LAST_UPDATE_DATE_FILE, None, lambda s: datetime.strptime(s, "%Y-%m-%d").date())
    if last_update == today:
        print(f"⚠️ Already updated today ({today}). Skipping.")
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
    
    # === STEP 1: DATA ===
    print("\nSTEP 1: Fetching Data (Closing Prices Only)")
    prices_df = fetch_price_data_with_forward_fill(TICKERS, LOOKBACK_DAYS)
    
    if prices_df is None or len(prices_df) == 0 or len(prices_df.columns) == 0:
        print("❌ No price data available. Using existing portfolio values.")
        value = get_last_log_value() or INITIAL_CAPITAL
        weights = get_last_weights()
        cash_pounds = get_last_cash_pounds()
        realised = get_last_realised()
        log_portfolio(now, value, weights, cash_pounds, realised, value + realised)
        return
    
    returns = calculate_returns(prices_df)
    exp_ret, cov, _ = calculate_annualised_stats(returns)
    spy_value = get_spy_value(INITIAL_CAPITAL)
    print(f"   SPY: £{spy_value:.2f}")
    
    # === STEP 2: UPDATE PORTFOLIO ===
    print("\nSTEP 2: Updating Portfolio Value")
    daily_rate = daily_cash_rate(CASH_INTEREST_RATE)
    
    value = get_last_log_value()
    realised = get_last_realised()
    is_first = value is None
    
    if is_first:
        value = INITIAL_CAPITAL
        cash_pounds = 0.0
        cash_percentage = 0.0
        asset_values = {t: 0.0 for t in TICKERS}
        weights = {t: 0.0 for t in TICKERS}
        print(f"   First run - starting at £{value:.2f}")
        
        log_datetime = datetime.combine(today, current_time)
        total_wealth = value + realised
        log_portfolio(log_datetime, value, weights, cash_pounds, realised, total_wealth)
        save_state(LAST_UPDATE_DATE_FILE, today.strftime("%Y-%m-%d"))
    else:
        weights = get_last_weights()
        asset_values = get_last_asset_values()
        cash_pounds = get_last_cash_pounds()
        cash_percentage = get_last_cash_percentage()
        
        last_date = load_date(LAST_UPDATE_DATE_FILE)
        if last_date is None:
            last_date = today - timedelta(days=1)
            while last_date.weekday() >= 5:
                last_date -= timedelta(days=1)
        
        if last_date != today:
            print(f"   Last logged: {last_date}, Today: {today}")
            
            # Get updates for all days between
            update_results = update_portfolio_between_dates(
                prices_df, asset_values, cash_pounds, last_date, today, daily_rate
            )
            
            # Log each day in between
            for day, day_values, day_cash, day_weights in update_results:
                day_total = sum(day_values.values()) + day_cash
                day_total_wealth = day_total + realised
                
                log_datetime = datetime.combine(day, current_time)
                log_portfolio(log_datetime, day_total, day_weights, day_cash, realised, day_total_wealth)
                print(f"\n✅ Logged {day.strftime('%Y-%m-%d')}: £{day_total:.4f}")
                
                # Update current state
                asset_values = day_values
                cash_pounds = day_cash
                weights = day_weights
            
            # Final state
            value = sum(asset_values.values()) + cash_pounds
        else:
            value = sum(asset_values.values()) + cash_pounds
    
    # === STEP 2.5: TAKE-PROFIT ===
    force_rebalance = False
    if RELATIVE_TAKE_PROFIT_PCT > 0:
        target = spy_value * (1 + RELATIVE_TAKE_PROFIT_PCT)
        print(f"\n📊 Take-Profit: £{value:.2f} vs £{target:.2f}")
        if value > target:
            profit = value - spy_value
            realised += profit
            value = spy_value
            force_rebalance = True
            print(f"💰 Take-Profit: £{profit:.2f} locked")
            log_take_profit_event(now, profit, value, realised)
    
    # === STEP 3: OPTIMISE PORTFOLIO ===
    print("\nSTEP 3: Optimising Portfolio")
    try:
        opt = optimise_portfolios(exp_ret, cov, RISK_FREE_RATE)
        print(generate_portfolio_summary(opt))
        target_weights = {t: opt['msr_weights'][i] for i, t in enumerate(TICKERS)}
    except Exception as e:
        print(f"   ⚠️ Optimisation failed: {e}, using equal weights")
        target_weights = {t: 1.0/len(TICKERS) for t in TICKERS}
    
    # === STEP 4: GARCH + KELLY ===
    print("\nSTEP 4: GARCH + Kelly")
    try:
        models, _, _ = fit_garch_for_assets(returns)
        avg_vol = get_average_volatility(get_latest_volatility(models, returns))
    except:
        avg_vol = returns.std().mean() * np.sqrt(252)
    print(f"   Volatility: {avg_vol*100:.2f}%")
    
    kelly_returns = returns.iloc[-KELLY_LOOKBACK:] if len(returns) >= KELLY_LOOKBACK else returns
    try:
        ek, ck, _ = calculate_annualised_stats(kelly_returns)
        ok = optimise_portfolios(ek, ck, RISK_FREE_RATE)
        wk = ok['msr_weights']
        mu = np.sum(ek * wk)
        sigma = np.sqrt(wk.T @ ck @ wk)
        f_star = (mu - RISK_FREE_RATE) / (sigma ** 2) if sigma > 0 else 0.0
    except:
        f_star = 0.0
    
    prev_f = load_float('kelly_state.txt')
    if prev_f is not None and (prev_f > 0) != (f_star > 0):
        print(f"🔄 Kelly sign change: {prev_f:.4f} → {f_star:.4f}")
        log_kelly_sign_change(now, prev_f, f_star)
    save_float('kelly_state.txt', f_star)
    
    cash_cap = KELLY_BASE_CAP if f_star > 0 else KELLY_MAX_CAP
    if avg_vol <= CASH_MIN_VOLATILITY:
        target_cash_percentage = 0.0
    elif avg_vol >= CASH_MAX_VOLATILITY:
        target_cash_percentage = cash_cap
    else:
        fraction = (avg_vol - CASH_MIN_VOLATILITY) / (CASH_MAX_VOLATILITY - CASH_MIN_VOLATILITY)
        target_cash_percentage = fraction * cash_cap
    
    target_cash_pounds = value * target_cash_percentage
    
    print(f"   f*: {f_star:.3f}")
    print(f"   Current Cash: £{cash_pounds:.2f} ({cash_percentage*100:.1f}%)")
    print(f"   Target Cash: £{target_cash_pounds:.2f} ({target_cash_percentage*100:.1f}%)")
    print(f"   Cash Cap: {cash_cap*100:.0f}%")
    
    adjusted_target = {t: w * (1 - target_cash_percentage) for t, w in target_weights.items()}
    
    # === STEP 5: REBALANCE CHECK ===
    print("\nSTEP 5: Rebalance Check")
    last_rebalance = load_date(LAST_REBALANCE_FILE)
    now_date = now.date()
    rebalance_needed, reason, drift, asset = should_rebalance(now_date, last_rebalance, weights, adjusted_target)
    
    print(f"   Last: {last_rebalance.strftime('%Y-%m-%d') if last_rebalance else 'Never'}")
    print(f"   Needed: {rebalance_needed} - {reason}")
    
    # === STEP 6: EXECUTE ===
    if rebalance_needed:
        print("\nSTEP 6: Executing Rebalance")
        orders, cost = generate_orders(value, weights, adjusted_target, cash_pounds, target_cash_pounds)
        print(f"   Value: £{value:.2f}")
        print(f"   Current Cash: £{cash_pounds:.2f} → Target Cash: £{target_cash_pounds:.2f}")
        print(f"   Reason: {reason}")
        
        if orders:
            print("\nOrders:")
            for ticker, order in orders.items():
                print(f"   {order['action']} {ticker}: £{order['amount']:.2f}")
        else:
            print("  No orders needed")
        
        print(f"   Cost: £{cost:.4f} ({ROUND_TRIP_COST_PCT*100:.1f}%)")
        
        save_state(LAST_REBALANCE_FILE, now_date.strftime("%Y-%m-%d"))
        log_rebalance_event(now, adjusted_target, target_cash_pounds, value, cost, reason)
        
        cash_pounds = target_cash_pounds
        cash_percentage = target_cash_percentage
        weights = adjusted_target.copy()
        asset_values = {t: value * weights.get(t, 0.0) for t in TICKERS}
        
        total_wealth = value + realised
        log_portfolio(now, value, weights, cash_pounds, realised, total_wealth)
    else:
        days_until = REBALANCE_MAX_DAYS - (now_date - last_rebalance).days if last_rebalance else REBALANCE_MAX_DAYS
        print(f"\n✅ No rebalance. Next in {days_until} days")
    
    # === SAVE STATE ===
    save_state(LAST_UPDATE_DATE_FILE, today.strftime("%Y-%m-%d"))
    
    # === SUMMARY ===
    total_wealth = value + realised
    print("\n" + "=" * 60)
    print("FINAL ENGINE COMPLETE")
    print(f"Active: £{value:.2f}, Cash: £{cash_pounds:.2f} ({cash_percentage*100:.1f}%)")
    print(f"Realised: £{realised:.2f}, Total: £{total_wealth:.2f}")
    print(f"Return: {(total_wealth / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()