"""
final_trading_engine.py - Final Engine for Optimal_Ethical 15-asset portfolio.
Features: GARCH volatility, Kelly sign change detection, take-profit, rebalancing.
Multi-exchange holiday handling with forward-fill for closed markets.
FIXED: Cash is treated as a real asset - only changes via interest or rebalance.
FIXED: Full precision stored for accurate tracking, rounded version for readability.
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

def load_date(filename):
    filepath = get_log_path(filename)
    if not os.path.exists(filepath):
        return datetime.now().date()
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            try:
                return datetime.strptime(content, "%Y-%m-%d").date()
            except ValueError:
                return datetime.strptime(content, "%Y-%m-%d %H:%M:%S").date()
    except:
        return datetime.now().date()

def load_float(filename):
    return load_state(filename, None, float)

def get_last_log_value():
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    return float(df.iloc[-1]['portfolio_value']) if df is not None and len(df) > 0 else None

def get_last_weights():
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    if df is not None and len(df) > 0:
        latest = df.iloc[-1]
        portfolio_value = float(latest['portfolio_value'])
        weights = {}
        for t in TICKERS:
            val = float(latest.get(f'{t}_value', 0))
            weights[t] = val / portfolio_value if portfolio_value > 0 else 0.0
        return weights
    return {t: 0.0 for t in TICKERS}

def get_last_cash_pounds():
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    return float(df.iloc[-1]['cash_pounds']) if df is not None and len(df) > 0 else 0.0

def get_last_cash_percentage():
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    if df is not None and len(df) > 0:
        latest = df.iloc[-1]
        portfolio_value = float(latest['portfolio_value'])
        cash_pounds = float(latest['cash_pounds'])
        return cash_pounds / portfolio_value if portfolio_value > 0 else 0.0
    return 0.0

def get_last_realised():
    df = pd.read_csv(get_log_path(PORTFOLIO_LOG_FILE)) if os.path.exists(get_log_path(PORTFOLIO_LOG_FILE)) else None
    return float(df.iloc[-1]['realised_profit']) if df is not None and len(df) > 0 else 0.0

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
    """
    Log portfolio state with FULL precision for accurate tracking.
    Also creates a rounded version for readability.
    """
    # Full precision entry
    entry_full = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': value,
        'cash_pounds': cash_pounds,
        'realised_profit': realised,
        'total_wealth': total_wealth,
    }
    for t in TICKERS:
        entry_full[f'{t}_value'] = value * weights.get(t, 0.0)
    
    # Log full precision
    log_entry(PORTFOLIO_LOG_FILE, entry_full, ['portfolio_value', 'cash_pounds', 'realised_profit'])
    
    # Also log rounded version for readability
    entry_rounded = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': round(value, 2),
        'cash_pounds': round(cash_pounds, 2),
        'realised_profit': round(realised, 2),
        'total_wealth': round(total_wealth, 2),
    }
    for t in TICKERS:
        entry_rounded[f'{t}_value'] = round(value * weights.get(t, 0.0), 2)
    
    # Log rounded version (dedup separately)
    log_entry('portfolio_log_rounded.csv', entry_rounded, ['portfolio_value', 'cash_pounds', 'realised_profit'])

def log_rebalance_event(date, target, cash_pounds, value, cost, reason):
    """
    Log rebalance with FULL precision.
    Also creates a rounded version for readability.
    """
    # Full precision entry
    entry_full = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'cash_pounds': cash_pounds,
        'portfolio_value': value,
        'total_cost': cost,
        'rebalance_reason': reason,
    }
    for t in TICKERS:
        entry_full[f'{t}_target_value'] = value * target.get(t, 0.0)
    
    # Log full precision
    log_entry(REBALANCE_LOG_FILE, entry_full, ['cash_pounds', 'portfolio_value', 'rebalance_reason'])
    
    # Also log rounded version for readability
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
    try:
        spy = yf.download("SPY", period="1d", interval="1m", progress=False)
        if len(spy) == 0:
            spy = yf.download("SPY", period="5d", progress=False)["Close"]
        if len(spy) == 0:
            return initial
        current = float(spy["Close"].iloc[-1] if "Close" in spy else spy.iloc[-1])
        start = yf.download("SPY", start="2020-01-02", end="2020-01-03", progress=False)["Close"]
        if len(start) == 0:
            start = yf.download("SPY", start="2020-01-06", end="2020-01-07", progress=False)["Close"]
        return initial if len(start) == 0 else (current / float(start.iloc[0]) * initial)
    except:
        return initial

# ============================================================================
# MULTI-EXCHANGE PRICE FUNCTIONS
# ============================================================================

def get_latest_prices():
    """
    Get latest CLOSING prices only (no intraday) for consistency.
    """
    prices = {}
    for t in TICKERS:
        try:
            # Use daily data with a buffer
            data = yf.download(t, period="10d", progress=False)
            if len(data) > 0:
                if "Close" in data.columns:
                    closes = data["Close"]
                    if isinstance(closes, pd.DataFrame):
                        closes = closes.iloc[:, 0]
                    closes = closes[closes > 0]
                    if len(closes) > 0:
                        prices[t] = float(closes.iloc[-1])
                        continue
                else:
                    col = data.iloc[:, 0]
                    if isinstance(col, pd.DataFrame):
                        col = col.iloc[:, 0]
                    col = col[col > 0]
                    if len(col) > 0:
                        prices[t] = float(col.iloc[-1])
                        continue
            prices[t] = 0.0
        except:
            prices[t] = 0.0
    return prices

def fetch_price_data_with_forward_fill(tickers, lookback_days):
    """Fetch price data with forward-fill for holidays. Ensures all tickers have data on EVERY business day."""
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
    
    # Find common index
    common_index = None
    for series in valid_series.values():
        if common_index is None:
            common_index = series.index
        else:
            common_index = common_index.intersection(series.index)
    if common_index is None or len(common_index) == 0:
        common_index = next(iter(valid_series.values())).index
    
    # Create aligned DataFrame with forward-fill
    df = pd.DataFrame()
    for ticker, series in valid_series.items():
        aligned = series.reindex(common_index).ffill().bfill().fillna(0)
        df[ticker] = aligned
    
    df = df.ffill().bfill()
    
    # Reindex to business days
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

def update_portfolio_value(prices, weights, cash_pounds, cash_percentage, value, daily_rate, last_date):
    """
    Update portfolio value with cash treated as a real asset.
    Cash only changes via interest accrual (not by recalculating percentage).
    Uses CLOSING prices only for consistency.
    Returns (new_value, new_cash_pounds, new_cash_percentage, update_date).
    """
    today = datetime.now().date()
    
    # Skip if already updated today or no data
    if last_date == today or prices is None or len(prices) < 1:
        return value, cash_pounds, cash_percentage, today
    
    # Get current closing prices
    current = get_latest_prices()
    
    # Get previous closing prices from price history
    prev = {}
    for t in TICKERS:
        if t in prices.columns and len(prices) > 0:
            series = prices[t]
            non_zero = series[series > 0]
            if len(non_zero) > 0:
                prev[t] = float(non_zero.iloc[-1])
            else:
                prev[t] = float(series.iloc[-1]) if len(series) > 0 else 0.0
        else:
            prev[t] = current.get(t, 0.0)
    
    # Calculate equity return (only for assets with valid prices)
    equity_return = 0.0
    print("\n📊 Price Check:")
    for t in TICKERS:
        w = weights.get(t, 0.0)
        if w > 0.001:
            curr = current.get(t, 0.0)
            prev_price = prev.get(t, curr)
            if prev_price > 0 and curr > 0:
                asset_return = (curr / prev_price - 1)
                equity_return += w * asset_return
                print(f"   {t}: weight={w*100:.1f}%, prev=£{prev_price:.2f}, curr=£{curr:.2f}, change={asset_return*100:+.2f}%")
            else:
                print(f"   {t}: weight={w*100:.1f}%, prev=£{prev_price:.2f}, curr=£{curr:.2f} ⚠️")
    
    # Calculate equity value and new equity value
    equity_value = value * (1 - cash_percentage)
    new_equity_value = equity_value * (1 + equity_return)
    
    # Calculate cash interest on existing cash
    cash_interest = cash_pounds * daily_rate
    new_cash_pounds = cash_pounds + cash_interest
    
    # New total value
    new_value = new_equity_value + new_cash_pounds
    
    # Cash percentage changes only due to relative performance
    new_cash_percentage = new_cash_pounds / new_value if new_value > 0 else cash_percentage
    
    print(f"\n   Equity Value: £{equity_value:.2f} → £{new_equity_value:.2f} ({equity_return*100:+.2f}%)")
    print(f"   Cash: £{cash_pounds:.2f} → £{new_cash_pounds:.2f} (interest: £{cash_interest:.4f})")
    print(f"   New Total: £{new_value:.2f}")
    print(f"   Cash %: {new_cash_percentage*100:.1f}%")
    
    # Prevent extreme moves (sanity check)
    if new_value < value * 0.5 or new_value > value * 2:
        print(f"   ⚠️ WARNING: Extreme value change detected! (£{value:.2f} → £{new_value:.2f})")
        print(f"   ⚠️ This is likely due to data issues. Keeping previous value.")
        return value, cash_pounds, cash_percentage, today
    
    return new_value, new_cash_pounds, new_cash_percentage, today

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
    if isinstance(date, datetime):
        date = date.date()
    if isinstance(last_date, datetime):
        last_date = last_date.date()
    
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
    """
    Generate buy/sell orders to rebalance portfolio.
    Cash is treated as an asset - if cash needs to change, generate a cash order.
    """
    orders = {}
    total_trade = 0.0
    
    # Calculate current and target cash
    current_cash = cash_pounds
    target_cash = target_cash_pounds
    cash_diff = target_cash - current_cash
    
    # If cash needs to change, add a cash order
    if abs(cash_diff) > 0.01:
        orders['CASH'] = {
            'action': 'BUY' if cash_diff > 0 else 'SELL',
            'amount': abs(cash_diff),
            'ticker': 'CASH'
        }
        total_trade += abs(cash_diff)
    
    # Generate asset orders
    for t in TICKERS:
        current_val = value * current_weights.get(t, 0.0)
        target_val = value * target_weights.get(t, 0.0)
        diff = target_val - current_val
        
        if abs(diff) > 0.01:
            orders[t] = {
                'action': 'BUY' if diff > 0 else 'SELL',
                'amount': abs(diff),
                'ticker': t
            }
            total_trade += abs(diff)
    
    return orders, total_trade * ROUND_TRIP_COST_PCT

# === MAIN ===

def main():
    now = datetime.now()
    today = now.date()
    
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
    print("\nSTEP 1: Fetching Data (Multi-Exchange with Holiday Handling)")
    prices = fetch_price_data_with_forward_fill(TICKERS, LOOKBACK_DAYS)
    
    if prices is None or len(prices) == 0 or len(prices.columns) == 0:
        print("❌ No price data available. Using existing portfolio values.")
        value = get_last_log_value()
        if value is None:
            value = INITIAL_CAPITAL
        weights = get_last_weights()
        cash_pounds = get_last_cash_pounds()
        realised = get_last_realised()
        total_wealth = value + realised
        log_portfolio(now, value, weights, cash_pounds, realised, total_wealth)
        print(f"\n⚠️ No price data - logged existing values")
        return
    
    returns = calculate_returns(prices)
    exp_ret, cov, _ = calculate_annualised_stats(returns)
    spy_value = get_spy_value(INITIAL_CAPITAL)
    print(f"   SPY: £{spy_value:.2f}")
    
    # === STEP 2: UPDATE VALUE ===
    print("\nSTEP 2: Updating Portfolio Value")
    daily_rate = daily_cash_rate(CASH_INTEREST_RATE)
    
    value = get_last_log_value()
    realised = get_last_realised()
    is_first = value is None
    
    if is_first:
        value = INITIAL_CAPITAL
        cash_pounds = 0.0
        cash_percentage = 0.0
        weights = {t: 0.0 for t in TICKERS}
        print(f"   First run - starting at £{value:.2f}")
    else:
        weights = get_last_weights()
        cash_pounds = get_last_cash_pounds()
        cash_percentage = get_last_cash_percentage()
        print(f"   Last: £{value:.2f}, Cash: £{cash_pounds:.2f} ({cash_percentage*100:.1f}%), Realised: £{realised:.2f}")
        old = value
        value, cash_pounds, cash_percentage, new_date = update_portfolio_value(
            prices, weights, cash_pounds, cash_percentage, value, daily_rate, last_update
        )
        print(f"   Updated: £{value:.2f} (£{value - old:+.2f})")
        if new_date:
            save_state(LAST_UPDATE_DATE_FILE, new_date)
    
    if last_update is None:
        save_state(LAST_UPDATE_DATE_FILE, today)
    
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
    
    # Kelly calculation with sign change detection
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
    
    # Sign change detection
    prev_f = load_float('kelly_state.txt')
    if prev_f is not None:
        if (prev_f > 0) != (f_star > 0):
            force_rebalance = True
            print(f"🔄 Kelly sign change: {prev_f:.4f} → {f_star:.4f}")
            log_kelly_sign_change(now, prev_f, f_star)
    save_state('kelly_state.txt', f_star)
    
    # Calculate target cash allocation
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
    
    # Adjust target weights for cash
    adjusted_target = {t: w * (1 - target_cash_percentage) for t, w in target_weights.items()}
    
    # === STEP 5: REBALANCE CHECK ===
    print("\nSTEP 5: Rebalance Check")
    last_rebalance = load_date(LAST_REBALANCE_FILE)
    now_date = now.date()
    rebalance_needed, reason, drift, asset = should_rebalance(now_date, last_rebalance, weights, adjusted_target)
    
    if force_rebalance and not rebalance_needed:
        rebalance_needed, reason = True, "Kelly sign change"
    elif force_rebalance and "Take-profit" in reason:
        reason = "Take-profit reset"
    
    print(f"   Last: {last_rebalance.strftime('%Y-%m-%d')}")
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
        
        # Update state after rebalance
        cash_pounds = target_cash_pounds
        cash_percentage = target_cash_percentage
        weights = adjusted_target.copy()
    else:
        days_until = REBALANCE_MAX_DAYS - (now_date - last_rebalance).days
        print(f"\n✅ No rebalance. Next in {days_until} days")
    
    # === STEP 7: LOG ===
    print("\nSTEP 7: Logging")
    total_wealth = value + realised
    log_portfolio(now, value, weights, cash_pounds, realised, total_wealth)
    
    # === SUMMARY (rounded for display) ===
    print("\n" + "=" * 60)
    print("FINAL ENGINE COMPLETE")
    print(f"Active: £{value:.2f}, Cash: £{cash_pounds:.2f} ({cash_percentage*100:.1f}%)")
    print(f"Realised: £{realised:.2f}, Total: £{total_wealth:.2f}")
    print(f"Return: {(total_wealth / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()