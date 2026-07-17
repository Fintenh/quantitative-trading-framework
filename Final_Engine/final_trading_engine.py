"""
final_trading_engine.py - Main execution script for Final Engine.
Uses Optimal_Ethical 11-asset portfolio with optimised parameters.
Combines drift-based and time-based rebalancing with GARCH volatility forecasting.
Includes take-profit logic to lock in gains relative to SPY.
FIXED: Portfolio value now updates correctly using latest available prices.
"""

import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Import from final config (all parameters pre-optimised)
from final_config import *

# Import from Phase 2 modules
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))

from data_fetcher import fetch_price_data, calculate_returns, calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios, generate_portfolio_summary
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility, determine_cash_allocation

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def ensure_log_dir():
    """Create logs directory if it doesn't exist."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def load_last_rebalance():
    """Load last rebalance date from file."""
    ensure_log_dir()
    filepath = os.path.join(LOG_DIR, LAST_REBALANCE_FILE)
    try:
        with open(filepath, 'r') as f:
            return datetime.strptime(f.read().strip(), "%Y-%m-%d")
    except:
        return datetime(2020, 1, 1)


def save_last_rebalance(date):
    """Save rebalance date to file."""
    ensure_log_dir()
    with open(os.path.join(LOG_DIR, LAST_REBALANCE_FILE), 'w') as f:
        f.write(date.strftime("%Y-%m-%d"))


def load_last_update_date():
    """Load the last date the portfolio was updated."""
    ensure_log_dir()
    filepath = os.path.join(LOG_DIR, LAST_UPDATE_DATE_FILE)
    try:
        with open(filepath, 'r') as f:
            return datetime.strptime(f.read().strip(), "%Y-%m-%d").date()
    except:
        return None


def save_last_update_date(date):
    """Save the date the portfolio was updated."""
    ensure_log_dir()
    with open(os.path.join(LOG_DIR, LAST_UPDATE_DATE_FILE), 'w') as f:
        if date is not None:
            f.write(date.strftime("%Y-%m-%d"))
        else:
            f.write(datetime.now().strftime("%Y-%m-%d"))


def get_last_portfolio_value():
    """Get the last logged portfolio value from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            return float(df.iloc[-1]['portfolio_value'])
    return None


def get_last_cash_allocation():
    """Get the last logged cash allocation from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            return float(df.iloc[-1]['cash_allocation'])
    return 0.0


def get_last_realised_profit():
    """Get the last logged realised profit from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0 and 'realised_profit' in df.columns:
            return float(df.iloc[-1]['realised_profit'])
    return 0.0


def get_last_weights():
    """Get the last logged portfolio weights from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            latest = df.iloc[-1]
            weights = {}
            for ticker in TICKERS:
                col = f'{ticker}_weight'
                if col in df.columns:
                    weights[ticker] = float(latest[col])
            return weights
    return {ticker: 0.0 for ticker in TICKERS}


def log_portfolio_state(date, value, weights, cash, realised_profit=0.0, total_wealth=None):
    """Log current portfolio state only if something changed."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    
    if total_wealth is None:
        total_wealth = value + realised_profit
    
    if isinstance(weights, list):
        weights_dict = {ticker: float(weights[i]) if i < len(weights) else 0.0 for i, ticker in enumerate(TICKERS)}
    else:
        weights_dict = {k: float(v) for k, v in weights.items()}
    
    entry = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'portfolio_value': float(value),
        'cash_allocation': float(cash),
        'realised_profit': float(realised_profit),
        'total_wealth': float(total_wealth),
        **{f'{ticker}_weight': float(weights_dict.get(ticker, 0.0)) for ticker in TICKERS}
    }
    
    df = pd.DataFrame([entry])
    
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        
        if len(existing) > 0:
            last = existing.iloc[-1]
            same = True
            
            for col in ['portfolio_value', 'cash_allocation', 'realised_profit', 'total_wealth']:
                if col in last:
                    try:
                        if abs(float(last[col]) - float(entry[col])) > 0.0001:
                            same = False
                            break
                    except:
                        same = False
                        break
            
            if same:
                for ticker in TICKERS:
                    col = f'{ticker}_weight'
                    if col in last:
                        try:
                            if abs(float(last[col]) - float(entry[col])) > 0.0001:
                                same = False
                                break
                        except:
                            same = False
                            break
            
            if same:
                return
        
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(log_file, index=False)


def log_rebalance(date, target_weights, cash, value, cost=0.0, reason=""):
    """Log rebalance event."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, REBALANCE_LOG_FILE)
    
    target_clean = {k: float(v) for k, v in target_weights.items()}
    
    entry = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'cash_allocation': float(cash),
        'portfolio_value': float(value),
        'total_cost': float(cost),
        'rebalance_reason': reason,
        **{ticker: float(target_clean.get(ticker, 0.0)) for ticker in TICKERS}
    }
    
    df = pd.DataFrame([entry])
    
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        
        if len(existing) > 0:
            last = existing.iloc[-1]
            same = True
            
            for col in ['cash_allocation', 'portfolio_value', 'total_cost', 'rebalance_reason']:
                if col in last:
                    if col == 'rebalance_reason':
                        if str(last[col]) != str(entry[col]):
                            same = False
                            break
                    else:
                        try:
                            if abs(float(last[col]) - float(entry[col])) > 0.0001:
                                same = False
                                break
                        except:
                            same = False
                            break
            
            if same:
                for ticker in TICKERS:
                    if ticker in last:
                        try:
                            if abs(float(last[ticker]) - float(entry[ticker])) > 0.0001:
                                same = False
                                break
                        except:
                            same = False
                            break
            
            if same:
                return
        
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(log_file, index=False)


def log_rebalance_decision(date, needed, reason, current, target):
    """Log rebalance decision for audit."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, REBALANCE_DECISIONS_FILE)
    
    current_clean = {k: float(v) for k, v in current.items()}
    target_clean = {k: float(v) for k, v in target.items()}
    
    entry = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'rebalance_needed': needed,
        'reason': reason,
        'current_weights': str(current_clean),
        'target_weights': str(target_clean)
    }
    
    df = pd.DataFrame([entry])
    
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        
        if len(existing) > 0:
            last = existing.iloc[-1]
            if (last['rebalance_needed'] == entry['rebalance_needed'] and
                last['reason'] == entry['reason']):
                return
        
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(log_file, index=False)


def log_take_profit(date, profit, new_active, total_realised):
    """Log a take-profit event."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, TAKE_PROFIT_LOG_FILE)
    
    entry = {
        'date': date.strftime("%Y-%m-%d %H:%M:%S"),
        'profit_withdrawn': float(profit),
        'new_active_value': float(new_active),
        'total_realised_profit': float(total_realised)
    }
    
    df = pd.DataFrame([entry])
    
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(log_file, index=False)
    
    print(f"✅ Take-profit logged: {log_file}")

# ============================================================================
# PORTFOLIO VALUE UPDATE
# ============================================================================

def daily_cash_rate(annual_rate):
    """Convert annual interest rate to daily compounded rate."""
    return (1 + annual_rate) ** (1 / 252) - 1 if annual_rate > 0 else 0.0


def get_latest_prices():
    """
    Get the most recent prices for all TICKERS.
    Tries live prices first, falls back to daily close if not available.
    """
    latest_prices = {}
    
    for ticker in TICKERS:
        try:
            # Try to get today's intraday data (1-minute intervals)
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            
            if len(data) > 0:
                latest_prices[ticker] = float(data["Close"].iloc[-1])
            else:
                # Fallback: get the most recent daily close
                data = yf.download(ticker, period="5d", progress=False)["Close"]
                if len(data) > 0:
                    latest_prices[ticker] = float(data.iloc[-1])
                else:
                    latest_prices[ticker] = 0.0
        except Exception as e:
            try:
                data = yf.download(ticker, period="5d", progress=False)["Close"]
                if len(data) > 0:
                    latest_prices[ticker] = float(data.iloc[-1])
                else:
                    latest_prices[ticker] = 0.0
            except:
                latest_prices[ticker] = 0.0
    
    return latest_prices


def update_portfolio_value(prices, weights, cash_pct, value, daily_rate, last_update_date):
    """
    Update portfolio value with market returns and cash interest.
    Uses the most recent available prices.
    """
    today = datetime.now().date()
    
    # If we've already updated today, don't update again
    if last_update_date == today:
        return value, last_update_date
    
    # Check if we have price data
    if len(prices) < 1:
        return value, today
    
    # Get latest prices (live if available, else daily close)
    current_prices = get_latest_prices()
    
    # Get the previous close prices from our historical data
    prev_prices = {ticker: float(prices[ticker].iloc[-1]) for ticker in TICKERS}
    
    # Calculate equity return since last update
    equity_return = 0.0
    valid_assets = 0
    
    for ticker in TICKERS:
        weight = weights.get(ticker, 0.0)
        if weight > 0:
            current = current_prices.get(ticker, 0.0)
            previous = prev_prices.get(ticker, 0.0)
            
            if current > 0 and previous > 0:
                equity_return += weight * (current / previous - 1)
                valid_assets += 1
    
    # If no valid assets, try using the most recent price from the DataFrame
    if valid_assets == 0:
        # Use the last available close as the "current" price
        for ticker in TICKERS:
            weight = weights.get(ticker, 0.0)
            if weight > 0:
                current = float(prices[ticker].iloc[-1])
                previous = float(prices[ticker].iloc[-2]) if len(prices) >= 2 else current
                if current > 0 and previous > 0:
                    equity_return += weight * (current / previous - 1)
                    valid_assets += 1
        
        if valid_assets == 0:
            return value, today
    
    total_return = equity_return * (1 - cash_pct) + daily_rate * cash_pct
    new_value = value * (1 + total_return)
    
    # Save today as the update date
    save_last_update_date(today)
    
    return new_value, today


def get_spy_value(initial=100.0):
    """Get current SPY value normalised to initial capital."""
    try:
        # Try to get live price first
        spy_data = yf.download("SPY", period="1d", interval="1m", progress=False)
        if len(spy_data) > 0:
            current_price = float(spy_data["Close"].iloc[-1])
        else:
            spy_data = yf.download("SPY", period="5d", progress=False)["Close"]
            if len(spy_data) > 0:
                current_price = float(spy_data.iloc[-1])
            else:
                return initial
        
        # Use Jan 2, 2020 (definitely a trading day)
        spy_start_data = yf.download("SPY", start="2020-01-02", end="2020-01-03", progress=False)["Close"]
        if len(spy_start_data) == 0:
            spy_start_data = yf.download("SPY", start="2020-01-06", end="2020-01-07", progress=False)["Close"]
        if len(spy_start_data) == 0:
            return initial
        
        return float(current_price) / float(spy_start_data.iloc[0]) * initial
    except Exception as e:
        return initial

# ============================================================================
# REBALANCE LOGIC
# ============================================================================

def calculate_drift(current, target):
    """Calculate drift for each asset and return max."""
    max_drift = 0.0
    max_asset = None
    
    for ticker in current:
        drift = abs(current.get(ticker, 0.0) - target.get(ticker, 0.0))
        if drift > max_drift:
            max_drift = drift
            max_asset = ticker
    
    return max_drift, max_asset


def should_rebalance(current_date, last_date, current_weights, target_weights):
    """
    Determine if rebalance is needed based on:
    1. First run (no money invested yet)
    2. Min days (prevent overtrading)
    3. Max days (quarterly max)
    4. Drift threshold
    """
    days_since = (current_date - last_date).days
    
    max_drift, drifting_asset = calculate_drift(current_weights, target_weights)
    
    # Check if this is the first run (all weights = 0 or near 0)
    total_current_weight = sum(current_weights.values())
    is_first_run = total_current_weight < 0.01
    
    if is_first_run:
        return True, "First run - initial portfolio setup", max_drift, drifting_asset
    
    # Too soon? Wait.
    if days_since < REBALANCE_MIN_DAYS:
        return False, f"Min days not met ({days_since}/{REBALANCE_MIN_DAYS})", max_drift, drifting_asset
    
    # Max time reached? Rebalance.
    if days_since >= REBALANCE_MAX_DAYS:
        return True, f"Time-based: {days_since} days >= {REBALANCE_MAX_DAYS} days", max_drift, drifting_asset
    
    # Check drift
    if max_drift > DRIFT_THRESHOLD:
        return True, f"Drift-based: {drifting_asset} drifted {max_drift*100:.2f}% > {DRIFT_THRESHOLD*100:.1f}%", max_drift, drifting_asset
    
    return False, f"Waiting... (days={days_since}, max_drift={max_drift*100:.2f}%)", max_drift, drifting_asset


def generate_orders(value, current, target):
    """Generate buy/sell orders to rebalance portfolio."""
    orders = {}
    total_trade = 0.0
    
    for ticker in TICKERS:
        current_val = value * current.get(ticker, 0.0)
        target_val = value * target.get(ticker, 0.0)
        diff = target_val - current_val
        
        if abs(diff) > 0.01:
            orders[ticker] = {
                'action': 'BUY' if diff > 0 else 'SELL',
                'amount': abs(diff),
                'ticker': ticker
            }
            total_trade += abs(diff)
    
    total_cost = total_trade * ROUND_TRIP_COST_PCT
    return orders, total_cost

# ============================================================================
# MAIN
# ============================================================================

def main():
    current_time = datetime.now()
    today = current_time.date()
    
    # Check if we've already updated today
    last_update_date = load_last_update_date()
    if last_update_date == today:
        print("=" * 60)
        print("FINAL TRADING ENGINE: OPTIMAL ETHICAL 11")
        print("=" * 60)
        print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n⚠️ Already updated today ({today}). Skipping update.")
        print("   To force an update, delete logs/last_update_date.txt")
        print("=" * 60)
        return
    
    print("=" * 60)
    print("FINAL TRADING ENGINE: OPTIMAL ETHICAL 11")
    print("=" * 60)
    print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {len(TICKERS)}")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Rebalance: {REBALANCE_MIN_DAYS}-{REBALANCE_MAX_DAYS} days")
    print(f"Drift Threshold: {DRIFT_THRESHOLD*100:.1f}%")
    print(f"Take-Profit: {RELATIVE_TAKE_PROFIT_PCT*100:.0f}% vs SPY")
    print(f"Kelly Lookback: {KELLY_LOOKBACK} days")
    print("=" * 60)
    
    # ------------------------------------------------------------------------
    # STEP 1: FETCH DATA
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 1: Fetching Data")
    print("-" * 40)
    
    prices = fetch_price_data(TICKERS, LOOKBACK_DAYS)
    returns = calculate_returns(prices)
    exp_ret, cov, _ = calculate_annualised_stats(returns)
    spy_value = get_spy_value(INITIAL_CAPITAL)
    print(f"SPY Value: £{spy_value:.2f}")
    
    # ------------------------------------------------------------------------
    # STEP 2: UPDATE PORTFOLIO VALUE
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 2: Updating Portfolio Value")
    print("-" * 40)
    
    daily_rate = daily_cash_rate(CASH_INTEREST_RATE)
    print(f"Daily Cash Rate: {daily_rate*100:.4f}%")
    
    portfolio_value = get_last_portfolio_value()
    realised_profit = get_last_realised_profit()
    last_update_date = load_last_update_date()
    is_first_run = portfolio_value is None
    
    if is_first_run:
        portfolio_value = INITIAL_CAPITAL
        realised_profit = 0.0
        cash_pct = 0.0
        current_weights = {ticker: 0.0 for ticker in TICKERS}
        print(f"First run - starting at £{portfolio_value:.2f}")
    else:
        current_weights = get_last_weights()
        cash_pct = get_last_cash_allocation()
        print(f"Last active: £{portfolio_value:.2f}, Realised: £{realised_profit:.2f}")
        print(f"Last update date: {last_update_date}")
        
        old_value = portfolio_value
        portfolio_value, new_update_date = update_portfolio_value(
            prices, current_weights, cash_pct, portfolio_value, daily_rate, last_update_date
        )
        print(f"Updated active: £{portfolio_value:.2f} (Change: £{portfolio_value - old_value:+.2f})")
        print(f"New update date: {new_update_date}")
        
        if new_update_date is not None:
            save_last_update_date(new_update_date)
    
    # If first run or no update, make sure we have a last update date
    if last_update_date is None:
        save_last_update_date(today)
    
    # ------------------------------------------------------------------------
    # STEP 2.5: TAKE-PROFIT CHECK
    # ------------------------------------------------------------------------
    force_rebalance = False
    
    if RELATIVE_TAKE_PROFIT_PCT > 0:
        target = spy_value * (1 + RELATIVE_TAKE_PROFIT_PCT)
        
        print(f"\n📊 TAKE-PROFIT CHECK:")
        print(f"   Active: £{portfolio_value:.2f}")
        print(f"   SPY: £{spy_value:.2f}")
        print(f"   Target (SPY × {1 + RELATIVE_TAKE_PROFIT_PCT:.2f}): £{target:.2f}")
        print(f"   Gap: £{portfolio_value - target:+.2f}")
        
        if portfolio_value > target:
            print("\n" + "=" * 40)
            print("💰 TAKE-PROFIT TRIGGERED!")
            print("=" * 40)
            print(f"   Portfolio Value: £{portfolio_value:.2f}")
            print(f"   SPY Value: £{spy_value:.2f}")
            print(f"   Target: £{target:.2f}")
            
            profit = portfolio_value - spy_value
            print(f"   Profit to withdraw: £{profit:.2f}")
            
            realised_profit += profit
            portfolio_value = spy_value
            
            print(f"   New Active Value: £{portfolio_value:.2f}")
            print(f"   Total Realised Profit: £{realised_profit:.2f}")
            print("=" * 40)
            
            log_take_profit(current_time, profit, portfolio_value, realised_profit)
            force_rebalance = True
    
    # ------------------------------------------------------------------------
    # STEP 3: OPTIMISE PORTFOLIO
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 3: Optimising Portfolio")
    print("-" * 40)
    
    opt_results = optimise_portfolios(exp_ret, cov, RISK_FREE_RATE)
    print(generate_portfolio_summary(opt_results))
    target_weights = {ticker: opt_results['msr_weights'][i] for i, ticker in enumerate(TICKERS)}
    
    # ------------------------------------------------------------------------
    # STEP 4: GARCH VOLATILITY FORECAST
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 4: GARCH Volatility Forecast")
    print("-" * 40)
    
    models, _, _ = fit_garch_for_assets(returns)
    vols = get_latest_volatility(models, returns)
    avg_vol = get_average_volatility(vols)
    
    # ------------------------------------------------------------------------
    # STEP 4.5: KELLY CASH CAP
    # ------------------------------------------------------------------------
    kelly_returns = returns.iloc[-KELLY_LOOKBACK:] if len(returns) >= KELLY_LOOKBACK else returns
    mu = kelly_returns.mean().mean() * 252
    sigma = kelly_returns.std().mean() * np.sqrt(252)
    
    if sigma > 0 and np.isfinite(sigma):
        f_star = (mu - RISK_FREE_RATE) / (sigma ** 2)
    else:
        f_star = 0.0
    
    kelly_cash_cap = KELLY_BASE_CAP if f_star > 0 else KELLY_MAX_CAP
    
    if avg_vol <= CASH_MIN_VOLATILITY:
        cash_pct = 0.0
    elif avg_vol >= CASH_MAX_VOLATILITY:
        cash_pct = kelly_cash_cap
    else:
        fraction = (avg_vol - CASH_MIN_VOLATILITY) / (CASH_MAX_VOLATILITY - CASH_MIN_VOLATILITY)
        cash_pct = fraction * kelly_cash_cap
    
    print(f"\n📊 KELLY STATUS:")
    print(f"   f*: {f_star:.3f}")
    print(f"   Expected Return (annualised): {mu*100:.1f}%")
    print(f"   Volatility (annualised): {sigma*100:.1f}%")
    print(f"   Kelly Cash Cap: {kelly_cash_cap*100:.0f}%")
    print(f"Average Volatility: {avg_vol*100:.2f}%")
    print(f"Cash Allocation: {cash_pct*100:.1f}%")
    print(f"Equity Allocation: {(1-cash_pct)*100:.1f}%")
    
    adjusted_target = {t: w * (1 - cash_pct) for t, w in target_weights.items()}
    
    # ------------------------------------------------------------------------
    # STEP 5: CHECK REBALANCE CONDITIONS
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 5: Checking Rebalance Conditions")
    print("-" * 40)
    
    last_rebalance = load_last_rebalance()
    rebalance_needed, reason, max_drift, drifting = should_rebalance(
        current_time, last_rebalance, current_weights, adjusted_target
    )
    
    if force_rebalance:
        rebalance_needed = True
        reason = "Take-profit reset"
    
    print(f"Last rebalance: {last_rebalance.strftime('%Y-%m-%d')}")
    print(f"Rebalance needed: {rebalance_needed}")
    print(f"Reason: {reason}")
    
    if max_drift > 0.01:
        print(f"\nMax Drift: {max_drift*100:.2f}% ({drifting})")
        print(f"Threshold: {DRIFT_THRESHOLD*100:.1f}%")
    
    # ------------------------------------------------------------------------
    # STEP 6: EXECUTE REBALANCE
    # ------------------------------------------------------------------------
    if rebalance_needed:
        print("\n" + "-" * 40)
        print("STEP 6: Executing Rebalance")
        print("-" * 40)
        
        orders, total_cost = generate_orders(portfolio_value, current_weights, adjusted_target)
        
        print(f"Portfolio Value: £{portfolio_value:.2f}")
        print(f"Cash Allocation: {cash_pct*100:.1f}%")
        print(f"Reason: {reason}")
        
        if orders:
            print("\nOrders:")
            for ticker, order in orders.items():
                print(f"  {order['action']} {ticker}: £{order['amount']:.2f}")
        else:
            print("  No orders needed")
        
        print(f"\nEstimated Cost: £{total_cost:.4f} ({ROUND_TRIP_COST_PCT*100:.1f}% round-trip)")
        
        save_last_rebalance(current_time)
        log_rebalance(current_time, adjusted_target, cash_pct, portfolio_value, total_cost, reason)
        current_weights = adjusted_target.copy()
    else:
        print(f"\n✅ No rebalance needed")
        days_until = REBALANCE_MAX_DAYS - (current_time - last_rebalance).days
        print(f"Next forced rebalance in {days_until} days")
    
    # ------------------------------------------------------------------------
    # STEP 7: LOG PORTFOLIO
    # ------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 7: Logging Portfolio Status")
    print("-" * 40)
    
    total_wealth = portfolio_value + realised_profit
    log_portfolio_state(current_time, portfolio_value, current_weights, cash_pct, realised_profit, total_wealth)
    log_rebalance_decision(current_time, rebalance_needed, reason, current_weights, adjusted_target)
    
    print(f"✅ Logged to {LOG_DIR}/")
    
    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL ENGINE COMPLETE")
    print("=" * 60)
    print(f"Active Portfolio: £{portfolio_value:.2f}")
    print(f"Realised Profits: £{realised_profit:.2f}")
    print(f"Total Wealth: £{total_wealth:.2f}")
    print(f"Total Return: {(total_wealth / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()