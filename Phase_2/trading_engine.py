"""
trading_engine.py - Main execution script for the Phase 2 trading system.
Combines drift-based and time-based rebalancing with GARCH volatility forecasting.
Includes take-profit logic to lock in gains relative to SPY.
"""

import os
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

from config import *
from data_fetcher import fetch_price_data, calculate_returns, calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios, generate_portfolio_summary
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility, determine_cash_allocation

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

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
        f.write(date.strftime("%Y-%m-%d"))


def get_last_portfolio_value():
    """Get the last logged portfolio value from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            return df.iloc[-1]['portfolio_value']
    return None


def get_last_cash_allocation():
    """Get the last logged cash allocation from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0:
            return df.iloc[-1]['cash_allocation']
    return 0.0


def get_last_realised_profit():
    """Get the last logged realised profit from the log file."""
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        if len(df) > 0 and 'realised_profit' in df.columns:
            return df.iloc[-1]['realised_profit']
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
                    weights[ticker] = latest[col]
            return weights
    return {ticker: 0.0 for ticker in TICKERS}


def log_portfolio_state(date, value, weights, cash, realised_profit=0.0, total_wealth=None):
    """Log current portfolio state only if something changed."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, PORTFOLIO_LOG_FILE)

    if total_wealth is None:
        total_wealth = value + realised_profit

    # Handle weights if passed as list (from TICKERS order)
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

        # Check if last entry is identical (avoid duplicates)
        if len(existing) > 0:
            last = existing.iloc[-1]
            same = True

            # Compare numeric fields (ignore date for comparison)
            for col in ['portfolio_value', 'cash_allocation', 'realised_profit', 'total_wealth']:
                if col in last:
                    try:
                        if abs(float(last[col]) - float(entry[col])) > 0.0001:
                            same = False
                            break
                    except:
                        same = False
                        break

            # Compare weights if numeric fields matched
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
                # Identical to last entry - skip logging
                return

        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(log_file, index=False)


def log_rebalance(date, target_weights, cash, value, cost=0.0, reason=""):
    """Log rebalance event."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, REBALANCE_LOG_FILE)

    # Convert numpy floats to Python floats for a clean CSV
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

        # Check if last entry is identical (avoid duplicates)
        if len(existing) > 0:
            last = existing.iloc[-1]
            same = True

            # Compare key fields
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

            # Compare weights if other fields matched
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
                # Identical rebalance - skip logging
                return

        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(log_file, index=False)


def log_rebalance_decision(date, needed, reason, current, target):
    """Log rebalance decision for audit."""
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, REBALANCE_DECISIONS_FILE)

    # Convert numpy floats to Python floats for a clean CSV
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

        # Check if last entry is identical (avoid duplicates)
        if len(existing) > 0:
            last = existing.iloc[-1]
            # Only check the actual data, not the timestamp
            if (last['rebalance_needed'] == entry['rebalance_needed'] and
                last['reason'] == entry['reason']):
                # Same decision as before - skip logging
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

    print(f"Take-profit logged: {log_file}")

# ---------------------------------------------------------------------------
# Portfolio value update
# ---------------------------------------------------------------------------

def daily_cash_rate(annual_rate):
    """Convert annual interest rate to daily compounded rate."""
    return (1 + annual_rate) ** (1 / 252) - 1 if annual_rate > 0 else 0.0


def update_portfolio_value(prices, weights, cash_pct, value, daily_rate, last_update_date):
    """
    Update portfolio value with market returns and cash interest.
    Only applies returns for days that haven't been processed yet.
    """
    if len(prices) < 2:
        return value, last_update_date

    today = datetime.now().date()

    # If no last update, use the most recent day's return
    if last_update_date is None:
        current = prices.iloc[-1]
        previous = prices.iloc[-2]
        date_index = prices.index[-1].date()

        equity_return = 0.0
        for ticker in TICKERS:
            if weights.get(ticker, 0.0) > 0:
                equity_return += weights[ticker] * (current[ticker] / previous[ticker] - 1)

        total_return = equity_return * (1 - cash_pct) + daily_rate * cash_pct
        new_value = value * (1 + total_return)

        return new_value, date_index

    # Find all dates between last_update_date and today that exist in prices
    try:
        future_prices = prices[prices.index.date > last_update_date]

        if len(future_prices) == 0:
            # No new data, return same value
            return value, last_update_date

        # Process each new day sequentially
        new_value = value
        for i in range(1, len(future_prices)):
            current = future_prices.iloc[i]
            previous = future_prices.iloc[i-1]

            equity_return = 0.0
            for ticker in TICKERS:
                if weights.get(ticker, 0.0) > 0:
                    equity_return += weights[ticker] * (current[ticker] / previous[ticker] - 1)

            total_return = equity_return * (1 - cash_pct) + daily_rate * cash_pct
            new_value *= (1 + total_return)

        return new_value, future_prices.index[-1].date()

    except Exception as e:
        # Fallback: use the most recent day
        current = prices.iloc[-1]
        previous = prices.iloc[-2]

        equity_return = 0.0
        for ticker in TICKERS:
            if weights.get(ticker, 0.0) > 0:
                equity_return += weights[ticker] * (current[ticker] / previous[ticker] - 1)

        total_return = equity_return * (1 - cash_pct) + daily_rate * cash_pct
        new_value = value * (1 + total_return)

        return new_value, prices.index[-1].date()


def get_spy_value(initial=100.0, start="2020-01-01"):
    """Get current SPY value normalised to initial capital."""
    try:
        spy = yf.download("SPY", period="5d", progress=False)["Close"]
        if len(spy) == 0:
            spy = yf.download("^GSPC", period="5d", progress=False)["Close"]
        if len(spy) == 0:
            return initial

        spy_start = yf.download("SPY", start=start, end=start, progress=False)["Close"]
        if len(spy_start) == 0:
            spy_start = yf.download("^GSPC", start=start, end=start, progress=False)["Close"]
        if len(spy_start) == 0:
            return initial

        return float(spy.iloc[-1]) / float(spy_start.iloc[-1]) * initial
    except:
        return initial

# ---------------------------------------------------------------------------
# Rebalance logic
# ---------------------------------------------------------------------------

def calculate_drift(current, target):
    """Calculate drift for each asset and return the max."""
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
    Determine if a rebalance is needed based on:
    1. Min days (prevent overtrading)
    2. Max days (quarterly max)
    3. Drift threshold (2.5% rule)
    """
    days_since = (current_date - last_date).days

    max_drift, drifting_asset = calculate_drift(current_weights, target_weights)

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
    """Generate buy/sell orders to rebalance the portfolio."""
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    current_time = datetime.now()

    print("=" * 60)
    print("PHASE 2: TRADING 212 EXECUTION ENGINE")
    print("=" * 60)
    print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {len(TICKERS)}")
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Rebalance: {REBALANCE_MIN_DAYS}-{REBALANCE_MAX_DAYS} days")
    print(f"Drift Threshold: {DRIFT_THRESHOLD*100:.1f}%")
    print(f"Take-Profit: {RELATIVE_TAKE_PROFIT_PCT*100:.0f}% vs SPY")
    print(f"Kelly Lookback: {KELLY_LOOKBACK} days")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: fetch data
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 1: Fetching Data")
    print("-" * 40)

    prices = fetch_price_data(TICKERS, LOOKBACK_DAYS)
    returns = calculate_returns(prices)
    exp_ret, cov, _ = calculate_annualised_stats(returns)
    spy_value = get_spy_value(INITIAL_CAPITAL, START_DATE)
    print(f"SPY Value: £{spy_value:.2f}")

    # ------------------------------------------------------------------
    # Step 2: update portfolio value
    # ------------------------------------------------------------------
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

        # Save the update date
        save_last_update_date(new_update_date)

    # ------------------------------------------------------------------
    # Step 2.5: take-profit check
    # ------------------------------------------------------------------
    force_rebalance = False

    if RELATIVE_TAKE_PROFIT_PCT > 0:
        target = spy_value * (1 + RELATIVE_TAKE_PROFIT_PCT)

        # Print take-profit status
        print(f"\nTAKE-PROFIT CHECK:")
        print(f"   Active: £{portfolio_value:.2f}")
        print(f"   SPY: £{spy_value:.2f}")
        print(f"   Target (SPY x {1 + RELATIVE_TAKE_PROFIT_PCT:.2f}): £{target:.2f}")
        print(f"   Gap: £{portfolio_value - target:+.2f}")

        if portfolio_value > target:
            print("\n" + "=" * 40)
            print("TAKE-PROFIT TRIGGERED")
            print("=" * 40)
            print(f"   Portfolio Value: £{portfolio_value:.2f}")
            print(f"   SPY Value: £{spy_value:.2f}")
            print(f"   Target: £{target:.2f}")

            # Calculate profit to withdraw
            profit = portfolio_value - spy_value
            print(f"   Profit to withdraw: £{profit:.2f}")

            # Update realised profit and reset active portfolio
            realised_profit += profit
            portfolio_value = spy_value

            print(f"   New Active Value: £{portfolio_value:.2f}")
            print(f"   Total Realised Profit: £{realised_profit:.2f}")
            print("=" * 40)

            # Log the take-profit event
            log_take_profit(current_time, profit, portfolio_value, realised_profit)

            # Force rebalance after take-profit
            force_rebalance = True

    # ------------------------------------------------------------------
    # Step 3: optimise portfolio
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 3: Optimising Portfolio")
    print("-" * 40)

    opt_results = optimise_portfolios(exp_ret, cov, RISK_FREE_RATE)
    print(generate_portfolio_summary(opt_results))
    target_weights = {ticker: opt_results['msr_weights'][i] for i, ticker in enumerate(TICKERS)}

    # ------------------------------------------------------------------
    # Step 4: GARCH volatility forecast
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 4: GARCH Volatility Forecast")
    print("-" * 40)

    models, _, _ = fit_garch_for_assets(returns)
    vols = get_latest_volatility(models, returns)
    avg_vol = get_average_volatility(vols)

    # ------------------------------------------------------------------
    # Step 4.5: Kelly cash cap
    # ------------------------------------------------------------------
    # Calculate the Kelly cash cap using the last KELLY_LOOKBACK days of returns
    kelly_returns = returns.iloc[-KELLY_LOOKBACK:] if len(returns) >= KELLY_LOOKBACK else returns
    mu = kelly_returns.mean().mean() * 252
    sigma = kelly_returns.std().mean() * np.sqrt(252)

    if sigma > 0 and np.isfinite(sigma):
        f_star = (mu - RISK_FREE_RATE) / (sigma ** 2)
    else:
        f_star = 0.0

    # Kelly cash cap: 20% if f* > 0, 100% if f* <= 0
    kelly_cash_cap = KELLY_BASE_CAP if f_star > 0 else KELLY_MAX_CAP

    # GARCH determines cash within the Kelly cap
    if avg_vol <= CASH_MIN_VOLATILITY:
        cash_pct = 0.0
    elif avg_vol >= CASH_MAX_VOLATILITY:
        cash_pct = kelly_cash_cap
    else:
        fraction = (avg_vol - CASH_MIN_VOLATILITY) / (CASH_MAX_VOLATILITY - CASH_MIN_VOLATILITY)
        cash_pct = fraction * kelly_cash_cap

    print(f"\nKELLY STATUS:")
    print(f"   f*: {f_star:.3f}")
    print(f"   Expected Return (annualised): {mu*100:.1f}%")
    print(f"   Volatility (annualised): {sigma*100:.1f}%")
    print(f"   Kelly Cash Cap: {kelly_cash_cap*100:.0f}%")
    print(f"Average Volatility: {avg_vol*100:.2f}%")
    print(f"Cash Allocation: {cash_pct*100:.1f}%")
    print(f"Equity Allocation: {(1-cash_pct)*100:.1f}%")

    # Adjust target weights with cash
    adjusted_target = {t: w * (1 - cash_pct) for t, w in target_weights.items()}

    # ------------------------------------------------------------------
    # Step 5: check rebalance conditions
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 5: Checking Rebalance Conditions")
    print("-" * 40)

    last_rebalance = load_last_rebalance()
    rebalance_needed, reason, max_drift, drifting = should_rebalance(
        current_time, last_rebalance, current_weights, adjusted_target
    )

    # Override if force_rebalance is True (from take-profit)
    if force_rebalance:
        rebalance_needed = True
        reason = "Take-profit reset"

    print(f"Last rebalance: {last_rebalance.strftime('%Y-%m-%d')}")
    print(f"Rebalance needed: {rebalance_needed}")
    print(f"Reason: {reason}")

    # Show drift report
    if max_drift > 0.01:
        print(f"\nMax Drift: {max_drift*100:.2f}% ({drifting})")
        print(f"Threshold: {DRIFT_THRESHOLD*100:.1f}%")

    # ------------------------------------------------------------------
    # Step 6: execute rebalance
    # ------------------------------------------------------------------
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
        print(f"\nNo rebalance needed")
        days_until = REBALANCE_MAX_DAYS - (current_time - last_rebalance).days
        print(f"Next forced rebalance in {days_until} days")

    # ------------------------------------------------------------------
    # Step 7: log portfolio
    # ------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("STEP 7: Logging Portfolio Status")
    print("-" * 40)

    total_wealth = portfolio_value + realised_profit
    log_portfolio_state(current_time, portfolio_value, current_weights, cash_pct, realised_profit, total_wealth)
    log_rebalance_decision(current_time, rebalance_needed, reason, current_weights, adjusted_target)

    print(f"Logged to {LOG_DIR}/")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"Active Portfolio: £{portfolio_value:.2f}")
    print(f"Realised Profits: £{realised_profit:.2f}")
    print(f"Total Wealth: £{total_wealth:.2f}")
    print(f"Total Return: {(total_wealth / INITIAL_CAPITAL - 1) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()