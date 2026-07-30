"""
final_performance_tracker.py
----------------------------
Tracks portfolio performance from the injection date onwards.
Uses the SAME price lookup as final_trading_engine.py.
FULL rebalancing logic: Kelly sign change detection, drift-based rebalancing, GARCH cash allocation.
Shows No Rules lines for Original_18, Optimal_Standard, and Optimal_Ethical.
OUTPUTS: Daily values for all portfolios.
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

# === PATHS ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for phase in ["Phase_2", "Phase_3", "Phase_5"]:
    path = os.path.join(parent_dir, phase)
    if os.path.exists(path):
        sys.path.insert(0, path)

# === IMPORTS ===
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

# === CONSTANTS ===
UNIVERSE_COLORS = {"Original_18": '#1f77b4', "Optimal_Standard": '#ff7f0e', "Optimal_Ethical": '#2ca02c'}
SPY_COLOR = '#d62728'

FINAL_ENGINE_DIR = os.path.join(parent_dir, "Final_Engine")
FINAL_LOGS_DIR = os.path.join(FINAL_ENGINE_DIR, "logs")
FINAL_FIGURES_DIR = os.path.join(FINAL_ENGINE_DIR, "figures")
os.makedirs(FINAL_LOGS_DIR, exist_ok=True)
os.makedirs(FINAL_FIGURES_DIR, exist_ok=True)

print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Logs directory: {FINAL_LOGS_DIR}")
print(f"📁 Figures directory: {FINAL_FIGURES_DIR}")

# ============================================================================
# PRICE FUNCTIONS - SAME AS TRADING ENGINE
# ============================================================================

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


def get_all_tickers():
    """Get all unique tickers across all universes."""
    all_tickers = set()
    all_tickers.update(ORIGINAL_UNIVERSE)
    all_tickers.update(OPTIMAL_STANDARD_PORTFOLIO)
    all_tickers.update(OPTIMAL_ETHICAL_PORTFOLIO)
    return list(all_tickers)


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


def get_closing_price_for_date(ticker, date, prices_df):
    """Get the CLOSING price for a specific ticker on a specific date."""
    try:
        start = date - timedelta(days=1)
        end = date + timedelta(days=1)
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if len(data) == 0:
            return get_previous_closing_price(ticker, date, prices_df)
        
        if "Close" in data.columns:
            closes = data["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            
            target_ts = pd.Timestamp(date)
            if target_ts in closes.index:
                val = closes.loc[target_ts]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if val > 0:
                    return float(val)
            
            idx = closes.index[closes.index <= target_ts]
            if len(idx) > 0:
                val = closes.loc[idx[-1]]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if val > 0:
                    return float(val)
        
        return get_previous_closing_price(ticker, date, prices_df)
    except:
        return get_previous_closing_price(ticker, date, prices_df)


def get_closing_prices_for_date(date, tickers, prices_df):
    """Get CLOSING prices for all tickers on a specific date."""
    prices = {}
    for t in tickers:
        prices[t] = get_closing_price_for_date(t, date, prices_df)
    return prices


def get_trading_days(start, end):
    return [d.date() for d in pd.date_range(start, end, freq='B') if d.weekday() < 5]


def daily_cash_rate(annual):
    return (1 + annual) ** (1 / 252) - 1 if annual > 0 else 0.0


# ============================================================================
# REBALANCE FUNCTIONS - FROM TRADING ENGINE
# ============================================================================

def calculate_drift(current, target):
    max_drift, max_asset = 0.0, None
    for t in current:
        drift = abs(current.get(t, 0.0) - target.get(t, 0.0))
        if drift > max_drift:
            max_drift, max_asset = drift, t
    return max_drift, max_asset


def calc_cash_allocation(returns, risk_free_rate, kelly_lookback, kelly_base_cap, kelly_max_cap,
                         cash_min_volatility, cash_max_volatility, cash_max_allocation):
    """Calculate cash allocation using Kelly + GARCH - SAME as trading engine."""
    try:
        from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
        from portfolio_optimiser import optimise_portfolios
        from data_fetcher import calculate_annualised_stats
        
        # Maximum Sharpe portfolio
        exp_ret, cov, _ = calculate_annualised_stats(returns)
        opt_results = optimise_portfolios(exp_ret, cov, risk_free_rate)
        weights = opt_results['msr_weights']
        mu = np.sum(exp_ret * weights)
        sigma = np.sqrt(weights.T @ cov @ weights)
        f_star = (mu - risk_free_rate) / (sigma ** 2) if sigma > 0 else 0.0
        
        # Kelly cash cap
        cash_cap = kelly_base_cap if f_star > 0 else kelly_max_cap
        
        # GARCH volatility
        models, _, _ = fit_garch_for_assets(returns)
        avg_vol = get_average_volatility(get_latest_volatility(models, returns))
        
    except Exception:
        avg_vol = returns.std().mean() * np.sqrt(252)
        cash_cap = cash_max_allocation
        f_star = 0.0
    
    # Cash allocation based on GARCH volatility
    if avg_vol <= cash_min_volatility:
        return 0.0, f_star
    elif avg_vol >= cash_max_volatility:
        return cash_cap, f_star
    else:
        fraction = (avg_vol - cash_min_volatility) / (cash_max_volatility - cash_min_volatility)
        return fraction * cash_cap, f_star


def generate_orders(value, current_weights, target_weights, cash_pounds, target_cash_pounds, tickers):
    """Generate buy/sell orders - SAME as trading engine."""
    orders = {}
    total_trade = 0.0
    
    cash_diff = target_cash_pounds - cash_pounds
    if abs(cash_diff) > 0.01:
        orders['CASH'] = {'action': 'BUY' if cash_diff > 0 else 'SELL', 'amount': abs(cash_diff)}
        total_trade += abs(cash_diff)
    
    for t in tickers:
        current_val = value * current_weights.get(t, 0.0)
        target_val = value * target_weights.get(t, 0.0)
        diff = target_val - current_val
        if abs(diff) > 0.01:
            orders[t] = {'action': 'BUY' if diff > 0 else 'SELL', 'amount': abs(diff)}
            total_trade += abs(diff)
    
    return orders, total_trade * TRANSACTION_COST_PCT


def optimise_portfolio(returns, risk_free_rate, tickers):
    """Optimise portfolio weights - SAME as trading engine."""
    try:
        from portfolio_optimiser import optimise_portfolios
        from data_fetcher import calculate_annualised_stats
        exp_ret, cov, _ = calculate_annualised_stats(returns)
        opt = optimise_portfolios(exp_ret, cov, risk_free_rate)
        return {t: opt['msr_weights'][i] for i, t in enumerate(tickers)}
    except:
        return {t: 1.0/len(tickers) for t in tickers}


# ============================================================================
# TRACKER WITH FULL REBALANCING LOGIC
# ============================================================================

def get_injection_state():
    """Get injection state from the log."""
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    if not os.path.exists(log_file):
        return None, None, None
    df = pd.read_csv(log_file)
    if len(df) == 0:
        return None, None, None
    first = df.iloc[0]
    date = pd.to_datetime(first['date']).date()
    asset_values = {t: float(first.get(f'{t}_value', 0)) for t in TICKERS}
    cash = float(first.get('cash_pounds', 0))
    return date, asset_values, cash


def run_universe_tracker(tickers, params, name, prices_df):
    """
    Track portfolio with FULL rebalancing logic.
    Combines: correct price lookup + Kelly sign change + drift-based rebalancing.
    """
    print(f"\n🚀 {name}...")
    print(f"   Kelly lookback: {params['kelly_lookback']} days")
    print(f"   Rebalance: {params['rebalance_min_days']}-{params['rebalance_max_days']} days")
    print(f"   Drift threshold: {params['drift_threshold']*100:.1f}%")
    
    injection_date, asset_values, cash = get_injection_state()
    if injection_date is None:
        print(f"   ⚠️ No injection date found, skipping {name}")
        return None, None
    
    # Filter asset values for this universe
    asset_values_filtered = {t: asset_values.get(t, 0.0) for t in tickers}
    
    today = datetime.now().date()
    days = get_trading_days(injection_date, today)
    
    dates, values = [], []
    current_values = asset_values_filtered.copy()
    current_cash = cash
    cash_pct = cash / (sum(current_values.values()) + cash) if (sum(current_values.values()) + cash) > 0 else 0.0
    daily_rate = daily_cash_rate(CASH_INTEREST_RATE)
    
    # State tracking
    prev_f_star = None
    last_rebalance = injection_date
    kelly_sign_changes = 0
    
    # Params
    lookback_days = params['lookback_days']
    rebalance_min = params['rebalance_min_days']
    rebalance_max = params['rebalance_max_days']
    drift_threshold = params['drift_threshold']
    kelly_lookback = params['kelly_lookback']
    
    # Initial state
    total = sum(current_values.values()) + current_cash
    dates.append(injection_date)
    values.append(total)
    
    print(f"\n   📅 Tracking from {injection_date} to {today} ({len(days)} trading days)")
    
    for day in days:
        if day == injection_date:
            continue
        
        prev_day = day - timedelta(days=1)
        while prev_day.weekday() >= 5:
            prev_day -= timedelta(days=1)
        
        # ---- APPLY PRICE CHANGES ----
        for t in tickers:
            if current_values.get(t, 0) > 0:
                today_price = get_closing_price_for_date(t, day, prices_df)
                yesterday_price = get_previous_closing_price(t, prev_day, prices_df)
                if yesterday_price > 0 and today_price > 0:
                    change = (today_price / yesterday_price) - 1
                    current_values[t] *= (1 + change)
        
        # ---- CASH INTEREST ----
        current_cash *= (1 + daily_rate)
        
        # ---- CALCULATE CURRENT STATE ----
        total = sum(current_values.values()) + current_cash
        cash_pct = current_cash / total if total > 0 else 0.0
        current_weights = {t: current_values[t] / total if total > 0 else 0.0 for t in tickers}
        
        # ---- REBALANCE CHECK ----
        # Get returns up to this day
        returns = prices_df[tickers].pct_change().dropna()
        returns = returns[returns.index <= pd.Timestamp(day)]
        
        if len(returns) > lookback_days * 0.5:
            # Optimise portfolio
            target_weights = optimise_portfolio(returns, RISK_FREE_RATE, tickers)
            
            # Calculate Kelly cash allocation
            target_cash_pct, f_star = calc_cash_allocation(
                returns, RISK_FREE_RATE, kelly_lookback,
                KELLY_BASE_CAP, KELLY_MAX_CAP,
                CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION
            )
            
            # ---- KELLY SIGN CHANGE DETECTION ----
            force_rebalance = False
            if prev_f_star is not None and (prev_f_star > 0) != (f_star > 0):
                kelly_sign_changes += 1
                force_rebalance = True
            prev_f_star = f_star
            
            # Adjust target weights for cash
            adjusted_target = {t: w * (1 - target_cash_pct) for t, w in target_weights.items()}
            
            # ---- REBALANCE CONDITIONS ----
            days_since = (day - last_rebalance).days
            drift, asset = calculate_drift(current_weights, adjusted_target)
            
            rebalance_needed = False
            reason = ""
            
            if sum(current_weights.values()) < 0.01:
                rebalance_needed = True
                reason = "First run"
            elif days_since >= rebalance_max:
                rebalance_needed = True
                reason = f"Time-based: {days_since} days"
            elif days_since >= rebalance_min and drift > drift_threshold:
                rebalance_needed = True
                reason = f"Drift: {asset} {drift*100:.2f}%"
            elif force_rebalance:
                rebalance_needed = True
                reason = "Kelly sign change"
            
            # ---- EXECUTE REBALANCE ----
            if rebalance_needed:
                target_cash = total * target_cash_pct
                
                # Apply rebalance
                new_values = {}
                for t in tickers:
                    new_values[t] = (total - target_cash) * adjusted_target.get(t, 0.0)
                
                # Only update if valid
                if sum(new_values.values()) > 0:
                    current_values = new_values
                    current_cash = target_cash
                    total = sum(current_values.values()) + current_cash
                    current_weights = {t: current_values[t] / total if total > 0 else 0.0 for t in tickers}
                    last_rebalance = day
                    print(f"   🔄 Rebalanced on {day.strftime('%Y-%m-%d')}: {reason}")
                    print(f"      Cash: {target_cash_pct*100:.1f}%, f*: {f_star:.3f}")
        
        dates.append(day)
        values.append(total)
    
    print(f"\n   ✅ {name} complete: {len(values)} days")
    print(f"   Kelly sign changes: {kelly_sign_changes}")
    
    return pd.Series(values, index=pd.DatetimeIndex(dates)), injection_date

# ============================================================================
# SPY
# ============================================================================

def download_spy(start_date, end_date):
    """Download SPY using closing prices."""
    print("   Downloading SPY...")
    try:
        end_dt = pd.Timestamp(end_date) + timedelta(days=2)
        spy = yf.download("SPY", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)["Close"]
        if len(spy) == 0:
            spy = yf.download("^GSPC", start=start_date, end=end_dt.strftime("%Y-%m-%d"), progress=False)["Close"]
        if len(spy) > 0:
            if isinstance(spy, pd.DataFrame):
                spy = spy.iloc[:, 0]
            business_days = pd.date_range(start=spy.index.min(), end=spy.index.max(), freq='B')
            spy = spy.reindex(business_days).ffill().bfill()
            print(f"   ✅ SPY: {len(spy)} days")
            return spy
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)

# ============================================================================
# HELPERS
# ============================================================================

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
    full = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='B')
    aligned = {}
    for name, s in data_dict.items():
        if s is not None and len(s) > 0:
            aligned[name] = s.reindex(full, method='ffill').bfill()
            if aligned[name].isna().any():
                first = aligned[name].first_valid_index()
                if first:
                    aligned[name] = aligned[name].fillna(aligned[name].loc[first])
        else:
            aligned[name] = pd.Series(dtype=float)
    return aligned

# ============================================================================
# PLOTTING
# ============================================================================

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
    print(f"✅ Saved: {filename}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FINAL PERFORMANCE TRACKER (Full Rebalancing Logic)")
    print("=" * 70)
    print(f"📊 Kelly caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")
    print("=" * 70)
    
    injection_date, _, _ = get_injection_state()
    if injection_date is None:
        print("❌ No injection date found. Run final_trading_engine.py first.")
        return
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    print(f"\n📅 Tracking from {injection_date} to {end_date}")
    print("=" * 70)
    
    # Fetch price data for ALL tickers across all universes
    all_tickers = get_all_tickers()
    print(f"\n📥 Fetching price data for {len(all_tickers)} tickers...")
    prices_df = fetch_price_data_with_forward_fill(all_tickers, LOOKBACK_DAYS)
    
    if prices_df is None or len(prices_df) == 0:
        print("❌ No price data available. Exiting.")
        return
    
    universes = {
        "Original_18": (ORIGINAL_UNIVERSE, ORIGINAL_PARAMS),
        "Optimal_Standard": (OPTIMAL_STANDARD_PORTFOLIO, OPTIMAL_STANDARD_PARAMS),
        "Optimal_Ethical": (OPTIMAL_ETHICAL_PORTFOLIO, OPTIMAL_ETHICAL_PARAMS),
    }
    
    print("\n📊 Kelly Lookback Values:")
    for name, (_, params) in universes.items():
        print(f"   {name}: {params['kelly_lookback']} days")
    print("=" * 70)
    
    all_data = {}
    
    for name, (tickers, params) in universes.items():
        print(f"\n{'='*60}\nRUNNING: {name}\n{'='*60}")
        series, inj_date = run_universe_tracker(tickers, params, name, prices_df)
        if series is not None:
            all_data[name] = normalise(series, inj_date)
    
    # SPY
    print("\n📊 Downloading SPY...")
    backtest_start = injection_date - timedelta(days=30)
    spy = download_spy(backtest_start.strftime("%Y-%m-%d"), end_date)
    all_data["SPY"] = normalise(spy, injection_date)
    
    # Align
    print("\n📊 Aligning series...")
    all_data = align_series(all_data)
    if all_data:
        first = next(iter(all_data.values()))
        print(f"   ✅ Aligned to {len(first)} business days")
    
    # PRINT DAILY VALUES
    print("\n" + "=" * 70)
    print("DAILY PORTFOLIO VALUES")
    print("=" * 70)
    
    combined_df = pd.DataFrame()
    for name, series in all_data.items():
        if series is not None and len(series) > 0:
            combined_df[name] = series
    
    print("\n" + combined_df.to_string(float_format=lambda x: f'{x:.2f}'))
    
    csv_path = os.path.join(FINAL_LOGS_DIR, "daily_portfolio_values.csv")
    combined_df.to_csv(csv_path, float_format='%.2f')
    print(f"\n✅ Daily values saved to: {csv_path}")
    
    # FINAL VALUES
    print("\n" + "=" * 70)
    print("FINAL VALUES")
    print("=" * 70)
    for name, series in all_data.items():
        if series is not None and len(series) > 0:
            val = float(series.iloc[-1])
            ret = (val / INITIAL_CAPITAL - 1) * 100
            print(f"   {name}: £{val:.2f} ({ret:.1f}%)")
    
    # PLOT
    print("\n📊 Generating plots...")
    title = f'No Rules Portfolio Performance Since Investment ({injection_date.strftime("%d/%m/%Y")})'
    plot_performance(all_data, injection_date, title, "final_performance_tracker.png", log_scale=False)
    plot_performance(all_data, injection_date, title, "final_performance_tracker_log.png", log_scale=True)

if __name__ == "__main__":
    main()