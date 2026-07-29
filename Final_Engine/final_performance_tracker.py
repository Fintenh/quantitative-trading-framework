"""
final_performance_tracker.py
----------------------------
Tracks portfolio performance from the injection date onwards.
Uses final_config.py for portfolio and parameters.
Shows No Rules lines for Original_18, Optimal_Standard, and Optimal_Ethical.
Kelly uses Maximum Sharpe portfolio with sign change detection.
UPDATED: Multi-exchange holiday handling with forward-fill for closed markets.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for phase in ["Phase_3", "Phase_2", "Phase_5"]:
    path = os.path.join(parent_dir, phase)
    if os.path.exists(path):
        import sys
        sys.path.insert(0, path)

from backtest_engine import BacktestEngine

# Import from configs
from ethical_universe import ORIGINAL_UNIVERSE
from final_config import (
    TICKERS,                    # Optimal_Ethical portfolio
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    TRANSACTION_COST_PCT,
    CASH_INTEREST_RATE,
    CASH_MIN_VOLATILITY,
    CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION,
    KELLY_BASE_CAP,
    KELLY_MAX_CAP,
    LOG_DIR,
)
from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO,
    OPTIMAL_ETHICAL_PORTFOLIO,
    ORIGINAL_PARAMS,
    OPTIMAL_STANDARD_PARAMS,
    OPTIMAL_ETHICAL_PARAMS,
)

# ============================================================================
# CONSTANTS
# ============================================================================

UNIVERSE_COLORS = {
    "Original_18": '#1f77b4',
    "Optimal_Standard": '#ff7f0e',
    "Optimal_Ethical": '#2ca02c',
}
SPY_COLOR = '#d62728'

# Paths
FINAL_ENGINE_DIR = os.path.join(parent_dir, "Final_Engine")
FINAL_LOGS_DIR = os.path.join(FINAL_ENGINE_DIR, "logs")
FINAL_FIGURES_DIR = os.path.join(FINAL_ENGINE_DIR, "figures")
os.makedirs(FINAL_LOGS_DIR, exist_ok=True)
os.makedirs(FINAL_FIGURES_DIR, exist_ok=True)

print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Logs directory: {FINAL_LOGS_DIR}")
print(f"📁 Figures directory: {FINAL_FIGURES_DIR}")

# ============================================================================
# BACKTEST ENGINE WITH KELLY (Matches ethical_backtest.py)
# ============================================================================

class BacktestEngineWithKelly(BacktestEngine):
    """BacktestEngine with Kelly + sign change detection."""
    
    def __init__(self, kelly_lookback=126, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.prev_f_star = None
        self.kelly_f_stars = []
        self.kelly_sign_changes = 0
        self._force_rebalance = False
    
    def _calc_cash_allocation(self, returns):
        """Calculate cash allocation using Kelly + GARCH with sign change detection."""
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            from portfolio_optimiser import optimise_portfolios
            from data_fetcher import calculate_annualised_stats
            
            # Maximum Sharpe portfolio
            exp_ret, cov, _ = calculate_annualised_stats(returns)
            opt_results = optimise_portfolios(exp_ret, cov, self.risk_free_rate)
            weights = opt_results['msr_weights']
            mu = np.sum(exp_ret * weights)
            sigma = np.sqrt(weights.T @ cov @ weights)
            f_star = (mu - self.risk_free_rate) / (sigma ** 2) if sigma > 0 else 0.0
            
            # Kelly cash cap - using config values
            cash_cap = KELLY_BASE_CAP if f_star > 0 else KELLY_MAX_CAP
            
            # Sign change detection (triggers emergency rebalance)
            current_sign = 1 if f_star > 0 else -1
            if self.prev_f_star is not None:
                prev_sign = 1 if self.prev_f_star > 0 else -1
                if current_sign != prev_sign:
                    self.kelly_sign_changes += 1
                    self._force_rebalance = True
            
            self.prev_f_star = f_star
            self.kelly_f_stars.append(f_star)
            
            # GARCH volatility
            models, _, _ = fit_garch_for_assets(returns)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns))
            
        except Exception:
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = self.cash_max_allocation
        
        # Cash allocation based on GARCH volatility
        if avg_vol <= self.cash_min_volatility:
            return 0.0
        elif avg_vol >= self.cash_max_volatility:
            return cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            return fraction * cash_cap
    
    def should_force_rebalance(self):
        """Check if Kelly sign change should force immediate rebalance."""
        if self._force_rebalance:
            self._force_rebalance = False
            return True
        return False


# ============================================================================
# HELPERS
# ============================================================================

def get_injection_date():
    """Get injection date from portfolio_log.csv in Final_Engine/logs."""
    log_file = os.path.join(FINAL_LOGS_DIR, "portfolio_log.csv")
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return None
    
    df = pd.read_csv(log_file)
    if len(df) == 0:
        print("❌ Log file is empty.")
        return None
    
    first_date = pd.to_datetime(df['date'].iloc[0]).date()
    print(f"📅 Injection date: {first_date}")
    return first_date


# ============================================================================
# MULTI-EXCHANGE FUNCTIONS WITH HOLIDAY HANDLING
# ============================================================================

def download_spy(start_date, end_date):
    """
    Download SPY data with forward-fill for holidays.
    If SPY is closed (US holiday), carries the last price forward.
    """
    print("   Downloading SPY with holiday handling...")
    
    try:
        # Try SPY first
        spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
        
        # If SPY fails, try S&P 500 index
        if len(spy_data) == 0:
            print("   ⚠️ SPY not available, trying ^GSPC...")
            spy_data = yf.download("^GSPC", start=start_date, end=end_date, progress=False)["Close"]
        
        # If we have data, apply forward-fill for holidays
        if len(spy_data) > 0:
            # Create full business day range
            business_days = pd.date_range(
                start=spy_data.index.min(),
                end=spy_data.index.max(),
                freq='B'  # Business days only
            )
            
            # Reindex to business days and forward-fill (holiday handling)
            spy_data = spy_data.reindex(business_days)
            spy_data = spy_data.ffill()  # Forward-fill holidays
            spy_data = spy_data.bfill()  # Back-fill start if needed
            
            print(f"   ✅ SPY: {len(spy_data)} business days of data")
            
            # Count how many days were forward-filled (holidays)
            original_len = len(yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"])
            if original_len > 0:
                filled_days = len(spy_data) - original_len
                if filled_days > 0:
                    print(f"   📌 Forward-filled {filled_days} holiday days")
            
            return spy_data
        
        # Ultimate fallback - generate dummy data
        print("   ⚠️ No SPY data available - using flat line")
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        spy_data = pd.Series([100.0] * len(dates), index=dates)
        return spy_data
        
    except Exception as e:
        print(f"   ⚠️ Error downloading SPY: {e}")
        # Fallback to dummy data
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.Series([100.0] * len(dates), index=dates)


def align_multi_exchange_data(data_dict):
    """
    Align multiple tickers/data series to the same business day index.
    All series get forward-filled so they have data on every business day.
    
    Args:
        data_dict: Dictionary of {name: Series} to align
    
    Returns:
        dict: Aligned series with common business day index
    """
    # Collect all dates from all series
    all_dates = pd.DatetimeIndex([])
    for series in data_dict.values():
        if series is not None and len(series) > 0:
            all_dates = all_dates.union(series.index)
    
    if len(all_dates) == 0:
        return data_dict
    
    # Create full business day range
    full_range = pd.date_range(
        start=all_dates.min(),
        end=all_dates.max(),
        freq='B'
    )
    
    # Reindex and forward-fill each series
    aligned = {}
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            # Reindex to full business day range
            aligned[name] = series.reindex(full_range, method='ffill')
            # Back-fill any leading NaNs (FIXED: use bfill() directly)
            aligned[name] = aligned[name].bfill()
            # If still NaN, fill with first valid value
            if aligned[name].isna().any():
                first_valid = aligned[name].first_valid_index()
                if first_valid:
                    aligned[name] = aligned[name].fillna(aligned[name].loc[first_valid])
        else:
            aligned[name] = pd.Series(dtype=float)
    
    return aligned


def normalise_to_injection(series, injection_date):
    """Normalise series to start at £100 on injection date."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    
    crop = series[series.index >= pd.Timestamp(injection_date)]
    if len(crop) == 0:
        first_after = series[series.index > pd.Timestamp(injection_date)]
        if len(first_after) > 0:
            crop = first_after
        else:
            return pd.Series(dtype=float)
    
    first_val = float(crop.iloc[0])
    if first_val <= 0:
        return pd.Series(dtype=float)
    
    return (crop / first_val) * INITIAL_CAPITAL


def run_backtest_for_universe(tickers, params, start_date, end_date, kelly_lookback, universe_name):
    """
    Run backtest for one universe (No Rules only).
    Matches ethical_backtest.py's "No Rules" configuration.
    """
    print(f"\n🚀 Running {universe_name}...")
    print(f"   Kelly lookback: {kelly_lookback} days")
    print(f"   Kelly caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")
    
    engine = BacktestEngineWithKelly(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=params['lookback_days'],
        rebalance_min_days=params['rebalance_min_days'],
        rebalance_max_days=params['rebalance_max_days'],
        drift_threshold=params['drift_threshold'],
        take_profit_pct=0.0,  # No Rules = no take-profit
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=kelly_lookback
    )
    results = engine.run()
    no_rules = results['equity_curve']['value']
    
    # Kelly stats
    if len(engine.kelly_f_stars) > 0:
        f_stars = np.array(engine.kelly_f_stars)
        print(f"   f* mean={np.mean(f_stars):.4f}, sign_changes={engine.kelly_sign_changes}")
    
    return no_rules


# ============================================================================
# PLOTTING
# ============================================================================

def add_price_annotation(ax, series, color, label, x_pos=0.02, y_pos=0.02):
    """
    Add the last price of a series as an annotation in the corner of the plot.
    """
    if series is None or len(series) == 0:
        return
    
    last_value = float(series.iloc[-1])
    last_date = series.index[-1].strftime('%d/%m/%Y')
    
    annotation = f"{label}: £{last_value:,.2f} ({last_date})"
    
    ax.text(
        x_pos, y_pos, annotation,
        transform=ax.transAxes,
        fontsize=9,
        color=color,
        weight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor=color,
            alpha=0.85
        ),
        verticalalignment='bottom'
    )


def create_performance_plot(ax, data_dict, title, is_log=False):
    """Create a performance plot with last price annotations."""
    
    # Plot all series
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            if name == "SPY":
                color = SPY_COLOR
                linewidth = 3
                label = 'SPY (Benchmark)'
                marker = 'o'
            else:
                color = UNIVERSE_COLORS.get(name, '#888888')
                linewidth = 2
                label = f'{name} - No Rules'
                marker = 'o'
            
            if is_log:
                ax.semilogy(series.index, series, color=color, linestyle='-', 
                           linewidth=linewidth, label=label, marker=marker, markersize=6)
            else:
                ax.plot(series.index, series, color=color, linestyle='-', 
                       linewidth=linewidth, label=label, marker=marker, markersize=6)
    
    # Collect data for axis scaling
    all_data = []
    all_dates = []
    for series in data_dict.values():
        if series is not None and len(series) > 0:
            all_data.extend(series.values)
            all_dates.extend(series.index)
    
    if not all_data:
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Auto-scale axes
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = (max_date - min_date).days
    padding = max(date_range * 0.05, 2)
    ax.set_xlim(min_date - pd.Timedelta(days=padding), max_date + pd.Timedelta(days=padding))
    
    min_val = min(all_data)
    max_val = max(all_data)
    if not is_log:
        range_val = max_val - min_val
        padding_y = max(range_val * 0.1, 5)
        ax.set_ylim(min_val - padding_y, max_val + padding_y)
    else:
        positive_data = [v for v in all_data if v > 0]
        if positive_data:
            ax.set_ylim(min(positive_data) * 0.9, max(all_data) * 1.1)
    
    # Formatting
    ax.set_xlabel('Date')
    y_label = 'Portfolio Value (£)' + (' - Log Scale' if is_log else '')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    # Add price annotations in the corner
    y_offset = 0.02
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            color = SPY_COLOR if name == "SPY" else UNIVERSE_COLORS.get(name, '#888888')
            label_short = name.replace('_', ' ').replace(' - No Rules', '')
            add_price_annotation(ax, series, color, label_short, x_pos=0.02, y_pos=y_offset)
            y_offset += 0.045


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FINAL PERFORMANCE TRACKER (Multi-Exchange with Holiday Handling)")
    print("=" * 70)
    print(f"📊 Optimal_Ethical: {len(TICKERS)} assets")
    print(f"📊 Kelly caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")
    print("=" * 70)
    
    # Get injection date
    injection_date = get_injection_date()
    if injection_date is None:
        print("\n❌ No injection date found.")
        print("   Make sure final_trading_engine.py has been run at least once.")
        return
    
    # Set date range
    injection_date_str = injection_date.strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    backtest_start = injection_date - timedelta(days=400)
    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    
    print(f"\n📅 Tracking from: {injection_date_str} to {end_date}")
    print("=" * 70)
    
    # Define universes - using config portfolios
    universes = {
        "Original_18": {
            'tickers': ORIGINAL_UNIVERSE,
            'params': ORIGINAL_PARAMS,
            'kelly': ORIGINAL_PARAMS['kelly_lookback'],
        },
        "Optimal_Standard": {
            'tickers': OPTIMAL_STANDARD_PORTFOLIO,
            'params': OPTIMAL_STANDARD_PARAMS,
            'kelly': OPTIMAL_STANDARD_PARAMS['kelly_lookback'],
        },
        "Optimal_Ethical": {
            'tickers': OPTIMAL_ETHICAL_PORTFOLIO,
            'params': OPTIMAL_ETHICAL_PARAMS,
            'kelly': OPTIMAL_ETHICAL_PARAMS['kelly_lookback'],
        },
    }
    
    # Print Kelly lookback values
    print("\n📊 Kelly Lookback Values:")
    for name, config in universes.items():
        print(f"   {name}: {config['kelly']} days")
    print("=" * 70)
    
    # Run backtests
    all_no_rules = {}
    for name, config in universes.items():
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print(f"{'='*60}")
        no_rules = run_backtest_for_universe(
            config['tickers'],
            config['params'],
            backtest_start_str,
            end_date,
            config['kelly'],
            name
        )
        all_no_rules[name] = no_rules
    
    # Download and normalise SPY (with holiday handling)
    print("\n📊 Downloading SPY with holiday handling...")
    spy = download_spy(backtest_start_str, end_date)
    
    # Normalise all series
    print("\n" + "=" * 70)
    print("NORMALISING TO INJECTION DATE")
    print("=" * 70)
    print(f"   Injection date: {injection_date}")
    
    norm_data = {}
    for name, series in all_no_rules.items():
        norm_data[name] = normalise_to_injection(series, injection_date)
    norm_data["SPY"] = normalise_to_injection(spy, injection_date)
    
    # ALIGN ALL SERIES TO COMMON BUSINESS DAY INDEX
    print("\n📊 Aligning all series to common business day index...")
    norm_data = align_multi_exchange_data(norm_data)
    
    # Count how many days we have
    if norm_data:
        first_series = next(iter(norm_data.values()))
        if first_series is not None and len(first_series) > 0:
            print(f"   ✅ All series aligned to {len(first_series)} business days")
    
    # Print final values
    print("\n" + "=" * 70)
    print("FINAL VALUES")
    print("=" * 70)
    
    for name, series in norm_data.items():
        if series is not None and len(series) > 0:
            val = float(series.iloc[-1])
            ret = (val / INITIAL_CAPITAL - 1) * 100
            print(f"   {name}: £{val:.2f} ({ret:.1f}%)")
    
    # Create plots
    print("\n📊 Generating plots...")
    
    title = f'No Rules Portfolio Performance Since Investment ({injection_date.strftime("%d/%m/%Y")})'
    
    # Linear plot
    fig, ax = plt.subplots(figsize=(14, 8))
    create_performance_plot(ax, norm_data, title, is_log=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_FIGURES_DIR, "final_performance_tracker.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Saved: {FINAL_FIGURES_DIR}/final_performance_tracker.png")
    
    # Log plot
    fig, ax = plt.subplots(figsize=(14, 8))
    create_performance_plot(ax, norm_data, title + ' - Log Scale', is_log=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_FIGURES_DIR, "final_performance_tracker_log.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Saved: {FINAL_FIGURES_DIR}/final_performance_tracker_log.png")
    
    print("\n" + "=" * 70)
    print("✅ PERFORMANCE TRACKER COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()