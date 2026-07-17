"""
final_performance_tracker.py
----------------------------
Tracks portfolio performance from the injection date onwards.
Simple, clean, no unnecessary complexity.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_5"))

from backtest_engine import BacktestEngine
from ethical_universe import ORIGINAL_UNIVERSE, STANDARD_UNIVERSE, ETHICAL_UNIVERSE
from ethical_config import ORIGINAL_PARAMS, OPTIMAL_STANDARD_PARAMS, OPTIMAL_ETHICAL_PARAMS

# ============================================================================
# CONFIGURATION
# ============================================================================

INITIAL_CAPITAL = 100.0
RISK_FREE_RATE = 0.045
TRANSACTION_COST_PCT = 0.004
CASH_INTEREST_RATE = 0.0525
CASH_MIN_VOLATILITY = 0.25
CASH_MAX_VOLATILITY = 0.50

FIGURES_DIR = os.path.join(script_dir, "figures")
LOGS_DIR = os.path.join(script_dir, "logs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Colours
UNIVERSE_COLORS = {
    "Original_18": {
        'no_rules': '#1f77b4', 'active': '#6baed6', 'total_wealth': '#9ecae1'
    },
    "Optimal_Standard": {
        'no_rules': '#ff7f0e', 'active': '#ffb347', 'total_wealth': '#ffd7a0'
    },
    "Optimal_Ethical": {
        'no_rules': '#2ca02c', 'active': '#66c266', 'total_wealth': '#a8dba8'
    },
}
SPY_COLOR = '#d62728'
LINE_STYLES = {'no_rules': '-', 'active': '--', 'total_wealth': '-.'}


# ============================================================================
# BACKTEST ENGINE WITH KELLY
# ============================================================================

class BacktestEngineWithKelly(BacktestEngine):
    def __init__(self, kelly_lookback=126, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_f_stars, self.kelly_cash_caps, self.kelly_cash_allocations = [], [], []
    
    def _calc_cash_allocation(self, returns):
        """Calculate cash allocation using Kelly."""
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            from portfolio_optimiser import optimise_portfolios
            from data_fetcher import calculate_annualised_stats
            
            exp_ret, cov, _ = calculate_annualised_stats(returns)
            opt_results = optimise_portfolios(exp_ret, cov, self.risk_free_rate)
            weights = opt_results['msr_weights']
            mu = np.sum(exp_ret * weights)
            sigma = np.sqrt(weights.T @ cov @ weights)
            f_star = (mu - self.risk_free_rate) / (sigma ** 2) if sigma > 0 else 0.0
            
            cash_cap = 0.20 if f_star > 0 else 1.00
            self.kelly_f_stars.append(f_star)
            self.kelly_cash_caps.append(cash_cap)
            
            models, _, _ = fit_garch_for_assets(returns)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns))
        except:
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = 0.20
        
        if avg_vol <= self.cash_min_volatility:
            return 0.0
        elif avg_vol >= self.cash_max_volatility:
            return cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            return fraction * cash_cap


# ============================================================================
# RUN BACKTEST
# ============================================================================

def run_backtest_for_universe(tickers, params, start_date, end_date, kelly_lookback):
    """Run backtest for one universe."""
    
    # With rules (take-profit enabled)
    engine = BacktestEngineWithKelly(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=params['lookback_days'],
        rebalance_min_days=params['rebalance_min_days'],
        rebalance_max_days=params['rebalance_max_days'],
        drift_threshold=params['drift_threshold'],
        take_profit_pct=params['take_profit_pct'],
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=kelly_lookback
    )
    results = engine.run()
    active = results['equity_curve']['value']
    total_wealth = results['wealth_curve']['value']
    
    # No rules (take-profit disabled)
    engine_nr = BacktestEngineWithKelly(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=params['lookback_days'],
        rebalance_min_days=params['rebalance_min_days'],
        rebalance_max_days=params['rebalance_max_days'],
        drift_threshold=params['drift_threshold'],
        take_profit_pct=0.0,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=kelly_lookback
    )
    results_nr = engine_nr.run()
    no_rules = results_nr['equity_curve']['value']
    
    return {'no_rules': no_rules, 'active': active, 'total_wealth': total_wealth}


# ============================================================================
# GET INJECTION DATE
# ============================================================================

def get_injection_date():
    """Get injection date from portfolio_log.csv."""
    log_file = os.path.join(LOGS_DIR, "portfolio_log.csv")
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
# SAFE LATEST VALUE
# ============================================================================

def latest(series):
    """Get the latest value from a series safely."""
    if series is None or len(series) == 0:
        return INITIAL_CAPITAL
    try:
        val = series.iloc[-1]
        if hasattr(val, 'values'):
            return float(val.values[0])
        return float(val)
    except Exception:
        return INITIAL_CAPITAL


# ============================================================================
# PRINT ALL 10 LINES
# ============================================================================

def print_all_lines(orig_no_rules, orig_active, orig_wealth,
                    std_no_rules, std_active, std_wealth,
                    eth_no_rules, eth_active, eth_wealth,
                    spy, injection_date):
    """Print all 10 lines with final values and returns."""
    
    print("\n" + "=" * 80)
    print("ALL 10 LINES - FINAL VALUES")
    print("=" * 80)
    
    # Helper to get return
    def get_return(series):
        val = latest(series)
        return (val / INITIAL_CAPITAL - 1) * 100
    
    # SPY
    spy_val = latest(spy)
    spy_ret = (spy_val / INITIAL_CAPITAL - 1) * 100
    
    print(f"\n📊 SPY (Benchmark):")
    print(f"   Final Value: £{spy_val:.2f}  |  Return: {spy_ret:.2f}%")
    print()
    
    print(f"📊 Original_18:")
    print(f"   No Rules:     £{latest(orig_no_rules):.2f}  |  Return: {get_return(orig_no_rules):.2f}%")
    print(f"   Active:       £{latest(orig_active):.2f}  |  Return: {get_return(orig_active):.2f}%")
    print(f"   Total Wealth: £{latest(orig_wealth):.2f}  |  Return: {get_return(orig_wealth):.2f}%")
    print()
    
    print(f"📊 Optimal_Standard:")
    print(f"   No Rules:     £{latest(std_no_rules):.2f}  |  Return: {get_return(std_no_rules):.2f}%")
    print(f"   Active:       £{latest(std_active):.2f}  |  Return: {get_return(std_active):.2f}%")
    print(f"   Total Wealth: £{latest(std_wealth):.2f}  |  Return: {get_return(std_wealth):.2f}%")
    print()
    
    print(f"📊 Optimal_Ethical:")
    print(f"   No Rules:     £{latest(eth_no_rules):.2f}  |  Return: {get_return(eth_no_rules):.2f}%")
    print(f"   Active:       £{latest(eth_active):.2f}  |  Return: {get_return(eth_active):.2f}%")
    print(f"   Total Wealth: £{latest(eth_wealth):.2f}  |  Return: {get_return(eth_wealth):.2f}%")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN TRACKER
# ============================================================================

def main():
    print("=" * 60)
    print("FINAL PERFORMANCE TRACKER")
    print("=" * 60)
    
    # Get injection date
    injection_date = get_injection_date()
    if injection_date is None:
        return
    
    # Set date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    injection_date_str = injection_date.strftime("%Y-%m-%d")
    
    # Start backtest 400 days before injection (enough for lookback)
    backtest_start = injection_date - timedelta(days=400)
    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    
    print(f"📅 Running backtest from {backtest_start_str} to {end_date}")
    print(f"📅 Tracking from injection date: {injection_date_str}")
    print()
    
    # Run backtests
    print("📊 Running Original_18...")
    orig = run_backtest_for_universe(
        ORIGINAL_UNIVERSE, ORIGINAL_PARAMS, backtest_start_str, end_date, 126
    )
    
    print("📊 Running Optimal_Standard...")
    std = run_backtest_for_universe(
        STANDARD_UNIVERSE, OPTIMAL_STANDARD_PARAMS, backtest_start_str, end_date, 280
    )
    
    print("📊 Running Optimal_Ethical...")
    eth = run_backtest_for_universe(
        ETHICAL_UNIVERSE, OPTIMAL_ETHICAL_PARAMS, backtest_start_str, end_date, 280
    )
    
    # Get SPY from injection date - ensure it's a Series
    spy_data = yf.download("SPY", start=injection_date_str, end=end_date, progress=False)["Close"]
    if isinstance(spy_data, pd.DataFrame):
        spy = (spy_data.iloc[:, 0] / spy_data.iloc[0, 0]) * INITIAL_CAPITAL
    else:
        spy = (spy_data / spy_data.iloc[0]) * INITIAL_CAPITAL
    
    # ================================================================
    # NORMALISE EVERYTHING TO START AT £100 ON INJECTION DATE
    # ================================================================
    
    def normalise_from_injection(series, injection_date):
        """Normalise a series so it starts at £100 on the injection date."""
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)
        
        # Crop from injection date
        crop = series[series.index >= pd.Timestamp(injection_date)]
        if len(crop) == 0:
            return pd.Series(dtype=float)
        
        # Normalise to £100 at first value
        first_val = float(crop.iloc[0])
        if first_val > 0:
            return (crop / first_val) * INITIAL_CAPITAL
        return crop
    
    # Normalise each series
    orig_no_rules = normalise_from_injection(orig['no_rules'], injection_date)
    orig_active = normalise_from_injection(orig['active'], injection_date)
    orig_wealth = normalise_from_injection(orig['total_wealth'], injection_date)
    
    std_no_rules = normalise_from_injection(std['no_rules'], injection_date)
    std_active = normalise_from_injection(std['active'], injection_date)
    std_wealth = normalise_from_injection(std['total_wealth'], injection_date)
    
    eth_no_rules = normalise_from_injection(eth['no_rules'], injection_date)
    eth_active = normalise_from_injection(eth['active'], injection_date)
    eth_wealth = normalise_from_injection(eth['total_wealth'], injection_date)
    
    # ================================================================
    # SUMMARY
    # ================================================================
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    latest_spy = latest(spy)
    latest_orig = latest(orig_wealth)
    latest_std = latest(std_wealth)
    latest_eth = latest(eth_wealth)
    
    spy_return = (latest_spy / INITIAL_CAPITAL - 1) * 100
    orig_return = (latest_orig / INITIAL_CAPITAL - 1) * 100
    std_return = (latest_std / INITIAL_CAPITAL - 1) * 100
    eth_return = (latest_eth / INITIAL_CAPITAL - 1) * 100
    
    print(f"\n📊 Period: {injection_date} to {spy.index[-1].date()}")
    print(f"\n📊 Final Values (starting at £100):")
    print(f"   SPY: £{latest_spy:.2f} ({spy_return:.1f}%)")
    print(f"   Original_18: £{latest_orig:.2f} ({orig_return:.1f}%)")
    print(f"   Optimal_Standard: £{latest_std:.2f} ({std_return:.1f}%)")
    print(f"   Optimal_Ethical: £{latest_eth:.2f} ({eth_return:.1f}%)")
    
    print(f"\n📊 Outperformance vs SPY:")
    print(f"   Original_18: {orig_return - spy_return:.1f}%")
    print(f"   Optimal_Standard: {std_return - spy_return:.1f}%")
    print(f"   Optimal_Ethical: {eth_return - spy_return:.1f}%")
    
    # ================================================================
    # PRINT ALL 10 LINES
    # ================================================================
    
    print_all_lines(orig_no_rules, orig_active, orig_wealth,
                    std_no_rules, std_active, std_wealth,
                    eth_no_rules, eth_active, eth_wealth,
                    spy, injection_date)
    
    # ================================================================
    # PLOT - LINEAR
    # ================================================================
    
    print("\n📊 Generating plots...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot all lines
    def plot_series(series, color, linestyle, lw, label):
        if len(series) > 0:
            ax.plot(series.index, series, color=color, linestyle=linestyle, linewidth=lw, label=label)
    
    # Original_18
    plot_series(orig_no_rules, UNIVERSE_COLORS["Original_18"]['no_rules'], LINE_STYLES['no_rules'], 2, 'Original_18 - No Rules')
    plot_series(orig_active, UNIVERSE_COLORS["Original_18"]['active'], LINE_STYLES['active'], 2, 'Original_18 - Active')
    plot_series(orig_wealth, UNIVERSE_COLORS["Original_18"]['total_wealth'], LINE_STYLES['total_wealth'], 2.5, 'Original_18 - Total Wealth')
    
    # Optimal_Standard
    plot_series(std_no_rules, UNIVERSE_COLORS["Optimal_Standard"]['no_rules'], LINE_STYLES['no_rules'], 2, 'Optimal_Standard - No Rules')
    plot_series(std_active, UNIVERSE_COLORS["Optimal_Standard"]['active'], LINE_STYLES['active'], 2, 'Optimal_Standard - Active')
    plot_series(std_wealth, UNIVERSE_COLORS["Optimal_Standard"]['total_wealth'], LINE_STYLES['total_wealth'], 2.5, 'Optimal_Standard - Total Wealth')
    
    # Optimal_Ethical
    plot_series(eth_no_rules, UNIVERSE_COLORS["Optimal_Ethical"]['no_rules'], LINE_STYLES['no_rules'], 2, 'Optimal_Ethical - No Rules')
    plot_series(eth_active, UNIVERSE_COLORS["Optimal_Ethical"]['active'], LINE_STYLES['active'], 2, 'Optimal_Ethical - Active')
    plot_series(eth_wealth, UNIVERSE_COLORS["Optimal_Ethical"]['total_wealth'], LINE_STYLES['total_wealth'], 2.5, 'Optimal_Ethical - Total Wealth')
    
    # SPY
    if len(spy) > 0:
        ax.plot(spy.index, spy, color=SPY_COLOR, linestyle='-', linewidth=3, label='SPY (Benchmark)')
    
    # Formatting
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=pd.Timestamp(injection_date), color='black', linestyle='--', alpha=0.5, label='Investment Date')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (£)')
    
    title = f'Portfolio Performance Since Investment ({injection_date})\n'
    title += f'Original: £{latest_orig:.2f}  |  Standard: £{latest_std:.2f}  |  Ethical: £{latest_eth:.2f}  |  SPY: £{latest_spy:.2f}'
    ax.set_title(title)
    
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "final_performance_tracker_linear.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✅ Saved: final_performance_tracker_linear.png")
    
    # ================================================================
    # PLOT - LOG SCALE
    # ================================================================
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    def plot_log(series, color, linestyle, lw, label):
        if len(series) > 0:
            try:
                if (series > 0).all():
                    ax.semilogy(series.index, series, color=color, linestyle=linestyle, linewidth=lw, label=label)
            except:
                pass
    
    # Original_18
    plot_log(orig_no_rules, UNIVERSE_COLORS["Original_18"]['no_rules'], LINE_STYLES['no_rules'], 2, 'Original_18 - No Rules')
    plot_log(orig_active, UNIVERSE_COLORS["Original_18"]['active'], LINE_STYLES['active'], 2, 'Original_18 - Active')
    plot_log(orig_wealth, UNIVERSE_COLORS["Original_18"]['total_wealth'], LINE_STYLES['total_wealth'], 2.5, 'Original_18 - Total Wealth')
    
    # Optimal_Standard
    plot_log(std_no_rules, UNIVERSE_COLORS["Optimal_Standard"]['no_rules'], LINE_STYLES['no_rules'], 2, 'Optimal_Standard - No Rules')
    plot_log(std_active, UNIVERSE_COLORS["Optimal_Standard"]['active'], LINE_STYLES['active'], 2, 'Optimal_Standard - Active')
    plot_log(std_wealth, UNIVERSE_COLORS["Optimal_Standard"]['total_wealth'], LINE_STYLES['total_wealth'], 2.5, 'Optimal_Standard - Total Wealth')
    
    # Optimal_Ethical
    plot_log(eth_no_rules, UNIVERSE_COLORS["Optimal_Ethical"]['no_rules'], LINE_STYLES['no_rules'], 2, 'Optimal_Ethical - No Rules')
    plot_log(eth_active, UNIVERSE_COLORS["Optimal_Ethical"]['active'], LINE_STYLES['active'], 2, 'Optimal_Ethical - Active')
    plot_log(eth_wealth, UNIVERSE_COLORS["Optimal_Ethical"]['total_wealth'], LINE_STYLES['total_wealth'], 2.5, 'Optimal_Ethical - Total Wealth')
    
    # SPY
    if len(spy) > 0:
        try:
            if (spy > 0).all():
                ax.semilogy(spy.index, spy, color=SPY_COLOR, linestyle='-', linewidth=3, label='SPY (Benchmark)')
        except:
            pass
    
    ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=pd.Timestamp(injection_date), color='black', linestyle='--', alpha=0.5, label='Investment Date')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (£) - Log Scale')
    ax.set_title(f'Portfolio Performance Since Investment ({injection_date}) - Log Scale')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "final_performance_tracker_log.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✅ Saved: final_performance_tracker_log.png")
    
    print("\n" + "=" * 60)
    print("✅ PERFORMANCE TRACKER COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()