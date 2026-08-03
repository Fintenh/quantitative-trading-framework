"""
run_backtest.py - Run backtest and save results.
Generates 5 graphs and CSV outputs.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from config_optimised import *
from backtest_engine import BacktestEngine
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from datetime import datetime

# Use the MacOSX backend so figures both display on screen and save to disk
import matplotlib
matplotlib.use('MacOSX')

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Logs directory: {logs_dir}")
print(f"Figures directory: {figures_dir}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FULL_START = "2010-01-01"
FULL_END = datetime.now().strftime("%Y-%m-%d")
HOLDOUT_START = "2020-01-01"
HOLDOUT_END = datetime.now().strftime("%Y-%m-%d")

print("\n" + "=" * 60)
print("CASH PARAMETERS (from config_optimised.py)")
print("=" * 60)
print(f"CASH_MIN_VOLATILITY: {CASH_MIN_VOLATILITY}")
print(f"CASH_MAX_VOLATILITY: {CASH_MAX_VOLATILITY}")
print(f"CASH_MAX_ALLOCATION: {CASH_MAX_ALLOCATION}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

def run_backtest(start_date, end_date, period_name):
    """Run backtest for a specific period."""
    print("\n" + "=" * 60)
    print(f"RUNNING BACKTEST: {period_name}")
    print("=" * 60)
    print(f"Lookback: {LOOKBACK_DAYS} days")
    print(f"Rebalance: {REBALANCE_MIN_DAYS}-{REBALANCE_MAX_DAYS} days")
    print(f"Drift: {DRIFT_THRESHOLD*100:.1f}%")
    print(f"Take-Profit: {RELATIVE_TAKE_PROFIT_PCT*100:.0f}% vs SPY")
    print(f"Cash Interest: {CASH_INTEREST_RATE*100:.2f}% AER")
    print("=" * 60)

    # Active strategy (with take-profit)
    print("\nRunning backtest with take-profit...")
    engine = BacktestEngine(
        tickers=TICKERS,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=LOOKBACK_DAYS,
        rebalance_min_days=REBALANCE_MIN_DAYS,
        rebalance_max_days=REBALANCE_MAX_DAYS,
        drift_threshold=DRIFT_THRESHOLD,
        take_profit_pct=RELATIVE_TAKE_PROFIT_PCT,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=CASH_MAX_ALLOCATION
    )
    results = engine.run()
    active = results['equity_curve']['value']
    total_wealth = results['wealth_curve']['value']

    # No rules (without take-profit)
    print("\nRunning backtest without take-profit...")
    engine_no_rules = BacktestEngine(
        tickers=TICKERS,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=LOOKBACK_DAYS,
        rebalance_min_days=REBALANCE_MIN_DAYS,
        rebalance_max_days=REBALANCE_MAX_DAYS,
        drift_threshold=DRIFT_THRESHOLD,
        take_profit_pct=0.0,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=CASH_MAX_ALLOCATION
    )
    results_no_rules = engine_no_rules.run()
    no_rules = results_no_rules['equity_curve']['value']

    # SPY benchmark
    print("\nDownloading SPY benchmark...")
    spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
    spy = (spy_data / spy_data.iloc[0]) * INITIAL_CAPITAL

    # Align dates
    common_dates = active.index.intersection(spy.index)
    common_dates = common_dates.intersection(no_rules.index)
    spy = spy.loc[common_dates]
    active = active.loc[common_dates]
    total_wealth = total_wealth.loc[common_dates]
    no_rules = no_rules.loc[common_dates]

    # Summary
    final_spy = float(spy.values.flatten()[-1])
    final_active = float(active.values.flatten()[-1])
    final_wealth = float(total_wealth.values.flatten()[-1])
    final_no_rules = float(no_rules.values.flatten()[-1])

    print("\n" + "=" * 60)
    print(f"PERFORMANCE SUMMARY: {period_name}")
    print("=" * 60)
    print(f"SPY (Buy & Hold):                    £{final_spy:.2f}")
    print(f"No Rules (No Take-Profit):          £{final_no_rules:.2f}")
    print(f"Active Portfolio (With Take-Profit): £{final_active:.2f}")
    print(f"Total Wealth (Active + Profits):     £{final_wealth:.2f}")
    print(f"Outperformance vs SPY:               £{final_wealth - final_spy:.2f}")

    spy_return = (final_spy / INITIAL_CAPITAL - 1) * 100
    active_return = (final_wealth / INITIAL_CAPITAL - 1) * 100
    print(f"\nSPY Return: {spy_return:.2f}%")
    print(f"Active Strategy Return: {active_return:.2f}%")
    print(f"Outperformance: {active_return - spy_return:.2f}%")

    return {
        'spy': spy,
        'active': active,
        'total_wealth': total_wealth,
        'no_rules': no_rules,
        'common_dates': common_dates,
        'final_spy': final_spy,
        'final_active': final_active,
        'final_wealth': final_wealth,
        'final_no_rules': final_no_rules,
        'spy_return': spy_return,
        'active_return': active_return,
        'period_name': period_name,
        'results': results
    }

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_performance_comparison(results, period_name, filename):
    """Plot 4-line performance comparison."""
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(results['spy'].index, results['spy'], 'k--', lw=2, label='SPY (Buy & Hold)')
    ax.plot(results['no_rules'].index, results['no_rules'], 'r-', lw=2, label='No Rules (No Take-Profit)')
    ax.plot(results['active'].index, results['active'], 'b-', lw=2, label='Active Portfolio (With Take-Profit)')
    ax.plot(results['total_wealth'].index, results['total_wealth'], 'g-', lw=3, label='Total Wealth (Active + Profits)')

    ax.axhline(y=INITIAL_CAPITAL, color='gray', ls='-', alpha=0.3)
    ax.set_title(f'Strategy Performance Comparison\n{period_name}', fontsize=16, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value (£)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Summary text box
    text = f'SPY: £{results["final_spy"]:.0f}  |  No Rules: £{results["final_no_rules"]:.0f}  |  Active: £{results["final_active"]:.0f}  |  Total Wealth: £{results["final_wealth"]:.0f}'
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_drawdown(results, period_name, filename):
    """Plot drawdown for the no-rules portfolio."""
    no_rules_returns = results['no_rules'].pct_change().dropna()
    cumulative = (1 + no_rules_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, 'r-', lw=1.5, label='No Rules Drawdown')
    ax.axhline(y=0, color='black', lw=0.5)

    for level in [-10, -20, -30, -40, -50]:
        ax.axhline(y=level, color='gray', ls='--', alpha=0.3)

    ax.set_title(f'Drawdown: No Rules Portfolio\n{period_name}', fontsize=14, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.annotate(f'Max Drawdown: {max_dd:.1f}%',
                xy=(max_dd_date, max_dd),
                xytext=(max_dd_date, max_dd - 5),
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_monthly_returns(results, period_name, filename):
    """Plot monthly returns bar chart."""
    monthly = results['total_wealth'].resample('ME').last()
    monthly_returns = monthly.pct_change().dropna() * 100

    if len(monthly_returns) == 0:
        print("No monthly returns data available")
        return pd.Series(), 0, 0, 0

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ['green' if x >= 0 else 'red' for x in monthly_returns.values]
    ax.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    ax.axhline(y=0, color='black', lw=0.8)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Monthly Return (%)', fontsize=12)
    ax.set_title(f'Monthly Returns (Active Strategy)\n{period_name}', fontsize=16, weight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.grid(True, alpha=0.3, axis='y')

    # Statistics
    total_return = (results['total_wealth'].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    num_months = len(monthly_returns)
    geo_mean = ((results['total_wealth'].iloc[-1] / INITIAL_CAPITAL) ** (1 / num_months) - 1) * 100
    positive = (monthly_returns > 0).sum()
    negative = (monthly_returns < 0).sum()
    win_rate = positive / (positive + negative) * 100 if (positive + negative) > 0 else 0

    summary = (
        f"Total Return: {total_return:.1f}% | "
        f"Geometric Mean: {geo_mean:.2f}%/mo | "
        f"Win Rate: {win_rate:.1f}% | "
        f"Best: {monthly_returns.max():.1f}% | "
        f"Worst: {monthly_returns.min():.1f}%"
    )
    ax.text(0.5, 0.98, summary, transform=ax.transAxes,
            fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return monthly_returns, total_return, geo_mean, win_rate


def plot_monthly_distribution(monthly_returns, geo_mean, period_name, filename):
    """Plot monthly returns distribution histogram."""
    if len(monthly_returns) == 0:
        print("No monthly returns data for distribution plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(monthly_returns, bins=25, color='blue', alpha=0.7, edgecolor='black', lw=0.5)
    ax.axvline(x=0, color='red', ls='--', lw=1.5, label='Zero Return')
    ax.axvline(x=geo_mean, color='green', lw=2, label=f'Geometric Mean: {geo_mean:.2f}%/mo')

    ax.set_xlabel('Monthly Return (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Monthly Returns\n{period_name}', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Run backtests
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("RUNNING FULL PERIOD BACKTEST (2010-Today)")
print("=" * 60)
full = run_backtest(FULL_START, FULL_END, f"Full Period (2010 to {datetime.now().strftime('%Y-%m-%d')})")

print("\n" + "=" * 60)
print("RUNNING HOLDOUT PERIOD BACKTEST (2020-Today)")
print("=" * 60)
holdout = run_backtest(HOLDOUT_START, HOLDOUT_END, f"Holdout Period (2020 to {datetime.now().strftime('%Y-%m-%d')})")

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------

print("\nGenerating plots...")

# Graph 1: full period performance
plot_performance_comparison(full, full['period_name'], "backtest_results_full.png")
print("Graph 1: backtest_results_full.png")

# Graph 2: holdout period performance
plot_performance_comparison(holdout, holdout['period_name'], "backtest_results_holdout.png")
print("Graph 2: backtest_results_holdout.png")

# Graph 3: drawdown (holdout)
plot_drawdown(holdout, holdout['period_name'], "drawdown_no_rules_holdout.png")
print("Graph 3: drawdown_no_rules_holdout.png")

# Graph 4: monthly returns bar chart (holdout)
monthly_returns, total_return, geo_mean, win_rate = plot_monthly_returns(
    holdout, holdout['period_name'], "monthly_returns_holdout.png"
)
print("Graph 4: monthly_returns_holdout.png")

# Graph 5: monthly returns distribution (holdout)
if len(monthly_returns) > 0:
    plot_monthly_distribution(monthly_returns, geo_mean, holdout['period_name'], "monthly_returns_distribution_holdout.png")
    print("Graph 5: monthly_returns_distribution_holdout.png")
else:
    print("Graph 5: No monthly returns data - skipping monthly distribution plot")

# ---------------------------------------------------------------------------
# Monthly returns summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("MONTHLY RETURNS SUMMARY")
print("=" * 60)
print(f"\nNumber of months: {len(monthly_returns)}")
print(f"Total Return: {total_return:.2f}%")
print(f"Geometric Mean: {geo_mean:.2f}%/mo")
print(f"Arithmetic Mean: {monthly_returns.mean():.2f}%/mo")
print(f"Median: {monthly_returns.median():.2f}%/mo")
print(f"Std Dev: {monthly_returns.std():.2f}%/mo")
print(f"Best: {monthly_returns.max():.2f}%")
print(f"Worst: {monthly_returns.min():.2f}%")
print(f"Positive months: {(monthly_returns > 0).sum()}")
print(f"Negative months: {(monthly_returns < 0).sum()}")
print(f"Win rate: {win_rate:.1f}%")

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------

print("\nSaving CSVs...")

def save_results(results, filename):
    df = pd.DataFrame({
        'date': results['common_dates'],
        'spy': results['spy'].values.flatten(),
        'no_rules': results['no_rules'].values.flatten(),
        'active': results['active'].values.flatten(),
        'total_wealth': results['total_wealth'].values.flatten()
    })
    df.to_csv(os.path.join(logs_dir, filename), index=False)
    print(f"{filename}")

save_results(full, "backtest_results_full.csv")
save_results(holdout, "backtest_results_holdout.csv")

# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)
print("\nGenerated Files:")
print(f"   - figures/backtest_results_full.png")
print(f"   - figures/backtest_results_holdout.png")
print(f"   - figures/drawdown_no_rules_holdout.png")
print(f"   - figures/monthly_returns_holdout.png")
print(f"   - figures/monthly_returns_distribution_holdout.png")
print(f"   - logs/backtest_results_full.csv")
print(f"   - logs/backtest_results_holdout.csv")
print("=" * 60)