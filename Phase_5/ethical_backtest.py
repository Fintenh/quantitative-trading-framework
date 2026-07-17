"""
ethical_backtest.py
-------------------
Runs backtests for Original_18, Optimal_Standard (11 assets), and Optimal_Ethical (11 assets).
Each universe has its own optimised parameters and Kelly lookback.
Plots FULL PERIOD (2010-2026) and HOLDOUT PERIOD (2020-2026).
Each universe has 3 lines: No Rules (with Kelly), Active (with Kelly), Total Wealth.
Colour families: Original=Blue, Standard=Orange, Ethical=Green.

PLOTS: 4 graphs total
- Full Period (Linear)
- Full Period (Log Scale) - shows true growth rate
- Holdout Period (Linear)
- Holdout Period (Log Scale) - shows true growth rate
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
from portfolio_optimiser import optimise_portfolios
from data_fetcher import calculate_annualised_stats
from ethical_universe import ORIGINAL_UNIVERSE, STANDARD_UNIVERSE, ETHICAL_UNIVERSE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# ============================================================================
# COLOURS
# ============================================================================

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

LINE_STYLES = {
    'no_rules': '-',
    'active': '--',
    'total_wealth': '-.',
}

# ============================================================================
# KELLY LOOKBACKS
# ============================================================================

KELLY_LOOKBACKS = {
    'Original_18': 126,
    'Optimal_Standard': 280,
    'Optimal_Ethical': 280,
}

# ============================================================================
# PARAMETERS
# ============================================================================

PARAMS = {
    'Original_18': {
        'lookback_days': 290, 'rebalance_min_days': 55, 'rebalance_max_days': 85,
        'drift_threshold': 0.035, 'take_profit_pct': 0.174,
        'cash_max_allocation': 0.20, 'kelly_lookback': KELLY_LOOKBACKS['Original_18']
    },
    'Optimal_Standard': {
        'lookback_days': 275, 'rebalance_min_days': 110, 'rebalance_max_days': 155,
        'drift_threshold': 0.078, 'take_profit_pct': 0.303,
        'cash_max_allocation': 0.20, 'kelly_lookback': KELLY_LOOKBACKS['Optimal_Standard']
    },
    'Optimal_Ethical': {
        'lookback_days': 300, 'rebalance_min_days': 120, 'rebalance_max_days': 135,
        'drift_threshold': 0.005, 'take_profit_pct': 0.256,
        'cash_max_allocation': 0.20, 'kelly_lookback': KELLY_LOOKBACKS['Optimal_Ethical']
    },
}

UNIVERSES = [
    ("Original_18", ORIGINAL_UNIVERSE, PARAMS['Original_18']),
    ("Optimal_Standard", STANDARD_UNIVERSE, PARAMS['Optimal_Standard']),
    ("Optimal_Ethical", ETHICAL_UNIVERSE, PARAMS['Optimal_Ethical']),
]

# ============================================================================
# CONFIGURATION
# ============================================================================

FULL_START, FULL_END = "2010-01-01", datetime.now().strftime("%Y-%m-%d")
HOLDOUT_START, HOLDOUT_END = "2020-01-01", FULL_END

INITIAL_CAPITAL = 100.0
RISK_FREE_RATE = 0.045
TRANSACTION_COST_PCT = 0.004
CASH_INTEREST_RATE = 0.0525
CASH_MIN_VOLATILITY = 0.25
CASH_MAX_VOLATILITY = 0.50

LOGS_DIR = os.path.join(script_dir, "logs")
FIGURES_DIR = os.path.join(script_dir, "figures")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Logs directory: {LOGS_DIR}")
print(f"📁 Figures directory: {FIGURES_DIR}")

# ============================================================================
# BACKTEST ENGINE WITH KELLY
# ============================================================================

class BacktestEngineWithKelly(BacktestEngine):
    def __init__(self, kelly_lookback=126, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_f_stars, self.kelly_cash_caps, self.kelly_cash_allocations = [], [], []
    
    def _calc_cash_allocation(self, returns):
        try:
            exp_ret, cov, _ = calculate_annualised_stats(returns)
            opt_results = optimise_portfolios(exp_ret, cov, self.risk_free_rate)
            weights = opt_results['msr_weights']
            mu = np.sum(exp_ret * weights)
            sigma = np.sqrt(weights.T @ cov @ weights)
            f_star = (mu - self.risk_free_rate) / (sigma ** 2) if sigma > 0 and np.isfinite(sigma) else 0.0
        except:
            mu = returns.mean(axis=1).mean() * 252 if isinstance(returns, pd.DataFrame) else returns.mean() * 252
            sigma = returns.std(axis=1).mean() * np.sqrt(252) if isinstance(returns, pd.DataFrame) else returns.std() * np.sqrt(252)
            f_star = (mu - self.risk_free_rate) / (sigma ** 2) if sigma > 0 and np.isfinite(sigma) else 0.0
        
        cash_cap = 0.20 if f_star > 0 else 1.00
        self.kelly_f_stars.append(f_star)
        self.kelly_cash_caps.append(cash_cap)
        
        try:
            models, _, _ = fit_garch_for_assets(returns)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns))
        except:
            avg_vol = returns.std().mean() * np.sqrt(252)
        
        if avg_vol <= self.cash_min_volatility:
            garch_cash = 0.0
        elif avg_vol >= self.cash_max_volatility:
            garch_cash = cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            garch_cash = fraction * cash_cap
        
        self.kelly_cash_allocations.append(garch_cash)
        return garch_cash

# ============================================================================
# RUN BACKTEST
# ============================================================================

def run_backtest(start_date, end_date, tickers, params, label):
    print(f"\n{'='*60}\nRUNNING: {label}\n{'='*60}")
    print(f"Assets: {len(tickers)} | Lookback: {params['lookback_days']} | "
          f"Rebalance: {params['rebalance_min_days']}-{params['rebalance_max_days']} | "
          f"Drift: {params['drift_threshold']*100:.1f}% | Take-Profit: {params['take_profit_pct']*100:.1f}% | "
          f"Kelly: {params['kelly_lookback']} days")
    
    spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
    spy = (spy_data / spy_data.iloc[0]) * INITIAL_CAPITAL
    
    # Active: Kelly + rules
    engine = BacktestEngineWithKelly(tickers=tickers, start_date=start_date, end_date=end_date,
        initial_capital=INITIAL_CAPITAL, lookback_days=params['lookback_days'],
        rebalance_min_days=params['rebalance_min_days'], rebalance_max_days=params['rebalance_max_days'],
        drift_threshold=params['drift_threshold'], take_profit_pct=params['take_profit_pct'],
        risk_free_rate=RISK_FREE_RATE, transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE, cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY, cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=params['kelly_lookback'])
    results = engine.run()
    total_wealth, active = results['wealth_curve']['value'], results['equity_curve']['value']
    
    # No Rules: Kelly + NO take-profit
    engine_nr = BacktestEngineWithKelly(tickers=tickers, start_date=start_date, end_date=end_date,
        initial_capital=INITIAL_CAPITAL, lookback_days=params['lookback_days'],
        rebalance_min_days=params['rebalance_min_days'], rebalance_max_days=params['rebalance_max_days'],
        drift_threshold=params['drift_threshold'], take_profit_pct=0.0,
        risk_free_rate=RISK_FREE_RATE, transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE, cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY, cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=params['kelly_lookback'])
    results_nr = engine_nr.run()
    no_rules = results_nr['equity_curve']['value']
    
    common = total_wealth.index.intersection(spy.index).intersection(no_rules.index).intersection(active.index)
    spy, total_wealth, active, no_rules = spy.loc[common], total_wealth.loc[common], active.loc[common], no_rules.loc[common]
    
    f_spy, f_nr, f_active, f_wealth = [float(x.values.flatten()[-1]) for x in [spy, no_rules, active, total_wealth]]
    
    print(f"\nPERFORMANCE SUMMARY: {label}")
    print(f"SPY: £{f_spy:.2f} | No Rules (with Kelly): £{f_nr:.2f} | Active: £{f_active:.2f} | Total Wealth: £{f_wealth:.2f}")
    
    spy_ret, nr_ret, act_ret = [(x / INITIAL_CAPITAL - 1) * 100 for x in [f_spy, f_nr, f_wealth]]
    print(f"SPY: {spy_ret:.1f}% | No Rules: {nr_ret:.1f}% | Active: {act_ret:.1f}% | Outperformance: {act_ret - spy_ret:.1f}%")
    
    metrics = calculate_metrics(equity_curve=total_wealth, initial_capital=INITIAL_CAPITAL,
                               risk_free_rate=RISK_FREE_RATE, trades=results.get('trades', []))
    metrics['num_trades'] = len(results.get('trades', []))
    
    metrics_nr = calculate_metrics(equity_curve=no_rules, initial_capital=INITIAL_CAPITAL,
                                   risk_free_rate=RISK_FREE_RATE, trades=results_nr.get('trades', []))
    metrics_nr['num_trades'] = len(results_nr.get('trades', []))
    
    return {'spy': spy, 'no_rules': no_rules, 'active': active, 'total_wealth': total_wealth,
            'common_dates': common, 'final_spy': f_spy, 'final_no_rules': f_nr, 'final_active': f_active,
            'final_wealth': f_wealth, 'spy_return': spy_ret, 'no_rules_return': nr_ret, 'active_return': act_ret,
            'metrics': metrics, 'metrics_no_rules': metrics_nr, 'results': results, 'results_no_rules': results_nr}

# ============================================================================
# PLOT FUNCTION - Creates both Linear and Log versions
# ============================================================================

def plot_performance(results_dict, title, filename_linear, filename_log, period_label):
    """Plots performance in BOTH Linear and Log scales."""
    
    # --- LINEAR PLOT ---
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    
    for name, res in results_dict.items():
        colors = UNIVERSE_COLORS[name]
        d = res[period_label]['common_dates']
        ax1.plot(d, res[period_label]['no_rules'].loc[d], color=colors['no_rules'], linestyle='-', lw=2,
                label=f'{name} - No Rules')
        ax1.plot(d, res[period_label]['active'].loc[d], color=colors['active'], linestyle='--', lw=2,
                label=f'{name} - Active')
        ax1.plot(d, res[period_label]['total_wealth'].loc[d], color=colors['total_wealth'], linestyle='-.', lw=2.5,
                label=f'{name} - Total Wealth')
    
    spy_d = results_dict["Original_18"][period_label]['common_dates']
    ax1.plot(spy_d, results_dict["Original_18"][period_label]['spy'].loc[spy_d],
            color=SPY_COLOR, linestyle='-', lw=3, label='SPY (Benchmark)')
    ax1.axhline(INITIAL_CAPITAL, color='gray', ls='-', alpha=0.3)
    ax1.set_title(title + '\n(Linear Scale)', fontsize=16, weight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Value (£)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename_linear), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Saved: {filename_linear}")
    
    # --- LOG SCALE PLOT ---
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    for name, res in results_dict.items():
        colors = UNIVERSE_COLORS[name]
        d = res[period_label]['common_dates']
        # Use semilogy for log scale on y-axis
        ax2.semilogy(d, res[period_label]['no_rules'].loc[d], color=colors['no_rules'], linestyle='-', lw=2,
                    label=f'{name} - No Rules')
        ax2.semilogy(d, res[period_label]['active'].loc[d], color=colors['active'], linestyle='--', lw=2,
                    label=f'{name} - Active')
        ax2.semilogy(d, res[period_label]['total_wealth'].loc[d], color=colors['total_wealth'], linestyle='-.', lw=2.5,
                    label=f'{name} - Total Wealth')
    
    spy_d = results_dict["Original_18"][period_label]['common_dates']
    ax2.semilogy(spy_d, results_dict["Original_18"][period_label]['spy'].loc[spy_d],
                color=SPY_COLOR, linestyle='-', lw=3, label='SPY (Benchmark)')
    ax2.axhline(INITIAL_CAPITAL, color='gray', ls='-', alpha=0.3)
    ax2.set_title(title + '\n(Log Scale - Shows True Growth Rate)', fontsize=16, weight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Value (£) - Log Scale', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename_log), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Saved: {filename_log}")

# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("OPTIMAL PORTFOLIO BACKTEST (KELLY FOR BOTH NO RULES AND ACTIVE)")
print("=" * 70)
print(f"📁 Original_18: 18 assets (Kelly: {KELLY_LOOKBACKS['Original_18']} days)")
print(f"📁 Optimal_Standard: 11 assets (Kelly: {KELLY_LOOKBACKS['Optimal_Standard']} days)")
print(f"📁 Optimal_Ethical: 11 assets (Kelly: {KELLY_LOOKBACKS['Optimal_Ethical']} days)")
print(f"📁 Periods: {FULL_START} to {FULL_END} | Holdout: {HOLDOUT_START} to {HOLDOUT_END}")
print(f"📁 Plotting: 4 graphs total (Linear + Log for each period)")
print("=" * 70)

all_results = {}
for name, tickers, params in UNIVERSES:
    all_results[name] = {
        'full': run_backtest(FULL_START, FULL_END, tickers, params, f"{name} - Full"),
        'holdout': run_backtest(HOLDOUT_START, HOLDOUT_END, tickers, params, f"{name} - Holdout")
    }

# ============================================================================
# TABLES
# ============================================================================

for period, label in [('full', 'FULL PERIOD (2010-2026)'), ('holdout', 'HOLDOUT PERIOD (2020-2026)')]:
    print(f"\n{'='*70}\nCOMPARISON SUMMARY - {label}\n{'='*70}")
    print("| Universe | Assets | Kelly | No Rules Return | Active Return | Sharpe | Max DD | Trades |")
    print("|----------|--------|-------|-----------------|---------------|--------|--------|--------|")
    for name in all_results.keys():
        r = all_results[name][period]
        p = PARAMS[name]
        print(f"| {name:<14} | {len(UNIVERSES[0][1] if name=='Original_18' else UNIVERSES[1][1] if name=='Optimal_Standard' else UNIVERSES[2][1]):>6} | {p['kelly_lookback']:>5} | "
              f"{r['no_rules_return']:>15.1f}% | {r['active_return']:>13.1f}% | {r['metrics']['sharpe_ratio']:>6.3f} | "
              f"{r['metrics']['max_drawdown']*100:>6.2f}% | {r['metrics']['num_trades']:>6} |")

# ============================================================================
# PLOTS - 4 GRAPHS TOTAL
# ============================================================================

# 1. Full Period - Linear
plot_performance(all_results, 
    'Optimal Standard vs Optimal Ethical vs Original Portfolio Performance\nFull Period (2010-2026)',
    'optimal_portfolio_comparison_full_linear.png',
    'optimal_portfolio_comparison_full_log.png',
    'full')

# 2. Holdout Period - Linear & Log
plot_performance(all_results,
    'Optimal Standard vs Optimal Ethical vs Original Portfolio Performance\nHoldout Period (2020-2026)',
    'optimal_portfolio_comparison_holdout_linear.png',
    'optimal_portfolio_comparison_holdout_log.png',
    'holdout')

# ============================================================================
# SAVE CSVS
# ============================================================================

print("\n📊 Saving CSVs...")

def save_csv(period, label):
    common = all_results["Original_18"][period]['common_dates']
    for name in all_results.keys():
        common = common.intersection(all_results[name][period]['common_dates'])
    
    df = pd.DataFrame({'date': common})
    for name in all_results.keys():
        r = all_results[name][period]
        prefix = name.lower().replace('_', '_')
        df[f'{prefix}_no_rules'] = r['no_rules'].loc[common].values.flatten()
        df[f'{prefix}_active'] = r['active'].loc[common].values.flatten()
        df[f'{prefix}_total_wealth'] = r['total_wealth'].loc[common].values.flatten()
    df['spy_benchmark'] = all_results["Original_18"][period]['spy'].loc[common].values.flatten()
    df.to_csv(os.path.join(LOGS_DIR, f"optimal_portfolio_backtest_results_{label}_kelly.csv"), index=False)
    print(f"✅ optimal_portfolio_backtest_results_{label}_kelly.csv")

save_csv('full', 'full')
save_csv('holdout', 'holdout')

print("\n" + "=" * 70)
print("✅ OPTIMAL PORTFOLIO BACKTEST COMPLETE (4 PLOTS - LINEAR + LOG)!")
print("=" * 70)
print("\n📊 Generated Files:")
print(f"   - figures/optimal_portfolio_comparison_full_linear.png")
print(f"   - figures/optimal_portfolio_comparison_full_log.png")
print(f"   - figures/optimal_portfolio_comparison_holdout_linear.png")
print(f"   - figures/optimal_portfolio_comparison_holdout_log.png")
print(f"   - logs/optimal_portfolio_backtest_results_full_kelly.csv")
print(f"   - logs/optimal_portfolio_backtest_results_holdout_kelly.csv")
print("=" * 70)