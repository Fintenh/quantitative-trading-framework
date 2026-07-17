"""
kelly_lookback_optimiser_ethical.py
------------------------------------
Phase 4: Find optimal Kelly lookback period for Optimal_Standard and Optimal_Ethical.
Tests different lookback periods for calculating Kelly Criterion cash cap.
EXTENDED SEARCH: Both universes now test up to 504 days (2 years).
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add Phase 3 to path for backtest engine
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import from existing modules
from config_optimised import *
from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics

# Import ethical universes (11-asset portfolios)
from ethical_universe import STANDARD_UNIVERSE, ETHICAL_UNIVERSE

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("PHASE 4: KELLY LOOKBACK OPTIMISATION (OPTIMAL PORTFOLIOS)")
print("=" * 70)
print(f"📁 Working directory: {script_dir}")
print(f"📁 Logs directory: {logs_dir}")
print(f"📁 Figures directory: {figures_dir}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Kelly lookback periods to test - EXTENDED for both universes (up to 504 days = 2 years)
KELLY_LOOKBACKS = [60, 75, 90, 105, 120, 126, 135, 150, 165, 180, 200, 252, 
                   280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 504]

# Base cash cap (20%)
BASE_CASH_CAP = 0.20

# Backtest period - set end date to today
BACKTEST_START = "2010-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")

print(f"\n📊 Configuration:")
print(f"  Kelly lookbacks to test: {KELLY_LOOKBACKS} (EXTENDED for both)")
print(f"  Base cash cap: {BASE_CASH_CAP*100:.0f}%")
print(f"  Backtest period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# KELLY CASH CAP FUNCTION
# ============================================================================

def calculate_kelly_cash_cap(returns_series, risk_free=0.045):
    """
    Calculate cash cap using Kelly Criterion.
    
    Returns:
        cash_cap: 0.20 if f* > 0, 1.00 if f* <= 0
        f_star: The Kelly fraction
    """
    if isinstance(returns_series, pd.DataFrame):
        if len(returns_series.columns) > 0:
            returns_series = returns_series.iloc[:, 0]
        else:
            return BASE_CASH_CAP, 0.0
    
    returns_series = returns_series.dropna()
    
    if len(returns_series) < 10:
        return BASE_CASH_CAP, 0.0
    
    mu = returns_series.mean() * 252
    sigma = returns_series.std() * np.sqrt(252)
    
    if sigma <= 0 or not np.isfinite(sigma):
        return BASE_CASH_CAP, 0.0
    
    f_star = (mu - risk_free) / (sigma ** 2)
    
    if not np.isfinite(f_star):
        return BASE_CASH_CAP, 0.0
    
    if f_star > 0:
        cash_cap = BASE_CASH_CAP
    else:
        cash_cap = 1.0
    
    return cash_cap, f_star


# ============================================================================
# MODIFIED BACKTEST ENGINE (WITH KELLY)
# ============================================================================

class BacktestEngineWithKelly(BacktestEngine):
    """
    Extended BacktestEngine that uses Kelly Criterion for cash allocation.
    """
    
    def __init__(self, kelly_lookback=252, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_weights = []
        self.kelly_f_stars = []
        self.kelly_cash_caps = []
    
    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using GARCH + Kelly.
        """
        if len(returns) >= self.kelly_lookback:
            kelly_returns = returns.iloc[-self.kelly_lookback:]
        else:
            kelly_returns = returns
        
        cash_cap, f_star = calculate_kelly_cash_cap(
            kelly_returns, 
            self.risk_free_rate
        )
        
        self.kelly_f_stars.append(f_star)
        self.kelly_cash_caps.append(cash_cap)
        
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            models, _, _ = fit_garch_for_assets(returns)
            vols = get_latest_volatility(models, returns)
            avg_vol = get_average_volatility(vols)
        except Exception as e:
            avg_vol = returns.std().mean() * np.sqrt(252)
        
        if avg_vol <= self.cash_min_volatility:
            garch_cash = 0.0
        elif avg_vol >= self.cash_max_volatility:
            garch_cash = cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            garch_cash = fraction * cash_cap
        
        self.kelly_weights.append(garch_cash)
        
        return garch_cash


# ============================================================================
# RUN KELLY BACKTESTS FOR A GIVEN UNIVERSE
# ============================================================================

def run_kelly_backtests_for_universe(tickers, universe_name, params, lookbacks):
    """
    Run Kelly lookback optimisation for a specific universe with given lookbacks.
    """
    print("\n" + "=" * 70)
    print(f"RUNNING KELLY OPTIMISATION: {universe_name} ({len(tickers)} assets)")
    print(f"Lookbacks to test: {len(lookbacks)} values")
    print("=" * 70)
    
    # Run baseline (NO Kelly)
    print("\n📊 Running Baseline (NO Kelly)...")
    engine_baseline = BacktestEngine(
        tickers=tickers,
        start_date=BACKTEST_START,
        end_date=BACKTEST_END,
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
        cash_max_allocation=BASE_CASH_CAP
    )
    
    results_baseline = engine_baseline.run()
    metrics_baseline = calculate_metrics(
        equity_curve=results_baseline['wealth_curve']['value'],
        initial_capital=INITIAL_CAPITAL,
        risk_free_rate=RISK_FREE_RATE
    )
    metrics_baseline['kelly_lookback'] = 'Baseline'
    metrics_baseline['total_return_pct'] = metrics_baseline['total_return'] * 100
    metrics_baseline['annual_return_pct'] = metrics_baseline['annualised_return'] * 100
    metrics_baseline['num_trades'] = len(results_baseline.get('trades', []))
    
    print(f"\n📊 Baseline Results ({universe_name}):")
    print(f"  Total Return: {metrics_baseline['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics_baseline['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics_baseline['max_drawdown']*100:.2f}%")
    
    # Run Kelly backtests
    print("\n" + "-" * 40)
    print("RUNNING KELLY BACKTESTS")
    print("-" * 40)
    
    all_results = []
    
    for kelly_lookback in lookbacks:
        print(f"\n🔄 Testing Kelly lookback: {kelly_lookback} days")
        
        try:
            engine = BacktestEngineWithKelly(
                tickers=tickers,
                start_date=BACKTEST_START,
                end_date=BACKTEST_END,
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
                cash_max_allocation=BASE_CASH_CAP,
                kelly_lookback=kelly_lookback
            )
            
            results = engine.run()
            
            metrics = calculate_metrics(
                equity_curve=results['wealth_curve']['value'],
                initial_capital=INITIAL_CAPITAL,
                risk_free_rate=RISK_FREE_RATE
            )
            
            metrics['kelly_lookback'] = kelly_lookback
            metrics['total_return_pct'] = metrics['total_return'] * 100
            metrics['annual_return_pct'] = metrics['annualised_return'] * 100
            metrics['num_trades'] = len(results.get('trades', []))
            
            all_results.append(metrics)
            
            print(f"  ✅ Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"     Sharpe: {metrics['sharpe_ratio']:.3f}")
            print(f"     Max DD: {metrics['max_drawdown']*100:.2f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Compile results
    if len(all_results) > 0:
        df_results = pd.DataFrame(all_results)
        
        baseline_row = pd.DataFrame([{
            'kelly_lookback': 'Baseline',
            'total_return_pct': metrics_baseline['total_return_pct'],
            'annual_return_pct': metrics_baseline['annual_return_pct'],
            'sharpe_ratio': metrics_baseline['sharpe_ratio'],
            'max_drawdown': metrics_baseline['max_drawdown'],
            'num_trades': metrics_baseline['num_trades']
        }])
        
        df_all = pd.concat([baseline_row, df_results], ignore_index=True)
        
        # Print results table
        print("\n" + "=" * 70)
        print(f"RESULTS SUMMARY: {universe_name}")
        print("=" * 70)
        print("\n| Lookback | Total Return | Annual Return | Sharpe | Max DD | Trades |")
        print("|----------|--------------|---------------|--------|--------|--------|")
        
        for _, row in df_all.iterrows():
            if row['kelly_lookback'] == 'Baseline':
                print(f"| Baseline | {row['total_return_pct']:>10.2f}% | {row['annual_return_pct']:>12.2f}% | {row['sharpe_ratio']:>6.3f} | {row['max_drawdown']*100:>6.2f}% | {row['num_trades']:>6} |")
            else:
                print(f"| {row['kelly_lookback']:>8} | {row['total_return_pct']:>10.2f}% | {row['annual_return_pct']:>12.2f}% | {row['sharpe_ratio']:>6.3f} | {row['max_drawdown']*100:>6.2f}% | {row['num_trades']:>6} |")
        
        # Best performers
        best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
        best_return = df_results.loc[df_results['total_return_pct'].idxmax()]
        best_dd = df_results.loc[df_results['max_drawdown'].idxmin()]
        
        print("\n" + "=" * 70)
        print(f"BEST PERFORMERS: {universe_name}")
        print("=" * 70)
        print(f"🏆 Best Sharpe:     {best_sharpe['kelly_lookback']} days (Sharpe={best_sharpe['sharpe_ratio']:.3f})")
        print(f"📈 Best Return:    {best_return['kelly_lookback']} days (Return={best_return['total_return_pct']:.2f}%)")
        print(f"🛡️ Best Drawdown:  {best_dd['kelly_lookback']} days (DD={best_dd['max_drawdown']*100:.2f}%)")
        
        best_kelly = best_return['kelly_lookback']
        
        # Save results
        safe_name = universe_name.replace(" ", "_")
        df_all.to_csv(os.path.join(logs_dir, f"kelly_lookback_results_{safe_name}.csv"), index=False)
        
        return {
            'universe_name': universe_name,
            'df_all': df_all,
            'df_results': df_results,
            'best_sharpe': best_sharpe,
            'best_return': best_return,
            'best_dd': best_dd,
            'best_kelly': best_kelly,
            'baseline': metrics_baseline
        }
    
    return None


# ============================================================================
# PLOT COMPARISON FUNCTION
# ============================================================================

def plot_kelly_comparison(standard_results, ethical_results):
    """
    Plot Kelly lookback comparison for Optimal_Standard and Optimal_Ethical side by side.
    """
    print("\n📊 Generating Kelly comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Define colours
    colors = {
        'Optimal_Standard': '#ff7f0e',
        'Optimal_Ethical': '#2ca02c'
    }
    
    # Plot each universe
    for idx, (results, name) in enumerate([
        (standard_results, 'Optimal_Standard'),
        (ethical_results, 'Optimal_Ethical')
    ]):
        if results is None:
            continue
            
        df = results['df_results']
        baseline = results['baseline']
        color = colors[name]
        row = idx // 2
        col = idx % 2
        
        # Return vs Lookback
        ax = axes[row, col]
        ax.plot(df['kelly_lookback'], df['total_return_pct'], 'o-', color=color, linewidth=2, markersize=6)
        ax.axhline(y=baseline['total_return_pct'], color='red', linestyle='--', linewidth=1.5, label='Baseline (No Kelly)')
        ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
        ax.set_ylabel('Total Return (%)', fontsize=9)
        ax.set_title(f'{name} - Return vs Kelly Lookback', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Annotate best
        best = df.loc[df['total_return_pct'].idxmax()]
        ax.annotate(f'Best: {best["kelly_lookback"]}d\n{best["total_return_pct"]:.1f}%',
                   xy=(best['kelly_lookback'], best['total_return_pct']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 3: Sharpe comparison
    ax = axes[1, 0]
    for results, name, color in [
        (standard_results, 'Optimal_Standard', '#ff7f0e'),
        (ethical_results, 'Optimal_Ethical', '#2ca02c')
    ]:
        if results is None:
            continue
        df = results['df_results']
        ax.plot(df['kelly_lookback'], df['sharpe_ratio'], 'o-', color=color, linewidth=2, markersize=6, label=name)
        ax.axhline(y=results['baseline']['sharpe_ratio'], color=color, linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
    ax.set_ylabel('Sharpe Ratio', fontsize=9)
    ax.set_title('Sharpe Ratio vs Kelly Lookback (Both Universes)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Drawdown comparison
    ax = axes[1, 1]
    for results, name, color in [
        (standard_results, 'Optimal_Standard', '#ff7f0e'),
        (ethical_results, 'Optimal_Ethical', '#2ca02c')
    ]:
        if results is None:
            continue
        df = results['df_results']
        ax.plot(df['kelly_lookback'], df['max_drawdown'] * 100, 'o-', color=color, linewidth=2, markersize=6, label=name)
        ax.axhline(y=results['baseline']['max_drawdown'] * 100, color=color, linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
    ax.set_ylabel('Max Drawdown (%)', fontsize=9)
    ax.set_title('Max Drawdown vs Kelly Lookback (Both Universes)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(figures_dir, "kelly_lookback_optimal_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved: kelly_lookback_optimal_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

print("\n" + "=" * 70)
print("STARTING KELLY LOOKBACK OPTIMISATION")
print("=" * 70)

# Define parameters for each universe (11-asset portfolios with optimised parameters)
UNIVERSE_PARAMS = {
    'Optimal_Standard': {
        'tickers': STANDARD_UNIVERSE,
        'lookbacks': KELLY_LOOKBACKS,
        'params': {
            'lookback_days': 275,
            'rebalance_min_days': 110,
            'rebalance_max_days': 155,
            'drift_threshold': 0.078,
            'take_profit_pct': 0.303,
        }
    },
    'Optimal_Ethical': {
        'tickers': ETHICAL_UNIVERSE,
        'lookbacks': KELLY_LOOKBACKS,
        'params': {
            'lookback_days': 300,
            'rebalance_min_days': 120,
            'rebalance_max_days': 135,
            'drift_threshold': 0.005,
            'take_profit_pct': 0.256,
        }
    }
}

# Run Kelly optimisation for Optimal_Standard (EXTENDED search)
standard_results = run_kelly_backtests_for_universe(
    UNIVERSE_PARAMS['Optimal_Standard']['tickers'],
    "Optimal_Standard",
    UNIVERSE_PARAMS['Optimal_Standard']['params'],
    UNIVERSE_PARAMS['Optimal_Standard']['lookbacks']
)

# Run Kelly optimisation for Optimal_Ethical (EXTENDED search)
ethical_results = run_kelly_backtests_for_universe(
    UNIVERSE_PARAMS['Optimal_Ethical']['tickers'],
    "Optimal_Ethical",
    UNIVERSE_PARAMS['Optimal_Ethical']['params'],
    UNIVERSE_PARAMS['Optimal_Ethical']['lookbacks']
)

# Generate comparison plots (shows both universes)
plot_kelly_comparison(standard_results, ethical_results)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY - OPTIMAL KELLY LOOKBACKS")
print("=" * 70)

print("\n| Universe | Optimal Lookback | Total Return | Sharpe | Max DD | Improvement vs Baseline |")
print("|----------|------------------|--------------|--------|--------|-------------------------|")

for results, name in [
    (standard_results, 'Optimal_Standard'),
    (ethical_results, 'Optimal_Ethical')
]:
    if results is not None:
        best = results['best_return']
        baseline = results['baseline']
        improvement = best['total_return_pct'] - baseline['total_return_pct']
        print(f"| {name:<14} | {best['kelly_lookback']:>16} | {best['total_return_pct']:>12.2f}% | {best['sharpe_ratio']:>6.3f} | {best['max_drawdown']*100:>6.2f}% | {improvement:>23.2f}% |")

# Save final summary
summary_data = []
for results, name in [(standard_results, 'Optimal_Standard'), (ethical_results, 'Optimal_Ethical')]:
    if results is not None:
        best = results['best_return']
        baseline = results['baseline']
        summary_data.append({
            'Universe': name,
            'Optimal_Lookback': best['kelly_lookback'],
            'Return': best['total_return_pct'],
            'Sharpe': best['sharpe_ratio'],
            'Max_Drawdown': best['max_drawdown'] * 100,
            'Baseline_Return': baseline['total_return_pct'],
            'Improvement': best['total_return_pct'] - baseline['total_return_pct'],
            'Trades': best['num_trades']
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(logs_dir, "kelly_lookback_optimal_summary.csv"), index=False)
print(f"\n✅ Summary saved: kelly_lookback_optimal_summary.csv")

print("\n" + "=" * 70)
print("🎉 KELLY LOOKBACK OPTIMISATION COMPLETE!")
print("=" * 70)
print("\n📊 Generated Files:")
print(f"   - figures/kelly_lookback_optimal_comparison.png")
print(f"   - logs/kelly_lookback_results_Optimal_Standard.csv")
print(f"   - logs/kelly_lookback_results_Optimal_Ethical.csv")
print(f"   - logs/kelly_lookback_optimal_summary.csv")
print("=" * 70)