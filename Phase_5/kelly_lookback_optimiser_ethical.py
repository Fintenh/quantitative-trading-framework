"""
kelly_lookback_optimiser_ethical.py
------------------------------------
Finds optimal Kelly lookback for Optimal_Standard and Optimal_Ethical.
FIXED: Actually uses the kelly_lookback window!
"""

import os
import sys
import warnings
import re
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics

from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO,
    OPTIMAL_ETHICAL_PORTFOLIO,
    OPTIMAL_STANDARD_PARAMS,
    OPTIMAL_ETHICAL_PARAMS,
    RISK_FREE_RATE,
    TRANSACTION_COST_PCT,
    CASH_INTEREST_RATE,
    CASH_MIN_VOLATILITY,
    CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION,
    START_DATE,
    HOLDOUT_END,
    LOG_DIR,
    FIGURES_DIR,
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("KELLY LOOKBACK OPTIMISATION")
print("=" * 70)

# Kelly lookbacks to test
KELLY_LOOKBACKS = [60, 75, 90, 105, 120, 126, 135, 150, 165, 180, 200, 252, 
                   280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 504]

BACKTEST_START = START_DATE
BACKTEST_END = HOLDOUT_END

print(f"\n📊 Testing {len(KELLY_LOOKBACKS)} lookback values")
print(f"📅 Period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# FIXED BACKTEST ENGINE WITH KELLY
# ============================================================================

class BacktestEngineWithKelly(BacktestEngine):
    def __init__(self, kelly_lookback=126, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_f_stars, self.kelly_cash_caps, self.kelly_cash_allocations = [], [], []
        self._debug_printed = False
    
    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using Kelly Criterion.
        FIXED: Directly calculates from the rolling window.
        """
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            from portfolio_optimiser import optimise_portfolios
            from data_fetcher import calculate_annualised_stats
            
            # ================================================================
            # FIX: Use kelly_lookback to select a rolling window
            # ================================================================
            # IMPORTANT: Use .iloc, NOT .loc! We want the LAST N days
            if len(returns) >= self.kelly_lookback:
                kelly_returns = returns.iloc[-self.kelly_lookback:].copy()
            else:
                kelly_returns = returns.copy()
            
            # DEBUG: Print once per engine
            if not self._debug_printed:
                print(f"\n  🔍 DEBUG: kelly_lookback={self.kelly_lookback}")
                print(f"     Full returns: {len(returns)} days")
                print(f"     Kelly window: {len(kelly_returns)} days")
                self._debug_printed = True
            
            # ================================================================
            # Calculate expected return and volatility from the WINDOW
            # ================================================================
            # Option 1: Use simple portfolio returns (equal weight for Kelly calculation)
            # This is what the original Kelly implementation does
            kelly_returns_series = kelly_returns.mean(axis=1)
            mu = kelly_returns_series.mean() * 252
            sigma = kelly_returns_series.std() * np.sqrt(252)
            
            # Full Kelly fraction
            if sigma > 0:
                f_star = (mu - self.risk_free_rate) / (sigma ** 2)
            else:
                f_star = 0.0
            
            # Kelly cap: 20% if f* > 0 (positive edge), 100% if f* <= 0 (negative edge)
            if f_star > 0:
                cash_cap = 0.20
            else:
                cash_cap = 1.00
            
            self.kelly_f_stars.append(f_star)
            self.kelly_cash_caps.append(cash_cap)
            
            # GARCH volatility for the full period
            models, _, _ = fit_garch_for_assets(returns)
            vols = get_latest_volatility(models, returns)
            avg_vol = get_average_volatility(vols)
            
        except Exception as e:
            print(f"  ⚠️ Error in _calc_cash_allocation: {e}")
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = 0.20
        
        # Cash allocation based on GARCH volatility (scaled by Kelly cap)
        if avg_vol <= self.cash_min_volatility:
            return 0.0
        elif avg_vol >= self.cash_max_volatility:
            return cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            return fraction * cash_cap


# ============================================================================
# UPDATE CONFIG - FIXED TO USE INTEGER
# ============================================================================

def update_config_kelly(universe_name, best_kelly):
    """Update KELLY_LOOKBACK in ethical_config.py using INTEGER."""
    config_path = os.path.join(script_dir, "ethical_config.py")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    if universe_name == "Optimal_Standard":
        param_name = "OPTIMAL_STANDARD_PARAMS"
    elif universe_name == "Optimal_Ethical":
        param_name = "OPTIMAL_ETHICAL_PARAMS"
    else:
        return
    
    # Convert to integer to avoid 60.0.0 bug
    best_kelly_int = int(best_kelly)
    
    pattern = rf"({param_name} = {{[^}}]*'kelly_lookback': )\d+"
    replacement = rf"\g<1>{best_kelly_int}"
    new_content = re.sub(pattern, replacement, content)
    
    with open(config_path, 'w') as f:
        f.write(new_content)
    
    print(f"   ✅ Updated {param_name} kelly_lookback to {best_kelly_int}")


# ============================================================================
# RUN KELLY OPTIMISATION
# ============================================================================

def run_kelly_optimisation(tickers, params, universe_name):
    print(f"\n{'='*70}")
    print(f"RUNNING: {universe_name} ({len(tickers)} assets)")
    print(f"{'='*70}")
    
    all_results = []
    
    for kelly_lookback in KELLY_LOOKBACKS:
        print(f"\n  Testing: {kelly_lookback} days...", end="", flush=True)
        
        try:
            engine = BacktestEngineWithKelly(
                tickers=tickers,
                start_date=BACKTEST_START,
                end_date=BACKTEST_END,
                initial_capital=100.0,
                lookback_days=params['lookback_days'],
                rebalance_min_days=params['rebalance_min_days'],
                rebalance_max_days=params['rebalance_max_days'],
                drift_threshold=params['drift_threshold'],
                take_profit_pct=0.0,  # No Rules
                risk_free_rate=RISK_FREE_RATE,
                transaction_cost_pct=TRANSACTION_COST_PCT,
                cash_interest_rate=CASH_INTEREST_RATE,
                cash_min_volatility=CASH_MIN_VOLATILITY,
                cash_max_volatility=CASH_MAX_VOLATILITY,
                cash_max_allocation=params['cash_max_allocation'],
                kelly_lookback=kelly_lookback
            )
            
            results = engine.run()
            equity = results['equity_curve']['value']
            trades = len(results.get('trades', []))
            
            metrics = calculate_metrics(
                equity_curve=equity,
                initial_capital=100.0,
                risk_free_rate=RISK_FREE_RATE
            )
            
            result_dict = {
                'kelly_lookback': kelly_lookback,
                'total_return_pct': metrics['total_return'] * 100,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': trades,
            }
            all_results.append(result_dict)
            
            print(f" Return: {result_dict['total_return_pct']:.1f}%, Sharpe: {result_dict['sharpe_ratio']:.3f}, Trades: {trades}")
            
        except Exception as e:
            print(f" ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) == 0:
        return None
    
    df_results = pd.DataFrame(all_results)
    
    print(f"\n  📊 Results Summary:")
    print(df_results[['kelly_lookback', 'total_return_pct', 'sharpe_ratio', 'num_trades']].to_string(index=False))
    
    best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    best_return = df_results.loc[df_results['total_return_pct'].idxmax()]
    
    print(f"\n  🏆 Best Sharpe: {best_sharpe['kelly_lookback']}d (Sharpe: {best_sharpe['sharpe_ratio']:.3f})")
    print(f"  🏆 Best Return: {best_return['kelly_lookback']}d ({best_return['total_return_pct']:.2f}%)")
    
    safe_name = universe_name.replace(" ", "_")
    df_results.to_csv(os.path.join(LOG_DIR, f"kelly_lookback_results_{safe_name}.csv"), index=False)
    
    return {
        'universe_name': universe_name,
        'df_results': df_results,
        'best_sharpe': best_sharpe,
        'best_return': best_return,
    }


# ============================================================================
# PLOT RESULTS
# ============================================================================

def plot_kelly_results(standard_results, ethical_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'Optimal_Standard': '#ff7f0e',
        'Optimal_Ethical': '#2ca02c'
    }
    
    for idx, (results, name) in enumerate([
        (standard_results, 'Optimal_Standard'),
        (ethical_results, 'Optimal_Ethical')
    ]):
        if results is None:
            continue
        
        df = results['df_results']
        color = colors[name]
        row = idx // 2
        col = idx % 2
        
        ax = axes[row, col]
        ax.plot(df['kelly_lookback'], df['total_return_pct'], 'o-', color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Kelly Lookback (days)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title(f'{name} - No Rules')
        ax.grid(True, alpha=0.3)
        
        best = df.loc[df['sharpe_ratio'].idxmax()]
        ax.axvline(x=best['kelly_lookback'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.annotate(f'Best: {best["kelly_lookback"]}d\nSharpe: {best["sharpe_ratio"]:.3f}',
                   xy=(best['kelly_lookback'], best['total_return_pct']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax = axes[1, 0]
    for results, name, color in [
        (standard_results, 'Optimal_Standard', '#ff7f0e'),
        (ethical_results, 'Optimal_Ethical', '#2ca02c')
    ]:
        if results is None:
            continue
        df = results['df_results']
        ax.plot(df['kelly_lookback'], df['sharpe_ratio'], 'o-', color=color, linewidth=2, markersize=6, label=name)
    
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for results, name, color in [
        (standard_results, 'Optimal_Standard', '#ff7f0e'),
        (ethical_results, 'Optimal_Ethical', '#2ca02c')
    ]:
        if results is None:
            continue
        df = results['df_results']
        ax.plot(df['kelly_lookback'], df['max_drawdown'] * 100, 'o-', color=color, linewidth=2, markersize=6, label=name)
    
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "kelly_lookback_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"✅ Saved: kelly_lookback_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL KELLY LOOKBACKS")
    print("=" * 70)
    print("💡 Using direct Kelly calculation from rolling window")
    print("=" * 70)
    
    universes = {
        'Optimal_Standard': {
            'tickers': OPTIMAL_STANDARD_PORTFOLIO,
            'params': OPTIMAL_STANDARD_PARAMS,
        },
        'Optimal_Ethical': {
            'tickers': OPTIMAL_ETHICAL_PORTFOLIO,
            'params': OPTIMAL_ETHICAL_PARAMS,
        }
    }
    
    results_dict = {}
    
    for name, config in universes.items():
        results = run_kelly_optimisation(
            config['tickers'],
            config['params'],
            name
        )
        results_dict[name] = results
    
    print("\n" + "=" * 70)
    print("UPDATING ETHICAL_CONFIG.PY")
    print("=" * 70)
    
    for name, results in results_dict.items():
        if results is not None:
            best_kelly = results['best_sharpe']['kelly_lookback']
            update_config_kelly(name, best_kelly)
    
    plot_kelly_results(
        results_dict.get('Optimal_Standard'),
        results_dict.get('Optimal_Ethical')
    )
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n| Universe | Optimal Lookback | Return | Sharpe | Max DD | Trades |")
    print("|----------|------------------|--------|--------|--------|--------|")
    
    for name, results in results_dict.items():
        if results is not None:
            best = results['best_sharpe']
            print(f"| {name:<14} | {best['kelly_lookback']:>16} | {best['total_return_pct']:>6.1f}% | {best['sharpe_ratio']:>6.3f} | {best['max_drawdown']*100:>6.2f}% | {best['num_trades']:>6} |")
    
    print("\n" + "=" * 70)
    print("🎉 COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()