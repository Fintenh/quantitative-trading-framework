"""
kelly_lookback_optimiser.py
----------------------------
Tests different Kelly lookback periods using the Phase 3 backtest.
Kelly f* is calculated using portfolio returns:
f* = (portfolio_return - risk_free_return) / volatility²
Sign changes trigger early rebalances.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics
from config_optimised import *
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("KELLY LOOKBACK OPTIMISATION")
print("=" * 70)
print("f* = (portfolio_return - risk_free_return) / volatility²")
print("Using equal-weight portfolio returns for Kelly calculation")
print("Sign changes trigger early rebalances")

# ============================================================================
# CONFIGURATION
# ============================================================================

KELLY_LOOKBACKS = [10, 30, 60, 90, 120, 135, 150, 180, 210, 252, 300, 360]
BACKTEST_START = "2016-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")

print(f"\nTesting {len(KELLY_LOOKBACKS)} lookback periods")
print(f"Period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# BACKTEST ENGINE WITH CORRECTED KELLY
# ============================================================================

class BacktestEngineWithKelly(BacktestEngine):
    """
    Phase 3 backtest engine with Kelly.
    Uses equal-weight portfolio returns for Kelly calculation.
    """
    
    def __init__(self, kelly_lookback=126, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.prev_f_star = None
        self.kelly_f_stars = []
        self.kelly_sign_changes = 0
        self._force_rebalance = False
        self._debug_count = 0
    
    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using Kelly.
        Uses equal-weight portfolio returns (simple and stable).
        """
        try:
            # 1. GARCH volatility (same as Phase 3)
            models, _, _ = fit_garch_for_assets(returns)
            vols = get_latest_volatility(models, returns)
            avg_vol = get_average_volatility(vols)
            
            # 2. Get the lookback window
            if len(returns) >= self.kelly_lookback:
                kelly_returns = returns.iloc[-self.kelly_lookback:].copy()
            else:
                kelly_returns = returns.copy()
            
            days = len(kelly_returns)
            
            # 3. Calculate equal-weight portfolio returns
            portfolio_returns = kelly_returns.mean(axis=1)
            
            # 4. Calculate ACTUAL portfolio return over the period (compounded)
            portfolio_return = (1 + portfolio_returns).prod() - 1
            
            # 5. Risk-free return over the same period
            risk_free_return = (1 + self.risk_free_rate) ** (days / 252) - 1
            
            # 6. Volatility over the period
            volatility = portfolio_returns.std()
            
            # 7. Excess return
            excess_return = portfolio_return - risk_free_return
            
            # 8. Kelly fraction
            if volatility > 0 and np.isfinite(volatility) and volatility < 1.0:
                f_star = excess_return / (volatility ** 2)
            else:
                f_star = 0.0
            
            # 9. Check for sign change (triggers rebalance)
            current_sign = 1 if f_star > 0 else -1
            if self.prev_f_star is not None:
                prev_sign = 1 if self.prev_f_star > 0 else -1
                if current_sign != prev_sign:
                    self.kelly_sign_changes += 1
                    self._force_rebalance = True
            
            self.prev_f_star = f_star
            
            # 10. Kelly cash cap
            if f_star > 0:
                cash_cap = 0.20  # Positive edge → 20% cash
            else:
                cash_cap = 1.00  # Negative edge → 100% cash
            
            # Store for debugging
            self.kelly_f_stars.append(f_star)
            
            # Debug output for first 10 iterations
            if self._debug_count < 10:
                self._debug_count += 1
                print(f"\n    Kelly Debug #{self._debug_count} (lookback={self.kelly_lookback}):")
                print(f"      Days: {days}")
                print(f"      Portfolio Return: {portfolio_return:.6f} ({portfolio_return*100:.4f}%)")
                print(f"      Risk-Free Return: {risk_free_return:.6f} ({risk_free_return*100:.4f}%)")
                print(f"      Excess Return: {excess_return:.6f} ({excess_return*100:.4f}%)")
                print(f"      Volatility: {volatility:.6f}")
                print(f"      Volatility²: {volatility**2:.8f}")
                print(f"      f*: {f_star:.4f}")
                print(f"      Sign: {'+' if f_star > 0 else '-'}")
                print(f"      Cash Cap: {cash_cap:.0%}")
            
        except Exception as e:
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = self.cash_max_allocation
            if self._debug_count < 10:
                print(f"    Kelly Error: {e}")
                self._debug_count += 1
        
        # 11. Cash allocation based on GARCH volatility
        if avg_vol <= self.cash_min_volatility:
            return 0.0
        elif avg_vol >= self.cash_max_volatility:
            return cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            return fraction * cash_cap
    
    def should_force_rebalance(self):
        """Check if Kelly sign change should force rebalance."""
        if self._force_rebalance:
            self._force_rebalance = False
            return True
        return False


# ============================================================================
# RUN BASELINE
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING BASELINE (No Rules, Fixed 20% Cash Cap)")
print("=" * 70)

baseline_engine = BacktestEngine(
    tickers=TICKERS,
    start_date=BACKTEST_START,
    end_date=BACKTEST_END,
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

baseline_results = baseline_engine.run()
baseline_metrics = calculate_metrics(
    equity_curve=baseline_results['wealth_curve']['value'],
    initial_capital=INITIAL_CAPITAL,
    risk_free_rate=RISK_FREE_RATE
)

print(f"\nBaseline Results:")
print(f"  Total Return: {baseline_metrics['total_return']*100:.2f}%")
print(f"  Sharpe Ratio: {baseline_metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {baseline_metrics['max_drawdown']*100:.2f}%")
print(f"  Trades: {len(baseline_results.get('trades', []))}")

# ============================================================================
# RUN KELLY BACKTESTS
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING KELLY BACKTESTS")
print("=" * 70)

all_results = []

for kelly_lookback in KELLY_LOOKBACKS:
    print(f"\nTesting lookback: {kelly_lookback} days...", end="", flush=True)
    
    try:
        engine = BacktestEngineWithKelly(
            tickers=TICKERS,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
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
            cash_max_allocation=CASH_MAX_ALLOCATION,
            kelly_lookback=kelly_lookback
        )
        
        results = engine.run()
        
        metrics = calculate_metrics(
            equity_curve=results['wealth_curve']['value'],
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=RISK_FREE_RATE
        )
        
        # Calculate statistics
        f_stars = np.array(engine.kelly_f_stars) if engine.kelly_f_stars else np.array([0])
        f_star_mean = np.mean(f_stars)
        f_star_positive = np.sum(f_stars > 0)
        f_star_negative = np.sum(f_stars <= 0)
        
        result = {
            'lookback': kelly_lookback,
            'total_return': metrics['total_return'] * 100,
            'sharpe': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'] * 100,
            'trades': len(results.get('trades', [])),
            'f_star_mean': f_star_mean,
            'f_star_positive': f_star_positive,
            'f_star_negative': f_star_negative,
            'sign_changes': engine.kelly_sign_changes
        }
        all_results.append(result)
        
        print(f" Return: {result['total_return']:.1f}%, Sharpe: {result['sharpe']:.3f}")
        print(f"    f* mean: {result['f_star_mean']:.4f}")
        print(f"    f* > 0: {result['f_star_positive']}, f* <= 0: {result['f_star_negative']}")
        print(f"    Sign Changes: {result['sign_changes']}")
        
    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\n| Lookback | Return | Sharpe | Max DD | Trades | Sign Chg | f* > 0 | f* <= 0 | f* Mean |")
print(f"|----------|--------|--------|--------|--------|----------|--------|---------|---------|")
print(f"| Baseline | {baseline_metrics['total_return']*100:>6.2f}% | {baseline_metrics['sharpe_ratio']:>6.3f} | {baseline_metrics['max_drawdown']*100:>6.2f}% | {len(baseline_results.get('trades', [])):>6} | {'-':>8} | {'-':>6} | {'-':>7} | {'-':>7} |")

for r in all_results:
    print(f"| {r['lookback']:>8} | {r['total_return']:>6.2f}% | {r['sharpe']:>6.3f} | {r['max_drawdown']:>6.2f}% | {r['trades']:>6} | {r['sign_changes']:>8} | {r['f_star_positive']:>6} | {r['f_star_negative']:>7} | {r['f_star_mean']:>7.2f} |")

# ============================================================================
# FIND OPTIMAL
# ============================================================================

if all_results:
    # Find best by Sharpe ratio
    best = max(all_results, key=lambda x: x['sharpe'])
    
    print(f"\n" + "=" * 70)
    print(f"OPTIMAL KELLY LOOKBACK: {best['lookback']} days")
    print("=" * 70)
    print(f"  Total Return: {best['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {best['sharpe']:.3f}")
    print(f"  Max Drawdown: {best['max_drawdown']:.2f}%")
    print(f"  Trades: {best['trades']}")
    print(f"  Sign Changes: {best['sign_changes']}")
    print(f"  f* mean: {best['f_star_mean']:.4f}")
    print(f"  f* > 0: {best['f_star_positive']}, f* <= 0: {best['f_star_negative']}")
    
    # Compare to baseline
    improvement = best['total_return'] - baseline_metrics['total_return'] * 100
    sharpe_improvement = best['sharpe'] - baseline_metrics['sharpe_ratio']
    print(f"\nImprovement vs Baseline:")
    print(f"  Return: +{improvement:.2f}%")
    print(f"  Sharpe: +{sharpe_improvement:.3f}")
    
    # Save config
    config = f'''"""
kelly_config.py - Kelly parameters from Phase 4 optimisation.
"""

KELLY_LOOKBACK = {best['lookback']}
KELLY_BASE_CAP = 0.20
KELLY_MAX_CAP = 1.00

# Performance with this lookback:
#   Total Return: {best['total_return']:.2f}%
#   Sharpe Ratio: {best['sharpe']:.3f}
#   Max Drawdown: {best['max_drawdown']:.2f}%
#   Trades: {best['trades']}
#   Sign Changes: {best['sign_changes']}
'''
    
    with open(os.path.join(script_dir, "kelly_config.py"), "w") as f:
        f.write(config)
    
    print(f"\nConfig saved to: {script_dir}/kelly_config.py")
    
    # ============================================================================
    # PLOT
    # ============================================================================
    
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Return vs Lookback
    ax = axes[0, 0]
    ax.plot([r['lookback'] for r in all_results], [r['total_return'] for r in all_results], 'o-', color='blue', linewidth=2, markersize=6)
    ax.axhline(y=baseline_metrics['total_return']*100, color='red', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Total Return vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sharpe vs Lookback
    ax = axes[0, 1]
    ax.plot([r['lookback'] for r in all_results], [r['sharpe'] for r in all_results], 'o-', color='green', linewidth=2, markersize=6)
    ax.axhline(y=baseline_metrics['sharpe_ratio'], color='red', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sign Changes vs Lookback
    ax = axes[0, 2]
    ax.bar([r['lookback'] for r in all_results], [r['sign_changes'] for r in all_results], color='purple', alpha=0.7)
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Number of Sign Changes')
    ax.set_title('Kelly Sign Changes vs Lookback')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: f* distribution
    ax = axes[1, 0]
    x = np.arange(len(all_results))
    width = 0.35
    ax.bar(x - width/2, [r['f_star_positive'] for r in all_results], width, label='f* > 0', color='green', alpha=0.7)
    ax.bar(x + width/2, [r['f_star_negative'] for r in all_results], width, label='f* <= 0', color='red', alpha=0.7)
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Count')
    ax.set_title('f* Distribution by Lookback')
    ax.set_xticks(x)
    ax.set_xticklabels([r['lookback'] for r in all_results])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: f* mean vs Lookback
    ax = axes[1, 1]
    ax.plot([r['lookback'] for r in all_results], [r['f_star_mean'] for r in all_results], 'o-', color='orange', linewidth=2, markersize=6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='f* = 0')
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Mean f*')
    ax.set_title('Mean f* vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Drawdown vs Lookback
    ax = axes[1, 2]
    ax.plot([r['lookback'] for r in all_results], [r['max_drawdown'] for r in all_results], 'o-', color='red', linewidth=2, markersize=6)
    ax.axhline(y=baseline_metrics['max_drawdown']*100, color='blue', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Kelly Lookback Optimisation Results', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "kelly_lookback_optimisation.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"Saved: {figures_dir}/kelly_lookback_optimisation.png")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)