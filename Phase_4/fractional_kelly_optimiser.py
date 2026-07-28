"""
fractional_kelly_optimiser.py
-----------------------------
Phase 4: Test different fractional Kelly values.
Uses the CORRECTED Kelly calculation with actual portfolio returns.
Tests different fractions:
  - 100% (Full Kelly)
  - 75% (Aggressive)
  - 50% (Moderate)
  - 25% (Conservative)
  - 0% (No Kelly - Baseline)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add Phase 3 to path for the backtest engine
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import from existing modules
from config_optimised import *
from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

# ============================================================================
# Setup
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("PHASE 4: FRACTIONAL KELLY OPTIMISATION")
print("=" * 70)
print("Using CORRECTED Kelly calculation with actual portfolio returns")
print("f* = (portfolio_return - risk_free_return) / volatility²")

# ============================================================================
# Configuration
# ============================================================================

# Fixed Kelly lookback (from optimisation)
KELLY_LOOKBACK = 135  # Optimal from kelly_lookback_optimiser.py

# Fractional Kelly values to test
KELLY_FRACTIONS = [1.00, 0.75, 0.50, 0.25, 0.0]

# Fraction names for display
FRACTION_NAMES = {
    1.00: "100% (Full Kelly)",
    0.75: "75% (Aggressive)",
    0.50: "50% (Moderate)",
    0.25: "25% (Conservative)",
    0.00: "0% (No Kelly - Baseline)"
}

# Backtest period (2016-2026 as per Phase 3 validation)
BACKTEST_START = "2016-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")

# Base cash cap (20%)
BASE_CASH_CAP = 0.20

print(f"\nConfiguration:")
print(f"  Kelly lookback: {KELLY_LOOKBACK} days (fixed, from optimisation)")
print(f"  Kelly fractions to test: {KELLY_FRACTIONS}")
print(f"  Base cash cap: {BASE_CASH_CAP*100:.0f}%")
print(f"  Backtest period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# Backtest engine with corrected fractional Kelly
# ============================================================================

class BacktestEngineWithFractionalKelly(BacktestEngine):
    """
    Extended BacktestEngine that uses CORRECTED Fractional Kelly.
    Uses equal-weight portfolio returns for Kelly calculation.
    """
    
    def __init__(self, kelly_lookback=126, kelly_fraction=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_fraction = kelly_fraction
        self.prev_f_star = None
        self.kelly_f_stars = []
        self.kelly_cash_caps = []
        self.kelly_sign_changes = 0
        self._force_rebalance = False
        self._debug_count = 0
    
    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using GARCH + CORRECTED Fractional Kelly.
        
        f* = (portfolio_return - risk_free_return) / volatility²
        Then fractional Kelly is applied to the cash cap.
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
            
            # 10. Fractional Kelly cash cap
            if f_star > 0:
                # Positive edge: use base cash cap
                cash_cap = BASE_CASH_CAP
            else:
                # Negative edge: apply fractional Kelly
                # Higher fraction = higher cash cap (more conservative)
                cash_cap = BASE_CASH_CAP + self.kelly_fraction * (1.0 - BASE_CASH_CAP)
            
            # Store for debugging
            self.kelly_f_stars.append(f_star)
            self.kelly_cash_caps.append(cash_cap)
            
            # Debug output for first 10 iterations
            if self._debug_count < 10:
                self._debug_count += 1
                print(f"\n    Kelly Debug #{self._debug_count} (fraction={self.kelly_fraction:.0%}):")
                print(f"      Days: {days}")
                print(f"      Portfolio Return: {portfolio_return:.6f} ({portfolio_return*100:.4f}%)")
                print(f"      Risk-Free Return: {risk_free_return:.6f} ({risk_free_return*100:.4f}%)")
                print(f"      Excess Return: {excess_return:.6f} ({excess_return*100:.4f}%)")
                print(f"      Volatility: {volatility:.6f}")
                print(f"      f*: {f_star:.4f}")
                print(f"      Sign: {'+' if f_star > 0 else '-'}")
                print(f"      Cash Cap: {cash_cap:.0%}")
            
        except Exception as e:
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = self.cash_max_allocation
            if self._debug_count < 10:
                print(f"    Kelly Error: {e}")
                self._debug_count += 1
        
        # 11. Cash allocation based on GARCH volatility (bounded by Kelly cap)
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
# Run baseline (No Kelly - fixed 20% cash cap)
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
    cash_max_allocation=BASE_CASH_CAP
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
# Run fractional Kelly backtests
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING FRACTIONAL KELLY BACKTESTS")
print("=" * 70)
print("Using CORRECTED Kelly calculation with actual portfolio returns")
print("Sign changes trigger early rebalances")
print("=" * 70)

all_results = []

for kelly_fraction in KELLY_FRACTIONS:
    fraction_name = FRACTION_NAMES[kelly_fraction]
    
    print(f"\nTesting Kelly fraction: {fraction_name}")
    print(f"   (Cash cap when f* <= 0: {BASE_CASH_CAP + kelly_fraction * (1.0 - BASE_CASH_CAP):.0%})")
    
    try:
        engine = BacktestEngineWithFractionalKelly(
            tickers=TICKERS,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
            initial_capital=INITIAL_CAPITAL,
            lookback_days=LOOKBACK_DAYS,
            rebalance_min_days=REBALANCE_MIN_DAYS,
            rebalance_max_days=REBALANCE_MAX_DAYS,
            drift_threshold=DRIFT_THRESHOLD,
            take_profit_pct=0.0,  # No take-profit
            risk_free_rate=RISK_FREE_RATE,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            cash_interest_rate=CASH_INTEREST_RATE,
            cash_min_volatility=CASH_MIN_VOLATILITY,
            cash_max_volatility=CASH_MAX_VOLATILITY,
            cash_max_allocation=BASE_CASH_CAP,
            kelly_lookback=KELLY_LOOKBACK,
            kelly_fraction=kelly_fraction
        )
        
        results = engine.run()
        
        metrics = calculate_metrics(
            equity_curve=results['wealth_curve']['value'],
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=RISK_FREE_RATE
        )
        
        # Calculate statistics
        f_stars = np.array(engine.kelly_f_stars) if engine.kelly_f_stars else np.array([0])
        
        result = {
            'kelly_fraction': kelly_fraction,
            'kelly_fraction_name': fraction_name,
            'total_return': metrics['total_return'] * 100,
            'sharpe': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'] * 100,
            'trades': len(results.get('trades', [])),
            'f_star_mean': np.mean(f_stars),
            'f_star_positive': np.sum(f_stars > 0),
            'f_star_negative': np.sum(f_stars <= 0),
            'sign_changes': engine.kelly_sign_changes,
            'cash_cap': BASE_CASH_CAP + kelly_fraction * (1.0 - BASE_CASH_CAP)
        }
        all_results.append(result)
        
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"     Sharpe: {result['sharpe']:.3f}")
        print(f"     Max DD: {result['max_drawdown']:.2f}%")
        print(f"     Sign Changes: {result['sign_changes']}")
        print(f"     f* mean: {result['f_star_mean']:.4f}")
        print(f"     f* > 0: {result['f_star_positive']}, f* <= 0: {result['f_star_negative']}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# Results summary
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

if len(all_results) > 0:
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('kelly_fraction', ascending=False)
    
    print(f"\n| Kelly Fraction | Cash Cap (f*<=0) | Return | Sharpe | Max DD | Trades | Sign Chg | f* > 0 | f* <= 0 |")
    print(f"|----------------|------------------|--------|--------|--------|--------|----------|--------|---------|")
    print(f"| Baseline       | {BASE_CASH_CAP*100:>16.0f}% | {baseline_metrics['total_return']*100:>6.2f}% | {baseline_metrics['sharpe_ratio']:>6.3f} | {baseline_metrics['max_drawdown']*100:>6.2f}% | {len(baseline_results.get('trades', [])):>6} | {'-':>8} | {'-':>6} | {'-':>7} |")
    
    for _, row in df_results.iterrows():
        print(f"| {row['kelly_fraction_name']:>14} | {row['cash_cap']*100:>16.0f}% | {row['total_return']:>6.2f}% | {row['sharpe']:>6.3f} | {row['max_drawdown']:>6.2f}% | {row['trades']:>6} | {row['sign_changes']:>8} | {row['f_star_positive']:>6} | {row['f_star_negative']:>7} |")
    
    # Best performers
    best_sharpe = df_results.loc[df_results['sharpe'].idxmax()]
    best_return = df_results.loc[df_results['total_return'].idxmax()]
    best_dd = df_results.loc[df_results['max_drawdown'].idxmin()]
    
    print("\n" + "=" * 70)
    print("BEST PERFORMERS")
    print("=" * 70)
    print(f"Best Sharpe:    {best_sharpe['kelly_fraction_name']} (Sharpe={best_sharpe['sharpe']:.3f}, Return={best_sharpe['total_return']:.2f}%)")
    print(f"Best Return:    {best_return['kelly_fraction_name']} (Return={best_return['total_return']:.2f}%, Sharpe={best_return['sharpe']:.3f})")
    print(f"Best Drawdown:  {best_dd['kelly_fraction_name']} (DD={best_dd['max_drawdown']:.2f}%, Return={best_dd['total_return']:.2f}%)")
    
    # Overall best by Sharpe
    best_overall = best_sharpe
    
    print(f"\n" + "=" * 70)
    print(f"OPTIMAL KELLY FRACTION: {best_overall['kelly_fraction_name']}")
    print("=" * 70)
    print(f"  Total Return: {best_overall['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {best_overall['sharpe']:.3f}")
    print(f"  Max Drawdown: {best_overall['max_drawdown']:.2f}%")
    print(f"  Trades: {best_overall['trades']}")
    print(f"  Sign Changes: {best_overall['sign_changes']}")
    print(f"  Cash cap (f*<=0): {best_overall['cash_cap']*100:.0f}%")
    
    # Compare to baseline
    improvement = best_overall['total_return'] - baseline_metrics['total_return'] * 100
    sharpe_improvement = best_overall['sharpe'] - baseline_metrics['sharpe_ratio']
    
    print(f"\nImprovement vs Baseline:")
    print(f"  Return: +{improvement:.2f}%")
    print(f"  Sharpe: +{sharpe_improvement:.3f}")
    
    # ============================================================================
    # Plots
    # ============================================================================
    
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert to numpy array for arithmetic operations
    x_labels = [f"{f*100:.0f}%" for f in df_results['kelly_fraction']]
    x_pos = np.arange(len(x_labels))  # ← FIX: Use np.arange() instead of range()
    
    # Plot 1: Return vs Kelly Fraction
    ax = axes[0, 0]
    ax.bar(x_pos, df_results['total_return'], color='blue', alpha=0.7)
    ax.axhline(y=baseline_metrics['total_return']*100, color='red', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Total Return vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Sharpe vs Kelly Fraction
    ax = axes[0, 1]
    ax.bar(x_pos, df_results['sharpe'], color='green', alpha=0.7)
    ax.axhline(y=baseline_metrics['sharpe_ratio'], color='red', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Sign Changes vs Kelly Fraction
    ax = axes[0, 2]
    ax.bar(x_pos, df_results['sign_changes'], color='purple', alpha=0.7)
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Number of Sign Changes')
    ax.set_title('Kelly Sign Changes vs Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Drawdown vs Kelly Fraction
    ax = axes[1, 0]
    ax.bar(x_pos, df_results['max_drawdown'], color='red', alpha=0.7)
    ax.axhline(y=baseline_metrics['max_drawdown']*100, color='blue', linestyle='--', linewidth=1.5, label='Baseline')
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: f* distribution vs Kelly Fraction
    ax = axes[1, 1]
    width = 0.35
    ax.bar(x_pos - width/2, df_results['f_star_positive'], width, label='f* > 0', color='green', alpha=0.7)
    ax.bar(x_pos + width/2, df_results['f_star_negative'], width, label='f* <= 0', color='red', alpha=0.7)
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Count')
    ax.set_title('f* Distribution by Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: f* mean vs Kelly Fraction
    ax = axes[1, 2]
    ax.bar(x_pos, df_results['f_star_mean'], color='orange', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='f* = 0')
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Mean f*')
    ax.set_title('Mean f* vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Fractional Kelly Optimisation Results', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fractional_kelly_results.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: fractional_kelly_results.png")
    
    # ============================================================================
    # Save results
    # ============================================================================
    
    print("\nSaving results...")
    
    df_results.to_csv(os.path.join(logs_dir, "fractional_kelly_results.csv"), index=False)
    
    # Generate config
    config_content = f'''"""
fractional_kelly_config.py
---------------------------
Fractional Kelly parameters from Phase 4 optimisation.
Optimal fraction found: {best_overall['kelly_fraction']*100:.0f}%
"""

# ============================================================================
# Fractional Kelly parameters (from Phase 4 optimisation)
# ============================================================================

KELLY_LOOKBACK = {KELLY_LOOKBACK}              # Optimal lookback for Kelly calculation
KELLY_FRACTION = {best_overall['kelly_fraction']:.2f}  # Optimal Kelly fraction
KELLY_BASE_CAP = {BASE_CASH_CAP}               # 20% base cash cap
KELLY_MAX_CAP = 1.00                           # 100% max cash cap when f* <= 0

# Kelly Cash Cap Rule (with Fractional Kelly):
#   If f* > 0:   cash_cap = KELLY_BASE_CAP (20%)
#   If f* <= 0:  cash_cap = KELLY_BASE_CAP + KELLY_FRACTION * (1.0 - KELLY_BASE_CAP)
#
#   With KELLY_FRACTION = {best_overall['kelly_fraction']*100:.0f}%:
#   Cash cap = {BASE_CASH_CAP + best_overall['kelly_fraction'] * (1.0 - BASE_CASH_CAP):.0%}

# Performance with this configuration:
#   Total Return: {best_overall['total_return']:.2f}%
#   Sharpe Ratio: {best_overall['sharpe']:.3f}
#   Max Drawdown: {best_overall['max_drawdown']:.2f}%
#   Trades: {best_overall['trades']}
#   Sign Changes: {best_overall['sign_changes']}
'''

    config_path = os.path.join(script_dir, "fractional_kelly_config.py")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Config saved: {config_path}")
    
    print(f"\nBest overall (by Sharpe): {best_overall['kelly_fraction_name']}")
    print(f"   Return: {best_overall['total_return']:.2f}%")
    print(f"   Sharpe: {best_overall['sharpe']:.3f}")
    print(f"   Drawdown: {best_overall['max_drawdown']:.2f}%")
    print(f"   Cash cap (f*<=0): {best_overall['cash_cap']*100:.0f}%")
    print(f"   Equity cap (f*<=0): {(1 - best_overall['cash_cap'])*100:.0f}%")

else:
    print("\nNo results generated.")
    df_results = pd.DataFrame()

print("\n" + "=" * 70)
print("FRACTIONAL KELLY OPTIMISATION COMPLETE")
print("=" * 70)

if len(all_results) > 0:
    print(f"\nResults saved to: {logs_dir}/fractional_kelly_results.csv")
    print(f"Figures saved to: {figures_dir}/fractional_kelly_results.png")
    print(f"Config saved to: {config_path}")

print("\nTo use this in your trading engine:")
print("  from fractional_kelly_config import *")