"""
fractional_kelly_optimiser.py
-----------------------------
Phase 4: Test different fractional Kelly values.
Uses optimal Kelly lookback (126 days) and tests different fractions:
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

# Add Phase 3 to path for backtest engine
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

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("PHASE 4: FRACTIONAL KELLY OPTIMISATION")
print("=" * 70)
print(f"📁 Working directory: {script_dir}")
print(f"📁 Logs directory: {logs_dir}")
print(f"📁 Figures directory: {figures_dir}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fixed Kelly lookback (from previous optimisation)
KELLY_LOOKBACK = 126

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

# Strategy parameters (from Phase 3)
BASE_PARAMS = {
    'lookback_days': LOOKBACK_DAYS,
    'rebalance_min_days': REBALANCE_MIN_DAYS,
    'rebalance_max_days': REBALANCE_MAX_DAYS,
    'drift_threshold': DRIFT_THRESHOLD,
    'take_profit_pct': RELATIVE_TAKE_PROFIT_PCT,
    'risk_free_rate': RISK_FREE_RATE,
    'transaction_cost_pct': TRANSACTION_COST_PCT,
    'cash_interest_rate': CASH_INTEREST_RATE,
    'cash_min_volatility': CASH_MIN_VOLATILITY,
    'cash_max_volatility': CASH_MAX_VOLATILITY,
}

# Base cash cap (20%)
BASE_CASH_CAP = 0.20

# Backtest period
BACKTEST_START = "2010-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")

print(f"\n📊 Configuration:")
print(f"  Kelly lookback: {KELLY_LOOKBACK} days (fixed)")
print(f"  Kelly fractions to test: {KELLY_FRACTIONS}")
print(f"  Base cash cap: {BASE_CASH_CAP*100:.0f}%")
print(f"  Backtest period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# KELLY CASH CAP FUNCTION (WITH FRACTIONAL KELLY)
# ============================================================================

def calculate_kelly_cash_cap(returns_series, risk_free=0.045, kelly_fraction=1.0):
    """
    Calculate cash cap using Fractional Kelly Criterion.
    
    Args:
        returns_series: A pandas Series of returns (not DataFrame)
        risk_free: Risk-free rate
        kelly_fraction: 0.0 to 1.0 (1.0 = full Kelly)
    
    Returns:
        cash_cap: The maximum cash allocation
        f_star: The Kelly fraction
    """
    # Ensure we have a Series, not DataFrame
    if isinstance(returns_series, pd.DataFrame):
        if len(returns_series.columns) > 0:
            returns_series = returns_series.iloc[:, 0]
        else:
            return BASE_CASH_CAP, 0.0
    
    # Drop NaN values
    returns_series = returns_series.dropna()
    
    if len(returns_series) < 10:
        return BASE_CASH_CAP, 0.0
    
    # Calculate expected return and volatility
    mu = returns_series.mean() * 252
    sigma = returns_series.std() * np.sqrt(252)
    
    if sigma <= 0 or not np.isfinite(sigma):
        return BASE_CASH_CAP, 0.0
    
    # Full Kelly
    f_star = (mu - risk_free) / (sigma ** 2)
    
    # Handle infinite or NaN
    if not np.isfinite(f_star):
        return BASE_CASH_CAP, 0.0
    
    # Calculate cash cap with fractional Kelly
    if f_star > 0:
        # Positive edge: use base cash cap
        cash_cap = BASE_CASH_CAP
    else:
        # Negative edge: apply fractional Kelly
        cash_cap = BASE_CASH_CAP + kelly_fraction * (1.0 - BASE_CASH_CAP)
    
    return cash_cap, f_star


# ============================================================================
# MODIFIED BACKTEST ENGINE (WITH FRACTIONAL KELLY)
# ============================================================================

class BacktestEngineWithFractionalKelly(BacktestEngine):
    """
    Extended BacktestEngine that uses Fractional Kelly Criterion for cash allocation.
    """
    
    def __init__(self, kelly_lookback=126, kelly_fraction=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_fraction = kelly_fraction
        self.kelly_weights = []  # Track Kelly decisions
        self.kelly_f_stars = []  # Track f* values
        self.kelly_cash_caps = []  # Track cash caps
    
    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using GARCH + Fractional Kelly.
        """
        # 1. Calculate Kelly cash cap using specified lookback
        if len(returns) >= self.kelly_lookback:
            kelly_returns = returns.iloc[-self.kelly_lookback:]
        else:
            kelly_returns = returns
        
        cash_cap, f_star = calculate_kelly_cash_cap(
            kelly_returns, 
            self.risk_free_rate,
            self.kelly_fraction
        )
        
        # Store for analysis
        self.kelly_f_stars.append(f_star)
        self.kelly_cash_caps.append(cash_cap)
        
        # 2. Calculate GARCH volatility
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            models, _, _ = fit_garch_for_assets(returns)
            vols = get_latest_volatility(models, returns)
            avg_vol = get_average_volatility(vols)
        except Exception as e:
            avg_vol = returns.std().mean() * np.sqrt(252)
        
        # 3. Calculate GARCH cash allocation (bounded by Kelly cap)
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
# RUN BACKTESTS FOR EACH KELLY FRACTION
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING FRACTIONAL KELLY BACKTESTS")
print("=" * 70)

all_results = []

for kelly_fraction in KELLY_FRACTIONS:
    fraction_name = FRACTION_NAMES[kelly_fraction]
    
    print(f"\n🔄 Testing Kelly fraction: {fraction_name}")
    print(f"   (Cash cap when f* <= 0: {BASE_CASH_CAP + kelly_fraction * (1.0 - BASE_CASH_CAP):.0%})")
    
    try:
        engine = BacktestEngineWithFractionalKelly(
            tickers=TICKERS,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
            initial_capital=INITIAL_CAPITAL,
            lookback_days=BASE_PARAMS['lookback_days'],
            rebalance_min_days=BASE_PARAMS['rebalance_min_days'],
            rebalance_max_days=BASE_PARAMS['rebalance_max_days'],
            drift_threshold=BASE_PARAMS['drift_threshold'],
            take_profit_pct=BASE_PARAMS['take_profit_pct'],
            risk_free_rate=BASE_PARAMS['risk_free_rate'],
            transaction_cost_pct=BASE_PARAMS['transaction_cost_pct'],
            cash_interest_rate=BASE_PARAMS['cash_interest_rate'],
            cash_min_volatility=BASE_PARAMS['cash_min_volatility'],
            cash_max_volatility=BASE_PARAMS['cash_max_volatility'],
            cash_max_allocation=BASE_CASH_CAP,
            kelly_lookback=KELLY_LOOKBACK,
            kelly_fraction=kelly_fraction
        )
        
        results = engine.run()
        
        metrics = calculate_metrics(
            equity_curve=results['wealth_curve']['value'],
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=BASE_PARAMS['risk_free_rate']
        )
        
        metrics['kelly_fraction'] = kelly_fraction
        metrics['kelly_fraction_name'] = fraction_name
        metrics['cash_cap'] = BASE_CASH_CAP + kelly_fraction * (1.0 - BASE_CASH_CAP)
        metrics['total_return_pct'] = metrics['total_return'] * 100
        metrics['annual_return_pct'] = metrics['annualised_return'] * 100
        metrics['num_trades'] = len(results.get('trades', []))
        
        all_results.append(metrics)
        
        print(f"  ✅ Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"     Sharpe: {metrics['sharpe_ratio']:.3f}")
        print(f"     Max DD: {metrics['max_drawdown']*100:.2f}%")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# COMPARE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

if len(all_results) > 0:
    df_results = pd.DataFrame(all_results)
    
    # Sort by Kelly fraction
    df_results = df_results.sort_values('kelly_fraction', ascending=False)
    
    print("\n| Kelly Fraction | Cash Cap (f*<=0) | Total Return | Annual Return | Sharpe | Max DD | Trades |")
    print("|----------------|------------------|--------------|---------------|--------|--------|--------|")
    
    for _, row in df_results.iterrows():
        print(f"| {row['kelly_fraction_name']:>14} | {row['cash_cap']*100:>14.0f}% | {row['total_return_pct']:>12.2f}% | {row['annual_return_pct']:>13.2f}% | {row['sharpe_ratio']:>6.3f} | {row['max_drawdown']*100:>6.2f}% | {row['num_trades']:>6} |")
    
    # Find best by different metrics
    best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    best_return = df_results.loc[df_results['total_return_pct'].idxmax()]
    best_dd = df_results.loc[df_results['max_drawdown'].idxmin()]
    
    print("\n" + "=" * 70)
    print("BEST PERFORMERS")
    print("=" * 70)
    print(f"🏆 Best Sharpe:     {best_sharpe['kelly_fraction_name']} (Sharpe={best_sharpe['sharpe_ratio']:.3f}, Return={best_sharpe['total_return_pct']:.2f}%)")
    print(f"📈 Best Return:    {best_return['kelly_fraction_name']} (Return={best_return['total_return_pct']:.2f}%, Sharpe={best_return['sharpe_ratio']:.3f})")
    print(f"🛡️ Best Drawdown:  {best_dd['kelly_fraction_name']} (DD={best_dd['max_drawdown']*100:.2f}%, Return={best_dd['total_return_pct']:.2f}%)")
    
    # ============================================================================
    # GENERATE PLOTS
    # ============================================================================
    
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert fractions to labels for x-axis
    x_labels = [f"{f*100:.0f}%" for f in df_results['kelly_fraction']]
    x_pos = range(len(x_labels))
    
    # Return vs Kelly Fraction
    ax = axes[0, 0]
    ax.bar(x_pos, df_results['total_return_pct'], color='blue', alpha=0.7)
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Total Return (%)')
    ax.set_title('Total Return vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Sharpe vs Kelly Fraction
    ax = axes[0, 1]
    ax.bar(x_pos, df_results['sharpe_ratio'], color='green', alpha=0.7)
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Drawdown vs Kelly Fraction
    ax = axes[1, 0]
    ax.bar(x_pos, df_results['max_drawdown'] * 100, color='red', alpha=0.7)
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Max Drawdown (%)')
    ax.set_title('Max Drawdown vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Trades vs Kelly Fraction
    ax = axes[1, 1]
    ax.bar(x_pos, df_results['num_trades'], color='purple', alpha=0.7)
    ax.set_xlabel('Kelly Fraction')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trades vs Kelly Fraction')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fractional_kelly_results.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved: fractional_kelly_results.png")
    
    # ============================================================================
    # CASH CAP vs KELLY FRACTION TABLE
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("CASH CAP VS KELLY FRACTION")
    print("=" * 70)
    print("\n| Kelly Fraction | Cash Cap (f*<=0) | Equity Cap |")
    print("|----------------|------------------|------------|")
    for _, row in df_results.iterrows():
        equity_cap = (1 - row['cash_cap']) * 100
        print(f"| {row['kelly_fraction_name']:>14} | {row['cash_cap']*100:>16.0f}% | {equity_cap:>10.0f}% |")
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    
    print("\n💾 Saving results...")
    
    df_results.to_csv(os.path.join(logs_dir, "fractional_kelly_results.csv"), index=False)
    
    # Find best overall (by Sharpe)
    best_overall = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    
    # Generate config
    config_update = f'''"""
fractional_kelly_config.py
---------------------------
Fractional Kelly parameters from Phase 4 optimisation.
Optimal fraction found: {best_overall['kelly_fraction']*100:.0f}%
"""

# ============================================================================
# FRACTIONAL KELLY PARAMETERS (from Phase 4 optimisation)
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
#   Total Return: {best_overall['total_return_pct']:.2f}%
#   Sharpe Ratio: {best_overall['sharpe_ratio']:.3f}
#   Max Drawdown: {best_overall['max_drawdown']*100:.2f}%
#   Trades: {best_overall['num_trades']}
'''

    config_path = os.path.join(script_dir, "fractional_kelly_config.py")
    with open(config_path, "w") as f:
        f.write(config_update)
    
    print(f"✅ Kelly config saved: {config_path}")
    
else:
    print("\n❌ No results generated.")
    df_results = pd.DataFrame()

print("\n" + "=" * 70)
print("🎉 FRACTIONAL KELLY OPTIMISATION COMPLETE!")
print("=" * 70)

if len(all_results) > 0:
    print(f"\n📊 Results saved to: {logs_dir}/fractional_kelly_results.csv")
    print(f"📊 Figures saved to: {figures_dir}/fractional_kelly_results.png")
    print(f"📝 Config saved to: {config_path}")
    
    print(f"\n🏆 Best overall (by Sharpe): {best_overall['kelly_fraction_name']}")
    print(f"   Return: {best_overall['total_return_pct']:.2f}%")
    print(f"   Sharpe: {best_overall['sharpe_ratio']:.3f}")
    print(f"   Drawdown: {best_overall['max_drawdown']*100:.2f}%")
    print(f"   Cash cap (f*<=0): {best_overall['cash_cap']*100:.0f}%")
    print(f"   Equity cap (f*<=0): {(1 - best_overall['cash_cap'])*100:.0f}%")
else:
    print("\n⚠️ No results generated. Check the errors above.")

print("\nTo use this in your trading engine:")
print("  from fractional_kelly_config import *")