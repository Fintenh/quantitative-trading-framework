"""
kelly_lookback_optimiser.py
----------------------------
Phase 4: Find optimal Kelly lookback period.
Tests different lookback periods for calculating Kelly Criterion cash cap.
Uses existing parameters but replaces fixed 20% cash cap with:
  - 20% if f* > 0
  - 100% if f* <= 0
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

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("PHASE 4: KELLY LOOKBACK OPTIMISATION")
print("=" * 70)
print(f"📁 Working directory: {script_dir}")
print(f"📁 Logs directory: {logs_dir}")
print(f"📁 Figures directory: {figures_dir}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Narrowed Kelly lookback periods (focused around the peak at 126 days)
KELLY_LOOKBACKS = [60, 75, 90, 105, 120, 126, 135, 150, 165, 180, 200]

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

# Backtest period - set end date to today
BACKTEST_START = "2010-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")

print(f"\n📊 Configuration:")
print(f"  Kelly lookbacks to test: {KELLY_LOOKBACKS}")
print(f"  Base cash cap: {BASE_CASH_CAP*100:.0f}%")
print(f"  Backtest period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# KELLY CASH CAP FUNCTION
# ============================================================================

def calculate_kelly_cash_cap(returns_series, risk_free=0.045):
    """
    Calculate cash cap using Kelly Criterion.
    
    Args:
        returns_series: A pandas Series of returns (not DataFrame)
        risk_free: Risk-free rate
    
    Returns:
        cash_cap: 0.20 if f* > 0, 1.00 if f* <= 0
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
    
    # Cash cap: 20% if f* > 0, 100% if f* <= 0
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
        self.kelly_weights = []  # Track Kelly decisions
        self.kelly_f_stars = []  # Track f* values
        self.kelly_cash_caps = []  # Track cash caps
    
    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using GARCH + Kelly.
        
        Kelly determines the cash cap (20% or 100%).
        GARCH determines the actual cash allocation within that cap.
        """
        # 1. Calculate Kelly cash cap using specified lookback
        if len(returns) >= self.kelly_lookback:
            kelly_returns = returns.iloc[-self.kelly_lookback:]
        else:
            kelly_returns = returns
        
        cash_cap, f_star = calculate_kelly_cash_cap(
            kelly_returns, 
            self.risk_free_rate
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
# RUN BASELINE BACKTEST
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING BASELINE (NO KELLY)")
print("=" * 70)

engine_baseline = BacktestEngine(
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
    cash_max_allocation=BASE_CASH_CAP
)

results_baseline = engine_baseline.run()
metrics_baseline = calculate_metrics(
    equity_curve=results_baseline['wealth_curve']['value'],
    initial_capital=INITIAL_CAPITAL,
    risk_free_rate=BASE_PARAMS['risk_free_rate']
)
metrics_baseline['kelly_lookback'] = 'Baseline'
metrics_baseline['total_return_pct'] = metrics_baseline['total_return'] * 100
metrics_baseline['annual_return_pct'] = metrics_baseline['annualised_return'] * 100
metrics_baseline['num_trades'] = len(results_baseline.get('trades', []))

print(f"\n📊 Baseline Results:")
print(f"  Total Return: {metrics_baseline['total_return_pct']:.2f}%")
print(f"  Sharpe Ratio: {metrics_baseline['sharpe_ratio']:.3f}")
print(f"  Max Drawdown: {metrics_baseline['max_drawdown']*100:.2f}%")

# ============================================================================
# RUN KELLY BACKTESTS
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING KELLY BACKTESTS")
print("=" * 70)

all_results = []

for kelly_lookback in KELLY_LOOKBACKS:
    print(f"\n🔄 Testing Kelly lookback: {kelly_lookback} days")
    
    try:
        engine = BacktestEngineWithKelly(
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
            kelly_lookback=kelly_lookback
        )
        
        results = engine.run()
        
        metrics = calculate_metrics(
            equity_curve=results['wealth_curve']['value'],
            initial_capital=INITIAL_CAPITAL,
            risk_free_rate=BASE_PARAMS['risk_free_rate']
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
    
    baseline_row = pd.DataFrame([{
        'kelly_lookback': 'Baseline',
        'total_return_pct': metrics_baseline['total_return_pct'],
        'annual_return_pct': metrics_baseline['annual_return_pct'],
        'sharpe_ratio': metrics_baseline['sharpe_ratio'],
        'max_drawdown': metrics_baseline['max_drawdown'],
        'num_trades': metrics_baseline['num_trades']
    }])
    
    df_all = pd.concat([baseline_row, df_results], ignore_index=True)
    
    print("\n| Lookback | Total Return | Annual Return | Sharpe | Max DD | Trades |")
    print("|----------|--------------|---------------|--------|--------|--------|")
    
    for _, row in df_all.iterrows():
        if row['kelly_lookback'] == 'Baseline':
            print(f"| Baseline | {row['total_return_pct']:>10.2f}% | {row['annual_return_pct']:>12.2f}% | {row['sharpe_ratio']:>6.3f} | {row['max_drawdown']*100:>6.2f}% | {row['num_trades']:>6} |")
        else:
            print(f"| {row['kelly_lookback']:>8} | {row['total_return_pct']:>10.2f}% | {row['annual_return_pct']:>12.2f}% | {row['sharpe_ratio']:>6.3f} | {row['max_drawdown']*100:>6.2f}% | {row['num_trades']:>6} |")
    
    if len(df_results) > 0:
        best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
        best_return = df_results.loc[df_results['total_return_pct'].idxmax()]
        best_dd = df_results.loc[df_results['max_drawdown'].idxmin()]
        
        print("\n" + "=" * 70)
        print("BEST PERFORMERS (vs Baseline)")
        print("=" * 70)
        print(f"🏆 Best Sharpe:     {best_sharpe['kelly_lookback']} days (Sharpe={best_sharpe['sharpe_ratio']:.3f})")
        print(f"📈 Best Return:    {best_return['kelly_lookback']} days (Return={best_return['total_return_pct']:.2f}%)")
        print(f"🛡️ Best Drawdown:  {best_dd['kelly_lookback']} days (DD={best_dd['max_drawdown']*100:.2f}%)")
        
        best_kelly = best_return['kelly_lookback']
        best_sharpe_val = best_return['sharpe_ratio']
        
        print(f"\n✅ Optimal Kelly lookback: {best_kelly} days (Return={best_return['total_return_pct']:.2f}%)")
        
        # ============================================================================
        # GENERATE PLOTS - COMPACT VERSION
        # ============================================================================
        
        print("\n📊 Generating plots...")
        
        # Create a more compact figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Return vs Lookback
        ax = axes[0, 0]
        ax.plot(df_results['kelly_lookback'], df_results['total_return_pct'], 'o-', color='blue', linewidth=2, markersize=6)
        ax.axhline(y=metrics_baseline['total_return_pct'], color='red', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
        ax.set_ylabel('Total Return (%)', fontsize=9)
        ax.set_title('Total Return vs Kelly Lookback', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Sharpe vs Lookback
        ax = axes[0, 1]
        ax.plot(df_results['kelly_lookback'], df_results['sharpe_ratio'], 'o-', color='green', linewidth=2, markersize=6)
        ax.axhline(y=metrics_baseline['sharpe_ratio'], color='red', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
        ax.set_ylabel('Sharpe Ratio', fontsize=9)
        ax.set_title('Sharpe Ratio vs Kelly Lookback', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Drawdown vs Lookback
        ax = axes[1, 0]
        ax.plot(df_results['kelly_lookback'], df_results['max_drawdown'] * 100, 'o-', color='orange', linewidth=2, markersize=6)
        ax.axhline(y=metrics_baseline['max_drawdown'] * 100, color='red', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
        ax.set_ylabel('Max Drawdown (%)', fontsize=9)
        ax.set_title('Max Drawdown vs Kelly Lookback', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Trades vs Lookback
        ax = axes[1, 1]
        ax.plot(df_results['kelly_lookback'], df_results['num_trades'], 'o-', color='purple', linewidth=2, markersize=6)
        ax.axhline(y=metrics_baseline['num_trades'], color='red', linestyle='--', linewidth=1.5, label='Baseline')
        ax.set_xlabel('Kelly Lookback (days)', fontsize=9)
        ax.set_ylabel('Number of Trades', fontsize=9)
        ax.set_title('Trades vs Kelly Lookback', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(figures_dir, "kelly_lookback_optimisation.png"), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"✅ Saved: kelly_lookback_optimisation.png")
        
        # ============================================================================
        # SAVE RESULTS
        # ============================================================================
        
        print("\n💾 Saving results...")
        
        df_all.to_csv(os.path.join(logs_dir, "kelly_lookback_results.csv"), index=False)
        
        config_update = f'''"""
kelly_config.py
----------------
Kelly Criterion parameters from Phase 4 optimisation.
Optimal lookback found: {best_kelly} days
"""

# ============================================================================
# KELLY PARAMETERS (from Phase 4 optimisation)
# ============================================================================

KELLY_LOOKBACK = {best_kelly}  # Optimal lookback for Kelly calculation
KELLY_BASE_CAP = 0.20          # 20% base cash cap
KELLY_MAX_CAP = 1.00           # 100% max cash cap when f* <= 0
KELLY_FRACTION = 1.00          # Full Kelly (0-1 for fractional)

# Kelly Cash Cap Rule:
#   If f* > 0:   cash_cap = KELLY_BASE_CAP (20%)
#   If f* <= 0:  cash_cap = KELLY_MAX_CAP (100%)

# Performance with this lookback:
#   Total Return: {best_return['total_return_pct']:.2f}%
#   Sharpe Ratio: {best_return['sharpe_ratio']:.3f}
#   Max Drawdown: {best_return['max_drawdown']*100:.2f}%
'''

        config_path = os.path.join(script_dir, "kelly_config.py")
        with open(config_path, "w") as f:
            f.write(config_update)
        
        print(f"✅ Kelly config saved: {config_path}")
        
    else:
        print("\n❌ No Kelly results to compare.")
        df_all = pd.DataFrame()

else:
    print("\n❌ No results from any Kelly backtest.")
    df_all = pd.DataFrame([{
        'kelly_lookback': 'Baseline',
        'total_return_pct': metrics_baseline['total_return_pct'],
        'annual_return_pct': metrics_baseline['annual_return_pct'],
        'sharpe_ratio': metrics_baseline['sharpe_ratio'],
        'max_drawdown': metrics_baseline['max_drawdown'],
        'num_trades': metrics_baseline['num_trades']
    }])
    df_all.to_csv(os.path.join(logs_dir, "kelly_lookback_results.csv"), index=False)

print("\n" + "=" * 70)
print("🎉 KELLY LOOKBACK OPTIMISATION COMPLETE!")
print("=" * 70)

if len(all_results) > 0:
    print(f"\n📊 Results saved to: {logs_dir}/kelly_lookback_results.csv")
    print(f"📊 Figures saved to: {figures_dir}/kelly_lookback_optimisation.png")
    print(f"📝 Kelly config saved to: {config_path}")
    
    print(f"\n🏆 Best Kelly lookback: {best_kelly} days")
    print(f"   Return: {best_return['total_return_pct']:.2f}%")
    print(f"   Sharpe: {best_return['sharpe_ratio']:.3f}")
    print(f"   Drawdown: {best_return['max_drawdown']*100:.2f}%")
else:
    print("\n⚠️ No Kelly results were generated. Check the errors above.")

print("\nTo use this in your trading engine:")
print("  from kelly_config import *")