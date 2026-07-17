"""
run_ethical_optimisation.py
----------------------------
Runs the complete robust optimisation pipeline for Optimal_Standard and Optimal_Ethical.
FIXED: Enforces rebalance_max_days >= rebalance_min_days.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, os.path.join(script_dir, "..", "Phase_3"))

from config_optimised import *
import yfinance as yf
import pandas as pd
from backtest_engine import BacktestEngine
from robust_optimiser import RobustOptimiser
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import ethical universes
from ethical_universe import STANDARD_UNIVERSE, ETHICAL_UNIVERSE

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Logs directory: {logs_dir}")
print(f"📁 Figures directory: {figures_dir}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Parameter search space
# IMPORTANT: rebalance_max_days range starts HIGHER than rebalance_min_days
PARAM_SPACE = {
    'lookback_days': [200, 500],           # Capped at 500 to avoid out-of-bounds
    'rebalance_min_days': [30, 120],
    'rebalance_max_days': [120, 200],      # Starts at 120 (>= max rebalance_min)
    'drift_threshold': [0.005, 0.10],
    'take_profit_pct': [0.15, 0.60],
}

# Optimisation settings
N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15
N_SPLITS = 5
POLYNOMIAL_DEGREE = 4
SMOOTHING_WINDOW = 11

print("\n" + "=" * 70)
print("EXPANDED PARAMETER SPACE (WITH VALID REBALANCE BOUNDS)")
print("=" * 70)
print(f"  lookback_days:      {PARAM_SPACE['lookback_days']}")
print(f"  rebalance_min_days: {PARAM_SPACE['rebalance_min_days']}")
print(f"  rebalance_max_days: {PARAM_SPACE['rebalance_max_days']}  (>= rebalance_min_days)")
print(f"  drift_threshold:    {PARAM_SPACE['drift_threshold']}")
print(f"  take_profit_pct:    {PARAM_SPACE['take_profit_pct']}")
print("=" * 70)

# ============================================================================
# FUNCTION: Download and filter data
# ============================================================================

def get_filtered_data(tickers):
    """Download data and filter assets with sufficient history."""
    print(f"\n📊 Downloading data for {len(tickers)} assets...")
    
    data = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False)["Close"]
    
    valid_tickers = []
    for ticker in tickers:
        if ticker in data.columns:
            available_pct = data[ticker].notna().sum() / len(data)
            if available_pct >= 0.8:
                valid_tickers.append(ticker)
    
    data = data[valid_tickers]
    print(f"✅ Using {len(valid_tickers)} assets with sufficient data")
    print(f"   Period: {data.index[0]} to {data.index[-1]}")
    print(f"   Total trading days: {len(data)}")
    
    return data, valid_tickers

# ============================================================================
# FUNCTION: Validate parameters before use
# ============================================================================

def validate_params(params):
    """Ensure rebalance_max_days >= rebalance_min_days."""
    if params['rebalance_max_days'] < params['rebalance_min_days']:
        print(f"   ⚠️ Fixing invalid params: rebalance_max ({params['rebalance_max_days']}) < rebalance_min ({params['rebalance_min_days']})")
        # Swap them
        params['rebalance_max_days'], params['rebalance_min_days'] = params['rebalance_min_days'], params['rebalance_max_days']
        print(f"   ✅ Fixed: rebalance_min={params['rebalance_min_days']}, rebalance_max={params['rebalance_max_days']}")
    return params

# ============================================================================
# FUNCTION: Run optimisation
# ============================================================================

def run_optimisation_for_universe(tickers, universe_name):
    """Run the SAME optimisation as Phase 3 for a given universe."""
    
    print("\n" + "=" * 70)
    print(f"RUNNING OPTIMISATION: {universe_name}")
    print("=" * 70)
    
    # Get filtered data
    data, valid_tickers = get_filtered_data(tickers)
    
    if len(valid_tickers) < 10:
        print(f"❌ Only {len(valid_tickers)} assets available. Need at least 10.")
        return None
    
    # Create optimiser (SAME as Phase 3)
    optimiser = RobustOptimiser(
        backtest_engine=BacktestEngine,
        param_space=PARAM_SPACE,
        n_bayesian_calls=N_BAYESIAN_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        polynomial_degree=POLYNOMIAL_DEGREE,
        smoothing_window=SMOOTHING_WINDOW,
        random_state=42,
        cash_interest_rate=CASH_INTEREST_RATE,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT
    )
    
    # Run optimisation
    results = optimiser.run(data, valid_tickers, n_splits=N_SPLITS)
    
    # Validate and fix parameters
    results['robust_params'] = validate_params(results['robust_params'])
    
    # Save results
    print("\n💾 Saving results...")
    
    safe_name = universe_name.replace(" ", "_")
    
    fold_df = optimiser.get_results_dataframe()
    if len(fold_df) > 0:
        fold_df.to_csv(os.path.join(logs_dir, f"optimisation_fold_results_{safe_name}.csv"), index=False)
        print(f"✅ Fold results: optimisation_fold_results_{safe_name}.csv")
    
    params_df = pd.DataFrame([results['robust_params']])
    params_df.to_csv(os.path.join(logs_dir, f"robust_parameters_{safe_name}.csv"), index=False)
    print(f"✅ Robust parameters: robust_parameters_{safe_name}.csv")
    
    perf_df = pd.DataFrame([results['final_performance']])
    perf_df.to_csv(os.path.join(logs_dir, f"final_performance_{safe_name}.csv"), index=False)
    print(f"✅ Final performance: final_performance_{safe_name}.csv")
    
    # Print results
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {universe_name}")
    print("=" * 70)
    
    print("\n📊 Robust Parameters:")
    for key, val in results['robust_params'].items():
        if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
            print(f"  {key}: {val} days")
        elif key == 'drift_threshold':
            print(f"  {key}: {val*100:.2f}%")
        else:
            print(f"  {key}: {val*100:.1f}%")
    
    print("\n📊 Boundary Check:")
    for key, val in results['robust_params'].items():
        if key in PARAM_SPACE:
            lower, upper = PARAM_SPACE[key]
            if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                if val <= lower:
                    print(f"  ⚠️ {key}: {val} is at LOWER boundary ({lower})")
                elif val >= upper:
                    print(f"  ⚠️ {key}: {val} is at UPPER boundary ({upper})")
                else:
                    print(f"  ✅ {key}: {val} is within range [{lower}, {upper}]")
            else:
                if abs(val - lower) < 0.001 or abs(val - lower) / lower < 0.01:
                    print(f"  ⚠️ {key}: {val:.4f} is at LOWER boundary ({lower})")
                elif abs(val - upper) < 0.001 or abs(val - upper) / upper < 0.01:
                    print(f"  ⚠️ {key}: {val:.4f} is at UPPER boundary ({upper})")
                else:
                    print(f"  ✅ {key}: {val:.4f} is within range [{lower}, {upper}]")
    
    print(f"\n📊 Robustness Score: {results['robustness_score']:.1%}")
    
    print("\n📊 Final Performance:")
    perf = results['final_performance']
    print(f"  Sharpe: {perf['sharpe_ratio']:.3f}")
    print(f"  Return: {perf['total_return']*100:.2f}%")
    print(f"  Drawdown: {perf['max_drawdown']*100:.2f}%")
    print(f"  Volatility: {perf['annualised_volatility']*100:.2f}%")
    print(f"  Trades: {perf['num_trades']}")
    
    print("\n📊 Fold Results:")
    for fold in results['fold_results']:
        print(f"  Fold {fold['fold']}: Sharpe={fold['test_sharpe']:.3f}, "
              f"Return={fold['test_return']*100:.2f}%, "
              f"Drawdown={fold['test_drawdown']*100:.2f}%")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("OPTIMAL PORTFOLIO OPTIMISATION")
print("=" * 70)
print(f"📁 Optimal_Standard: {len(STANDARD_UNIVERSE)} assets")
print(f"📁 Optimal_Ethical: {len(ETHICAL_UNIVERSE)} assets")
print(f"📁 Optimisation Period: {START_DATE} to {END_DATE}")
print(f"📁 Bayesian Calls per fold: {N_BAYESIAN_CALLS}")
print(f"📁 Number of folds: {N_SPLITS}")
print(f"📁 Estimated total backtests: {N_BAYESIAN_CALLS * N_SPLITS}")
print("=" * 70)

# Run optimisation for Optimal_Standard
standard_results = run_optimisation_for_universe(STANDARD_UNIVERSE, "Optimal_Standard")

# Run optimisation for Optimal_Ethical
ethical_results = run_optimisation_for_universe(ETHICAL_UNIVERSE, "Optimal_Ethical")

# ============================================================================
# PRINT COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("OPTIMISATION COMPLETE")
print("=" * 70)

print("\n| Universe | Assets | Lookback | Rebalance Min | Rebalance Max | Drift | Take-Profit | Sharpe | Return | Drawdown | Trades |")
print("|----------|--------|----------|---------------|---------------|-------|-------------|--------|--------|----------|--------|")

# Original (from Phase 3)
print(f"| {'Original_18':<10} | {18:>6} | {LOOKBACK_DAYS:>8} | {REBALANCE_MIN_DAYS:>13} | {REBALANCE_MAX_DAYS:>13} | {DRIFT_THRESHOLD*100:>5.1f}% | {RELATIVE_TAKE_PROFIT_PCT*100:>11.1f}% | {0.858:>6.3f} | {1111.45:>6.1f}% | {-22.37:>8.2f}% | {542:>6} |")

# Optimal_Standard
if standard_results is not None:
    p = standard_results['robust_params']
    perf = standard_results['final_performance']
    print(f"| {'Optimal_Standard':<10} | {len(STANDARD_UNIVERSE):>6} | {p['lookback_days']:>8} | {p['rebalance_min_days']:>13} | {p['rebalance_max_days']:>13} | {p['drift_threshold']*100:>5.1f}% | {p['take_profit_pct']*100:>11.1f}% | {perf['sharpe_ratio']:>6.3f} | {perf['total_return']*100:>6.1f}% | {perf['max_drawdown']*100:>8.2f}% | {perf['num_trades']:>6} |")

# Optimal_Ethical
if ethical_results is not None:
    p = ethical_results['robust_params']
    perf = ethical_results['final_performance']
    print(f"| {'Optimal_Ethical':<10} | {len(ETHICAL_UNIVERSE):>6} | {p['lookback_days']:>8} | {p['rebalance_min_days']:>13} | {p['rebalance_max_days']:>13} | {p['drift_threshold']*100:>5.1f}% | {p['take_profit_pct']*100:>11.1f}% | {perf['sharpe_ratio']:>6.3f} | {perf['total_return']*100:>6.1f}% | {perf['max_drawdown']*100:>8.2f}% | {perf['num_trades']:>6} |")

print("\n" + "=" * 70)
print("🎉 OPTIMISATION COMPLETE!")
print("=" * 70)