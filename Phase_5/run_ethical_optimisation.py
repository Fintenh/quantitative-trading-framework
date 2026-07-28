"""
run_ethical_optimisation.py
----------------------------
Runs the complete robust optimisation pipeline for Optimal_Standard and Optimal_Ethical.
Uses final portfolios from ethical_config.py.
AUTO-UPDATES ethical_config.py with optimised parameters.
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
import re
from backtest_engine import BacktestEngine
from robust_optimiser import RobustOptimiser
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO,
    OPTIMAL_ETHICAL_PORTFOLIO,
    OPTIMAL_STANDARD_PARAMS,
    OPTIMAL_ETHICAL_PARAMS,
    ORIGINAL_PARAMS,
    RISK_FREE_RATE,
    TRANSACTION_COST_PCT,
    CASH_INTEREST_RATE,
    CASH_MIN_VOLATILITY,
    CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION,
    START_DATE,
    HOLDOUT_START,
    HOLDOUT_END,
    LOG_DIR,
    FIGURES_DIR,
)

# ============================================================================
# SETUP
# ============================================================================

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Logs directory: {LOG_DIR}")
print(f"📁 Figures directory: {FIGURES_DIR}")

# ============================================================================
# CONFIGURATION
# ============================================================================

PARAM_SPACE = {
    'lookback_days': [200, 500],
    'rebalance_min_days': [30, 120],
    'rebalance_max_days': [120, 200],
    'drift_threshold': [0.005, 0.10],
    'take_profit_pct': [0.15, 0.60],
}

N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15
N_SPLITS = 5
POLYNOMIAL_DEGREE = 4
SMOOTHING_WINDOW = 11

print("\n" + "=" * 70)
print("ETHICAL PORTFOLIO OPTIMISATION")
print("=" * 70)
print(f"📊 Standard Portfolio ({len(OPTIMAL_STANDARD_PORTFOLIO)} assets):")
print(f"   {OPTIMAL_STANDARD_PORTFOLIO}")
print(f"\n📊 Ethical Portfolio ({len(OPTIMAL_ETHICAL_PORTFOLIO)} assets):")
print(f"   {OPTIMAL_ETHICAL_PORTFOLIO}")
print(f"\n📅 Period: {START_DATE} to {HOLDOUT_END}")
print("=" * 70)

# ============================================================================
# FUNCTION: Update config with optimised parameters
# ============================================================================

def update_config_params(universe_name, params):
    """
    Update OPTIMAL_STANDARD_PARAMS or OPTIMAL_ETHICAL_PARAMS in ethical_config.py.
    """
    print(f"\n💾 Updating parameters for {universe_name}...")
    
    config_path = os.path.join(script_dir, "ethical_config.py")
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Determine which parameter block to update
    if universe_name == "Optimal_Standard":
        param_name = "OPTIMAL_STANDARD_PARAMS"
    elif universe_name == "Optimal_Ethical":
        param_name = "OPTIMAL_ETHICAL_PARAMS"
    else:
        return
    
    # Build the new parameter string
    new_params = f"""{param_name} = {{
    'lookback_days': {params['lookback_days']},
    'rebalance_min_days': {params['rebalance_min_days']},
    'rebalance_max_days': {params['rebalance_max_days']},
    'drift_threshold': {params['drift_threshold']},
    'take_profit_pct': {params['take_profit_pct']},
    'cash_max_allocation': {params.get('cash_max_allocation', 0.20)},
    'kelly_lookback': {params.get('kelly_lookback', 252)},
}}"""
    
    # Find and replace the parameter block using regex
    pattern = rf'{param_name} = \{{[^}}]*\}}'
    new_content = re.sub(pattern, new_params, content, flags=re.DOTALL)
    
    with open(config_path, 'w') as f:
        f.write(new_content)
    
    print(f"   ✅ Updated {param_name} in {config_path}")

# ============================================================================
# FUNCTION: Get filtered data
# ============================================================================

def get_filtered_data(tickers):
    """Download data and filter assets with sufficient history."""
    print(f"\n📊 Downloading data for {len(tickers)} assets...")
    
    data = yf.download(tickers, start=START_DATE, end=HOLDOUT_END, progress=False)["Close"]
    
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
# FUNCTION: Validate parameters
# ============================================================================

def validate_params(params):
    """Ensure rebalance_max_days >= rebalance_min_days."""
    if params['rebalance_max_days'] < params['rebalance_min_days']:
        print(f"   ⚠️ Fixing invalid params: rebalance_max ({params['rebalance_max_days']}) < rebalance_min ({params['rebalance_min_days']})")
        params['rebalance_max_days'], params['rebalance_min_days'] = params['rebalance_min_days'], params['rebalance_max_days']
        print(f"   ✅ Fixed: rebalance_min={params['rebalance_min_days']}, rebalance_max={params['rebalance_max_days']}")
    return params

# ============================================================================
# FUNCTION: Run optimisation
# ============================================================================

def run_optimisation_for_universe(tickers, universe_name):
    """Run the optimisation for a given universe."""
    
    print("\n" + "=" * 70)
    print(f"RUNNING OPTIMISATION: {universe_name}")
    print("=" * 70)
    
    data, valid_tickers = get_filtered_data(tickers)
    
    if len(valid_tickers) < 5:
        print(f"❌ Only {len(valid_tickers)} assets available. Need at least 5.")
        return None
    
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
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=CASH_MAX_ALLOCATION,
    )
    
    results = optimiser.run(data, valid_tickers, n_splits=N_SPLITS)
    
    if results is None:
        return None
    
    results['robust_params'] = validate_params(results['robust_params'])
    
    # ========================================================================
    # UPDATE CONFIG WITH OPTIMISED PARAMETERS
    # ========================================================================
    update_config_params(universe_name, results['robust_params'])
    
    # Save results to CSV
    print("\n💾 Saving results...")
    
    safe_name = universe_name.replace(" ", "_")
    
    fold_df = optimiser.get_results_dataframe()
    if len(fold_df) > 0:
        fold_df.to_csv(os.path.join(LOG_DIR, f"optimisation_fold_results_{safe_name}.csv"), index=False)
        print(f"✅ Fold results: optimisation_fold_results_{safe_name}.csv")
    
    params_df = pd.DataFrame([results['robust_params']])
    params_df.to_csv(os.path.join(LOG_DIR, f"robust_parameters_{safe_name}.csv"), index=False)
    print(f"✅ Robust parameters: robust_parameters_{safe_name}.csv")
    
    perf_df = pd.DataFrame([results['final_performance']])
    perf_df.to_csv(os.path.join(LOG_DIR, f"final_performance_{safe_name}.csv"), index=False)
    print(f"✅ Final performance: final_performance_{safe_name}.csv")
    
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

def main():
    """Run the optimisation for both portfolios."""
    
    standard_results = run_optimisation_for_universe(
        OPTIMAL_STANDARD_PORTFOLIO, 
        "Optimal_Standard"
    )
    
    ethical_results = run_optimisation_for_universe(
        OPTIMAL_ETHICAL_PORTFOLIO,
        "Optimal_Ethical"
    )
    
    print("\n" + "=" * 70)
    print("OPTIMISATION COMPLETE")
    print("=" * 70)
    print("✅ ethical_config.py has been UPDATED with optimised parameters!")
    
    print("\n| Universe | Assets | Lookback | Rebalance Min | Rebalance Max | Drift | Take-Profit | Sharpe | Return | Drawdown | Trades |")
    print("|----------|--------|----------|---------------|---------------|-------|-------------|--------|--------|----------|--------|")
    
    print(f"| {'Original_18':<10} | {18:>6} | {LOOKBACK_DAYS:>8} | {REBALANCE_MIN_DAYS:>13} | {REBALANCE_MAX_DAYS:>13} | {DRIFT_THRESHOLD*100:>5.1f}% | {RELATIVE_TAKE_PROFIT_PCT*100:>11.1f}% | {0.858:>6.3f} | {1111.45:>6.1f}% | {-22.37:>8.2f}% | {542:>6} |")
    
    if standard_results is not None:
        p = standard_results['robust_params']
        perf = standard_results['final_performance']
        print(f"| {'Optimal_Standard':<10} | {len(OPTIMAL_STANDARD_PORTFOLIO):>6} | {p['lookback_days']:>8} | {p['rebalance_min_days']:>13} | {p['rebalance_max_days']:>13} | {p['drift_threshold']*100:>5.1f}% | {p['take_profit_pct']*100:>11.1f}% | {perf['sharpe_ratio']:>6.3f} | {perf['total_return']*100:>6.1f}% | {perf['max_drawdown']*100:>8.2f}% | {perf['num_trades']:>6} |")
    
    if ethical_results is not None:
        p = ethical_results['robust_params']
        perf = ethical_results['final_performance']
        print(f"| {'Optimal_Ethical':<10} | {len(OPTIMAL_ETHICAL_PORTFOLIO):>6} | {p['lookback_days']:>8} | {p['rebalance_min_days']:>13} | {p['rebalance_max_days']:>13} | {p['drift_threshold']*100:>5.1f}% | {p['take_profit_pct']*100:>11.1f}% | {perf['sharpe_ratio']:>6.3f} | {perf['total_return']*100:>6.1f}% | {perf['max_drawdown']*100:>8.2f}% | {perf['num_trades']:>6} |")
    
    print("\n" + "=" * 70)
    print("🎉 OPTIMISATION COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()