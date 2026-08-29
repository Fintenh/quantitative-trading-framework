"""
hmm_optimise.py - Robust walk‑forward optimisation of HMM regime parameters.
Drift fixed at 0.085. Only lookback per regime is optimised.
Uses the same robust pipeline as Phase 3 (walk‑forward CV, Bayesian, polynomial, aggregation).
All HMM/Kelly/GARCH features remain unchanged.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime
import re
import pprint
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
for phase in ["Phase_2", "Phase_3", "Phase_5"]:
    path = os.path.join(parent_dir, phase)
    if os.path.exists(path):
        sys.path.insert(0, path)
sys.path.insert(0, script_dir)

# Import our modules
import hmm_config
from hmm_config import *
from hmm_exploration import train_hmm
from hmm_engine import HMMEngineWithKelly
from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics
from robust_optimiser import RobustOptimiser

# Ensure constants exist
if not hasattr(hmm_config, 'SMOOTH_WINDOW'):
    hmm_config.SMOOTH_WINDOW = 7
    SMOOTH_WINDOW = 7
if not hasattr(hmm_config, 'CASH_MAX_ALLOCATION'):
    hmm_config.CASH_MAX_ALLOCATION = 0.20
    CASH_MAX_ALLOCATION = 0.20

# ---- Fixed parameters (Ethical baseline) ----
FIXED_DRIFT = 0.085
FIXED_ALPHA = 0.92
FIXED_SMOOTH = 7
FIXED_REBALANCE_MIN = 110      # from Phase 5 optimal
FIXED_REBALANCE_MAX = 130      # from Phase 5 optimal

# ---- Parameter space (only lookbacks per regime) ----
PARAM_SPACE_HMM = {
    'bull_lookback': [200, 600],
    'bear_lookback': [150, 500],
    'crash_lookback': [200, 600],
}

# ---- Robust optimiser settings (same as Phase 3) ----
N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15
N_SPLITS = 5
POLYNOMIAL_DEGREE = 4
SMOOTHING_WINDOW = 11

# ---------------------------------------------------------------------------
# Wrapper engine that accepts flat parameters and builds regime_params
# ---------------------------------------------------------------------------

class HMMEngineWrapper(HMMEngineWithKelly):
    """
    Wrapper to accept flat parameters and convert them to the nested regime_params.
    All other arguments are passed through.
    """
    def __init__(self, bull_lookback, bear_lookback, crash_lookback,
                 rebalance_min=FIXED_REBALANCE_MIN, rebalance_max=FIXED_REBALANCE_MAX,
                 drift_threshold=FIXED_DRIFT,
                 **kwargs):
        # Build regime_params from flat args
        regime_params = {
            'bull': {
                'lookback_days': int(bull_lookback),
                'drift_threshold': drift_threshold,
                'rebalance_min_days': int(rebalance_min),
                'rebalance_max_days': int(rebalance_max),
            },
            'bear': {
                'lookback_days': int(bear_lookback),
                'drift_threshold': drift_threshold,
                'rebalance_min_days': int(rebalance_min),
                'rebalance_max_days': int(rebalance_max),
            },
            'crash': {
                'lookback_days': int(crash_lookback),
                'drift_threshold': drift_threshold,
                'rebalance_min_days': int(rebalance_min),
                'rebalance_max_days': int(rebalance_max),
            }
        }
        # Pass regime_params and other kwargs to the parent
        # The parent expects 'regime_params' and also may need 'spy_data', etc.
        # We also need to pass the usual arguments like tickers, start_date, etc.
        # They are in kwargs.
        super().__init__(regime_params=regime_params, **kwargs)


# ---------------------------------------------------------------------------
# Helper to fetch data for the whole period and for folds
# ---------------------------------------------------------------------------

def get_full_data(tickers, start_date, end_date):
    """Download price data for all tickers."""
    print(f"\nDownloading data for {len(tickers)} assets from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.dropna(how='all', axis=1)
    return data


# ---------------------------------------------------------------------------
# Update config with robust parameters
# ---------------------------------------------------------------------------

def update_config(params):
    """Update hmm_config.py with the robust parameters."""
    config_path = os.path.join(script_dir, "hmm_config.py")

    new_regime_params = {
        'bull': {
            'lookback_days': int(params['bull_lookback']),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': FIXED_REBALANCE_MIN,
            'rebalance_max_days': FIXED_REBALANCE_MAX,
        },
        'bear': {
            'lookback_days': int(params['bear_lookback']),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': FIXED_REBALANCE_MIN,
            'rebalance_max_days': FIXED_REBALANCE_MAX,
        },
        'crash': {
            'lookback_days': int(params['crash_lookback']),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': FIXED_REBALANCE_MIN,
            'rebalance_max_days': FIXED_REBALANCE_MAX,
        }
    }

    new_dict_str = pprint.pformat(new_regime_params, indent=4)

    with open(config_path, 'r') as f:
        content = f.read()

    # Find REGIME_PARAMS block and replace
    match = re.search(r'^REGIME_PARAMS\s*=\s*\{', content, re.MULTILINE)
    if not match:
        print("❌ Could not find REGIME_PARAMS block. Aborting update.")
        return

    start_idx = match.start()
    brace_count = 0
    end_idx = start_idx
    for i in range(start_idx, len(content)):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if brace_count != 0:
        print("❌ Could not find matching closing brace. Aborting.")
        return

    new_block = f"REGIME_PARAMS = {new_dict_str}"
    content = content[:start_idx] + new_block + content[end_idx:]

    # Also update alpha and smooth if needed
    content = re.sub(r'ALPHA\s*=\s*[\d.]+', f'ALPHA = {FIXED_ALPHA:.3f}', content)
    content = re.sub(r'SMOOTH_WINDOW\s*=\s*\d+', f'SMOOTH_WINDOW = {FIXED_SMOOTH}', content)

    with open(config_path, 'w') as f:
        f.write(content)

    print(f"✅ Updated {config_path} with robust parameters (drift fixed at {FIXED_DRIFT*100:.1f}%).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 6: ROBUST HMM PARAMETER OPTIMISATION (ETHICAL, KELLY‑ENABLED)")
    print("=" * 70)
    print(f"Period for optimisation: {START_DATE} to {HOLDOUT_END}")   # from hmm_config
    print(f"Assets: {len(OPTIMAL_ETHICAL_PORTFOLIO)} assets")
    print(f"Fixed drift: {FIXED_DRIFT*100:.1f}% (Ethical baseline)")
    print(f"Fixed rebalance: min={FIXED_REBALANCE_MIN}, max={FIXED_REBALANCE_MAX} days")
    print(f"Optimising only lookback per regime: {list(PARAM_SPACE_HMM.keys())}")
    print("=" * 70)

    # ---- Ensure HMM model is trained ----
    if not (os.path.exists(HMM_MODEL_PATH) and os.path.exists(HMM_STATE_MAP_PATH)):
        print("\nNo saved HMM model found. Training now...")
        model, state_to_regime = train_hmm(TRAIN_START, TRAIN_END)
        with open(HMM_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(HMM_STATE_MAP_PATH, 'wb') as f:
            pickle.dump(state_to_regime, f)
    else:
        print("\nLoaded HMM model from disk.")

    # ---- Download full data for the entire period (2010-2026) ----
    # We'll use START_DATE from hmm_config (2010-01-01) and HOLDOUT_END (today)
    # However, RobustOptimiser will split this into folds automatically.
    data = get_full_data(OPTIMAL_ETHICAL_PORTFOLIO, START_DATE, HOLDOUT_END)

    # ---- Download SPY data for regime probabilities (needed by engine) ----
    print("\nDownloading SPY data for regime probabilities...")
    spy_data = yf.download("SPY", start=START_DATE, end=HOLDOUT_END, progress=False)["Close"]
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()

    # ---- Instantiate RobustOptimiser with the wrapper engine ----
    optimiser = RobustOptimiser(
        backtest_engine=HMMEngineWrapper,   # our wrapper that accepts flat params
        param_space=PARAM_SPACE_HMM,
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

    # ---- Run the robust optimisation pipeline ----
    print("\n" + "=" * 70)
    print("STARTING ROBUST OPTIMISATION (Walk‑Forward CV + Bayesian + Polynomial)")
    print("=" * 70)
    results = optimiser.run(data, OPTIMAL_ETHICAL_PORTFOLIO, n_splits=N_SPLITS)

    # ---- Extract robust parameters ----
    robust_params = results['robust_params']
    print("\n" + "=" * 70)
    print("ROBUST PARAMETERS (weighted by out‑of‑sample Sharpe)")
    print("=" * 70)
    for key, val in robust_params.items():
        if key in ['bull_lookback', 'bear_lookback', 'crash_lookback']:
            print(f"  {key}: {val} days")

    print(f"\nRobustness Score: {results['robustness_score']:.1%}")

    print("\nFinal Performance (on full data using robust params):")
    perf = results['final_performance']
    print(f"  Sharpe: {perf['sharpe_ratio']:.3f}")
    print(f"  Return: {perf['total_return']*100:.2f}%")
    print(f"  Drawdown: {perf['max_drawdown']*100:.2f}%")
    print(f"  Volatility: {perf['annualised_volatility']*100:.2f}%")
    print(f"  Trades: {perf['num_trades']}")

    # ---- Show fold results ----
    fold_df = optimiser.get_results_dataframe()
    print("\nFold Results:")
    print(fold_df[['fold', 'sharpe', 'return', 'drawdown', 'lookback_bull', 'lookback_bear', 'lookback_crash']].to_string(index=False))

    # ---- Update hmm_config.py with robust parameters ----
    update_config(robust_params)

    print("\n✅ Robust optimisation complete. You can now run hmm_run.py with the updated parameters.")


if __name__ == "__main__":
    main()