"""
hmm_optimise.py - Walk-forward validation for HMM regime parameters.
Drift fixed at 0.085; optimises lookback, rebalance min/max per regime.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Integer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter
import re
import pprint
import contextlib
from io import StringIO

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
from performance_metrics import calculate_metrics
from walk_forward_cv import WalkForwardCV

# ---- Fixed parameters ----
FIXED_DRIFT = 0.085
FIXED_ALPHA = 0.92
FIXED_SMOOTH = 7

# ---- Parameter space ----
PARAM_SPACE = [
    Integer(300, 600, name='bull_lookback'),
    Integer(60, 160, name='bull_min'),
    Integer(80, 220, name='bull_max'),
    Integer(300, 600, name='bear_lookback'),
    Integer(50, 160, name='bear_min'),
    Integer(60, 200, name='bear_max'),
    Integer(200, 500, name='crash_lookback'),
    Integer(60, 160, name='crash_min'),
    Integer(80, 200, name='crash_max'),
]
PARAM_NAMES = [p.name for p in PARAM_SPACE]

# ---- Settings ----
N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15
N_SPLITS = 5
POLYNOMIAL_DEGREE = 4
SMOOTHING_WINDOW = 11

# ---------------------------------------------------------------------------
# Helper: enforce min <= max
# ---------------------------------------------------------------------------

def enforce_min_max_for_dict(d):
    """Enforce min <= max for bull, bear, crash in a flat params dict."""
    for regime in ['bull', 'bear', 'crash']:
        min_key = f'{regime}_min'
        max_key = f'{regime}_max'
        if min_key in d and max_key in d:
            if d[min_key] > d[max_key]:
                d[min_key], d[max_key] = d[max_key], d[min_key]
    return d

def enforce_min_max_for_regime_params(regime_params):
    """Enforce min <= max for each regime in a nested regime_params dict."""
    for regime in ['bull', 'bear', 'crash']:
        if 'rebalance_min_days' in regime_params[regime] and 'rebalance_max_days' in regime_params[regime]:
            if regime_params[regime]['rebalance_min_days'] > regime_params[regime]['rebalance_max_days']:
                regime_params[regime]['rebalance_min_days'], regime_params[regime]['rebalance_max_days'] = \
                    regime_params[regime]['rebalance_max_days'], regime_params[regime]['rebalance_min_days']
    return regime_params

# ---------------------------------------------------------------------------
# Helper: run backtest
# ---------------------------------------------------------------------------

def run_hmm_backtest(tickers, start_date, end_date, regime_params, spy_data, verbose=False):
    """Run HMM backtest."""
    engine = HMMEngineWithKelly(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100.0,
        lookback_days=OPTIMAL_ETHICAL_PARAMS['lookback_days'],
        rebalance_min_days=110,
        rebalance_max_days=130,
        drift_threshold=FIXED_DRIFT,
        take_profit_pct=0.0,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=CASH_MAX_ALLOCATION,
        kelly_lookback=OPTIMAL_ETHICAL_PARAMS.get('kelly_lookback', 165),
        kelly_base_cap=KELLY_BASE_CAP,
        kelly_max_cap=KELLY_MAX_CAP,
        spy_data=spy_data,
        regime_params=regime_params,
        verbose=verbose,
    )
    if not verbose:
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
            results = engine.run()
    else:
        results = engine.run()
    wealth = results['wealth_curve']['value']
    metrics = calculate_metrics(wealth, 100.0, RISK_FREE_RATE)
    return metrics, len(results.get('trades', []))

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(params, tickers, train_data, spy_data, verbose=False):
    (bull_lookback, bull_min, bull_max,
     bear_lookback, bear_min, bear_max,
     crash_lookback, crash_min, crash_max) = params

    # Enforce min <= max (critical)
    if bull_min > bull_max:
        bull_min, bull_max = bull_max, bull_min
    if bear_min > bear_max:
        bear_min, bear_max = bear_max, bear_min
    if crash_min > crash_max:
        crash_min, crash_max = crash_max, crash_min

    regime_params = {
        'bull': {
            'lookback_days': int(bull_lookback),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': int(bull_min),
            'rebalance_max_days': int(bull_max),
        },
        'bear': {
            'lookback_days': int(bear_lookback),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': int(bear_min),
            'rebalance_max_days': int(bear_max),
        },
        'crash': {
            'lookback_days': int(crash_lookback),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': int(crash_min),
            'rebalance_max_days': int(crash_max),
        }
    }

    start_date = train_data.index[0].strftime('%Y-%m-%d')
    end_date = train_data.index[-1].strftime('%Y-%m-%d')
    metrics, trades = run_hmm_backtest(
        tickers, start_date, end_date, regime_params, spy_data, verbose=verbose
    )

    total_return = min(max(metrics['total_return'], -1), 2)
    return_score = (total_return + 1) / 3
    sharpe = min(max(metrics['sharpe_ratio'], -2), 3)
    sharpe_score = (sharpe + 2) / 5
    score = 0.70 * return_score + 0.30 * sharpe_score

    if verbose:
        print(f"  Return: {total_return*100:.2f}%, Sharpe: {sharpe:.4f}, Score: {score:.4f}")
    return -score

# ---------------------------------------------------------------------------
# Polynomial fitting
# ---------------------------------------------------------------------------

def fit_polynomial(results_df, param_names, poly_degree=4, smooth_window=11):
    optimum = {}
    for param in param_names:
        grouped = results_df.groupby(param)['score'].mean().reset_index()
        grouped = grouped.sort_values(param)
        x_vals = grouped[param].values
        y_vals = grouped['score'].values

        if len(x_vals) > poly_degree:
            try:
                X = x_vals.reshape(-1, 1)
                model = make_pipeline(
                    PolynomialFeatures(degree=poly_degree),
                    Ridge(alpha=0.1)
                )
                model.fit(X, y_vals)
                x_range = np.linspace(x_vals.min(), x_vals.max(), 1000)
                X_range = x_range.reshape(-1, 1)
                y_pred = model.predict(X_range)
                if len(y_pred) > smooth_window:
                    y_pred = savgol_filter(y_pred, smooth_window, 3)
                peak_idx = np.argmax(y_pred)
                optimal_x = x_range[peak_idx]
                if param in ['bull_lookback', 'bear_lookback', 'crash_lookback',
                             'bull_min', 'bear_min', 'crash_min',
                             'bull_max', 'bear_max', 'crash_max']:
                    optimal_x = int(round(optimal_x))
                else:
                    optimal_x = round(optimal_x, 3)
                optimum[param] = optimal_x
            except:
                max_idx = np.argmax(y_vals)
                optimum[param] = x_vals[max_idx]
        else:
            max_idx = np.argmax(y_vals)
            optimum[param] = x_vals[max_idx]
    return optimum

# ---------------------------------------------------------------------------
# Update config
# ---------------------------------------------------------------------------

def update_config(params):
    """Update hmm_config.py with robust parameters in consistent order."""
    config_path = os.path.join(script_dir, "hmm_config.py")

    # Build new regime params in consistent order: bull, bear, crash
    # Each regime: lookback_days, rebalance_min_days, rebalance_max_days, drift_threshold
    new_regime_params = {
        'bull': {
            'lookback_days': int(params['bull_lookback']),
            'rebalance_min_days': int(params['bull_min']),
            'rebalance_max_days': int(params['bull_max']),
            'drift_threshold': FIXED_DRIFT,
        },
        'bear': {
            'lookback_days': int(params['bear_lookback']),
            'rebalance_min_days': int(params['bear_min']),
            'rebalance_max_days': int(params['bear_max']),
            'drift_threshold': FIXED_DRIFT,
        },
        'crash': {
            'lookback_days': int(params['crash_lookback']),
            'rebalance_min_days': int(params['crash_min']),
            'rebalance_max_days': int(params['crash_max']),
            'drift_threshold': FIXED_DRIFT,
        }
    }

    new_dict_str = pprint.pformat(new_regime_params, indent=4, sort_dicts=False)

    with open(config_path, 'r') as f:
        content = f.read()

    match = re.search(r'^REGIME_PARAMS\s*=\s*\{', content, re.MULTILINE)
    if not match:
        print("Could not find REGIME_PARAMS block.")
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
        print("Could not find matching closing brace. Aborting.")
        return

    new_block = f"REGIME_PARAMS = {new_dict_str}"
    content = content[:start_idx] + new_block + content[end_idx:]

    # Also update alpha and smooth if needed
    content = re.sub(r'ALPHA\s*=\s*[\d.]+', f'ALPHA = {FIXED_ALPHA:.3f}', content)
    content = re.sub(r'SMOOTH_WINDOW\s*=\s*\d+', f'SMOOTH_WINDOW = {FIXED_SMOOTH}', content)

    with open(config_path, 'w') as f:
        f.write(content)

    print(f"Updated {config_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OPTIM_START = "2010-01-01"
    OPTIM_END = TEST_END

    print("=" * 70)
    print("PHASE 6: ROBUST HMM PARAMETER OPTIMISATION")
    print("=" * 70)
    print(f"Period: {OPTIM_START} to {OPTIM_END}")
    print(f"Assets: {len(OPTIMAL_ETHICAL_PORTFOLIO)}")
    print(f"Drift: {FIXED_DRIFT*100:.1f}% (fixed)")
    print(f"Calls per fold: {N_BAYESIAN_CALLS}, Folds: {N_SPLITS}")
    print("=" * 70)

    # ---- Load HMM ----
    if not (os.path.exists(HMM_MODEL_PATH) and os.path.exists(HMM_STATE_MAP_PATH)):
        print("\nTraining HMM...")
        model, state_to_regime = train_hmm(TRAIN_START, TRAIN_END)
        with open(HMM_MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(HMM_STATE_MAP_PATH, 'wb') as f:
            pickle.dump(state_to_regime, f)
    else:
        print("\nLoaded HMM.")

    # ---- Download data ----
    print(f"\nDownloading data...")
    data = yf.download(OPTIMAL_ETHICAL_PORTFOLIO, start=OPTIM_START, end=OPTIM_END, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.dropna(how='all', axis=1)

    print("Downloading SPY...")
    spy_data = yf.download("SPY", start=OPTIM_START, end=OPTIM_END, progress=False)["Close"]
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()

    # ---- Align SPY with data ----
    spy_aligned = spy_data.reindex(data.index).ffill().bfill()
    if isinstance(spy_aligned, pd.DataFrame):
        spy_aligned = spy_aligned.squeeze()

    # ---- Walk-forward ----
    cv = WalkForwardCV(train_size_years=5, test_size_years=1, step_size_years=1)
    folds = cv.split(data)
    if len(folds) < N_SPLITS:
        N_SPLITS_used = len(folds)
    else:
        N_SPLITS_used = N_SPLITS

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds[:N_SPLITS_used]):
        fold_info = cv.get_fold_info(data, fold_idx, train_idx, test_idx)
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx+1}/{N_SPLITS_used}")
        print(f"{'='*70}")
        print(f"  Train: {fold_info['train_start'].strftime('%Y-%m-%d')} -> {fold_info['train_end'].strftime('%Y-%m-%d')}")
        print(f"  Test:  {fold_info['test_start'].strftime('%Y-%m-%d')} -> {fold_info['test_end'].strftime('%Y-%m-%d')}")

        # ---- Bayesian ----
        print(f"\nBayesian Optimisation ({N_BAYESIAN_CALLS} calls)...")
        def obj_wrapper(params):
            return objective(params, OPTIMAL_ETHICAL_PORTFOLIO, train_data, spy_aligned, verbose=False)

        result = gp_minimize(
            obj_wrapper,
            PARAM_SPACE,
            n_calls=N_BAYESIAN_CALLS,
            n_initial_points=N_INITIAL_POINTS,
            random_state=42,
            verbose=False
        )

        # Collect results
        all_params = []
        all_scores = []
        for x in result.x_iters:
            score = -objective(x, OPTIMAL_ETHICAL_PORTFOLIO, train_data, spy_aligned, verbose=False)
            all_params.append(x)
            all_scores.append(score)

        results_df = pd.DataFrame(all_params, columns=PARAM_NAMES)
        results_df['score'] = all_scores

        best_idx = np.argmax(all_scores)
        best_params = all_params[best_idx]
        best_score = all_scores[best_idx]
        # Enforce min <= max on best_params for clean display
        best_params_dict = {PARAM_NAMES[i]: int(best_params[i]) for i in range(len(best_params))}
        best_params_dict = enforce_min_max_for_dict(best_params_dict)
        best_params_clean = best_params_dict
        print(f"  Best Score: {best_score:.3f}")
        print(f"  Best Params: {best_params_clean}")

        # ---- Polynomial ----
        print("\nPolynomial Fitting...")
        optimum_per_param = fit_polynomial(results_df, PARAM_NAMES, POLYNOMIAL_DEGREE, SMOOTHING_WINDOW)
        fold_optimal = [optimum_per_param[p] for p in PARAM_NAMES]

        # The polynomial fit can suggest a min above the corresponding max
        # for a regime, since each parameter is optimised independently -
        # enforce min <= max before using these values downstream.
        fold_optimal_dict = {PARAM_NAMES[i]: fold_optimal[i] for i in range(len(fold_optimal))}
        fold_optimal_dict = enforce_min_max_for_dict(fold_optimal_dict)
        fold_optimal = [fold_optimal_dict[p] for p in PARAM_NAMES]

        fold_optimal_clean = {PARAM_NAMES[i]: fold_optimal[i] for i in range(len(fold_optimal))}
        print(f"  Optimal: {fold_optimal_clean}")

        # ---- Validate ----
        regime_dict = {
            'bull': {
                'lookback_days': fold_optimal[0],
                'drift_threshold': FIXED_DRIFT,
                'rebalance_min_days': fold_optimal[1],
                'rebalance_max_days': fold_optimal[2],
            },
            'bear': {
                'lookback_days': fold_optimal[3],
                'drift_threshold': FIXED_DRIFT,
                'rebalance_min_days': fold_optimal[4],
                'rebalance_max_days': fold_optimal[5],
            },
            'crash': {
                'lookback_days': fold_optimal[6],
                'drift_threshold': FIXED_DRIFT,
                'rebalance_min_days': fold_optimal[7],
                'rebalance_max_days': fold_optimal[8],
            }
        }
        # Extra safety
        regime_dict = enforce_min_max_for_regime_params(regime_dict)

        test_start = test_data.index[0].strftime('%Y-%m-%d')
        test_end = test_data.index[-1].strftime('%Y-%m-%d')
        test_metrics, test_trades = run_hmm_backtest(
            OPTIMAL_ETHICAL_PORTFOLIO, test_start, test_end, regime_dict, spy_aligned, verbose=False
        )

        fold_results.append({
            'fold': fold_idx+1,
            'optimal_params': regime_dict,
            'test_sharpe': test_metrics['sharpe_ratio'],
            'test_return': test_metrics['total_return'],
            'test_drawdown': test_metrics['max_drawdown'],
            'trades': test_trades,
        })

        print(f"\n  Fold {fold_idx+1} Results:")
        print(f"     Sharpe: {test_metrics['sharpe_ratio']:.3f}")
        print(f"     Return: {test_metrics['total_return']*100:.2f}%")
        print(f"     Drawdown: {test_metrics['max_drawdown']*100:.2f}%")
        print(f"     Trades: {test_trades}")

    # ---- Aggregate ----
    if len(fold_results) == 0:
        robust_params = {
            'bull_lookback': 405, 'bull_min': 110, 'bull_max': 130,
            'bear_lookback': 405, 'bear_min': 110, 'bear_max': 130,
            'crash_lookback': 405, 'crash_min': 110, 'crash_max': 130,
        }
    else:
        sharpe_list = [r['test_sharpe'] for r in fold_results]
        weights = np.maximum(sharpe_list, 0)
        if weights.sum() == 0:
            weights = np.ones(len(sharpe_list)) / len(sharpe_list)
        else:
            weights = weights / weights.sum()

        # Build param arrays from fold results
        param_arrays = {p: [] for p in PARAM_NAMES}
        for r in fold_results:
            for p in PARAM_NAMES:
                if p.startswith('bull_'):
                    regime = 'bull'
                    suffix = p.replace('bull_', '')
                elif p.startswith('bear_'):
                    regime = 'bear'
                    suffix = p.replace('bear_', '')
                elif p.startswith('crash_'):
                    regime = 'crash'
                    suffix = p.replace('crash_', '')
                else:
                    continue
                # Map suffix to actual dict key
                if suffix == 'lookback':
                    key = 'lookback_days'
                elif suffix == 'min':
                    key = 'rebalance_min_days'
                elif suffix == 'max':
                    key = 'rebalance_max_days'
                else:
                    continue
                param_arrays[p].append(r['optimal_params'][regime][key])

        robust_params = {}
        for p in PARAM_NAMES:
            weighted_sum = sum(weights[i] * param_arrays[p][i] for i in range(len(fold_results)))
            if p in ['bull_lookback', 'bear_lookback', 'crash_lookback',
                     'bull_min', 'bear_min', 'crash_min',
                     'bull_max', 'bear_max', 'crash_max']:
                robust_params[p] = int(round(weighted_sum / 5) * 5)
            else:
                robust_params[p] = round(weighted_sum, 3)

        # Enforce min <= max for the final robust params
        robust_params = enforce_min_max_for_dict(robust_params)

    # ---- Print ----
    print("\n" + "=" * 70)
    print("ROBUST PARAMETERS")
    print("=" * 70)
    for regime in ['bull', 'bear', 'crash']:
        print(f"\n{regime.capitalize()}:")
        print(f"  Lookback: {robust_params[f'{regime}_lookback']} days")
        print(f"  Rebalance Min: {robust_params[f'{regime}_min']} days")
        print(f"  Rebalance Max: {robust_params[f'{regime}_max']} days")
        print(f"  Drift: {FIXED_DRIFT*100:.1f}%")

    # ---- Final backtest ----
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE")
    print("=" * 70)
    full_regime_dict = {
        'bull': {
            'lookback_days': robust_params['bull_lookback'],
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': robust_params['bull_min'],
            'rebalance_max_days': robust_params['bull_max'],
        },
        'bear': {
            'lookback_days': robust_params['bear_lookback'],
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': robust_params['bear_min'],
            'rebalance_max_days': robust_params['bear_max'],
        },
        'crash': {
            'lookback_days': robust_params['crash_lookback'],
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': robust_params['crash_min'],
            'rebalance_max_days': robust_params['crash_max'],
        }
    }
    full_start = data.index[0].strftime('%Y-%m-%d')
    full_end = data.index[-1].strftime('%Y-%m-%d')
    full_metrics, full_trades = run_hmm_backtest(
        OPTIMAL_ETHICAL_PORTFOLIO, full_start, full_end, full_regime_dict, spy_aligned, verbose=True
    )

    print(f"\nSharpe: {full_metrics['sharpe_ratio']:.3f}")
    print(f"Return: {full_metrics['total_return']*100:.2f}%")
    print(f"Drawdown: {full_metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility: {full_metrics['annualised_volatility']*100:.2f}%")
    print(f"Trades: {full_trades}")

    # ---- Update config ----
    update_config(robust_params)

    print("\nComplete.")


if __name__ == "__main__":
    main()