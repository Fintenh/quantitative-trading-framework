"""
hmm_overfit_demo.py - Demonstrates overfitting by optimising on a single period
and testing on a separate holdout period.
Plots the equity curves for comparison.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Integer
import contextlib
from io import StringIO
import pprint

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
from hmm_engine import HMMEngineWithKelly
from performance_metrics import calculate_metrics

# ---- User-defined training period ----
# Change these dates to test different training periods
TRAIN_START = "1993-01-01"
TRAIN_END = "2020-01-01"

# Test period is always 2020 to today (out-of-sample)
TEST_START = "2020-01-01"
TEST_END = datetime.now().strftime("%Y-%m-%d")

# ---- Fixed parameters ----
FIXED_DRIFT = 0.085

# ---- Parameter space (expanded to allow overfitting) ----
PARAM_SPACE = [
    Integer(200, 600, name='bull_lookback'),
    Integer(40, 160, name='bull_min'),
    Integer(80, 220, name='bull_max'),
    Integer(150, 500, name='bear_lookback'),
    Integer(30, 160, name='bear_min'),
    Integer(60, 200, name='bear_max'),
    Integer(200, 600, name='crash_lookback'),
    Integer(30, 160, name='crash_min'),
    Integer(60, 200, name='crash_max'),
]
PARAM_NAMES = [p.name for p in PARAM_SPACE]

# ---- Optimisation settings ----
N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15

# ---------------------------------------------------------------------------
# Helper: run backtest
# ---------------------------------------------------------------------------

def run_hmm_backtest(tickers, start_date, end_date, regime_params, spy_data, verbose=False):
    """Run HMM backtest and return metrics + equity curve."""
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
    return metrics, len(results.get('trades', [])), wealth

# ---------------------------------------------------------------------------
# Objective for Bayesian optimisation
# ---------------------------------------------------------------------------

def objective(params, tickers, train_data, spy_data, verbose=False):
    (bull_lookback, bull_min, bull_max,
     bear_lookback, bear_min, bear_max,
     crash_lookback, crash_min, crash_max) = params

    # Enforce min <= max
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
    metrics, trades, _ = run_hmm_backtest(
        tickers, start_date, end_date, regime_params, spy_data, verbose=verbose
    )

    # ---- Objective: 70% Return + 30% Sharpe ----
    total_return = min(max(metrics['total_return'], -1), 2)
    return_score = (total_return + 1) / 3
    sharpe = min(max(metrics['sharpe_ratio'], -2), 3)
    sharpe_score = (sharpe + 2) / 5
    score = 0.70 * return_score + 0.30 * sharpe_score

    if verbose:
        print(f"  Return: {total_return*100:.2f}%, Sharpe: {sharpe:.4f}, Score: {score:.4f}")
    return -score

# ---------------------------------------------------------------------------
# Download data
# ---------------------------------------------------------------------------

print("=" * 70)
print("OVERFITTING DEMONSTRATION WITH FUTURE OPTIMISATION")
print("=" * 70)
print(f"Training period (past): {TRAIN_START} to {TRAIN_END}")
print(f"Test period (future):   {TEST_START} to {TEST_END}")
print("=" * 70)

print("\nDownloading data...")
data = yf.download(OPTIMAL_ETHICAL_PORTFOLIO, start=TRAIN_START, end=TEST_END, progress=False)["Close"]
if isinstance(data, pd.DataFrame):
    data = data.dropna(how='all', axis=1)

print("Downloading SPY...")
spy_data = yf.download("SPY", start=TRAIN_START, end=TEST_END, progress=False)["Close"]
if isinstance(spy_data, pd.DataFrame):
    spy_data = spy_data.squeeze()

# Align SPY with data
spy_aligned = spy_data.reindex(data.index).ffill().bfill()
if isinstance(spy_aligned, pd.DataFrame):
    spy_aligned = spy_aligned.squeeze()

# Split into train and test
train_data = data[data.index < pd.Timestamp(TRAIN_END)]
test_data = data[data.index >= pd.Timestamp(TEST_START)]

print(f"\nTotal data: {len(data)} days")
print(f"Training data: {len(train_data)} days")
print(f"Testing data: {len(test_data)} days")
print("=" * 70)

# ---------------------------------------------------------------------------
# Function: run optimisation on a period
# ---------------------------------------------------------------------------

def run_optimisation(period_data, period_name, spy_data):
    """Run Bayesian optimisation on a given period and return the optimal regime dict."""
    print(f"\n--- Optimising on {period_name} ({period_data.index[0].strftime('%Y-%m-%d')} to {period_data.index[-1].strftime('%Y-%m-%d')}) ---")
    print(f"Data points: {len(period_data)}")

    def obj_wrapper(params):
        return objective(params, OPTIMAL_ETHICAL_PORTFOLIO, period_data, spy_data, verbose=False)

    result = gp_minimize(
        obj_wrapper,
        PARAM_SPACE,
        n_calls=N_BAYESIAN_CALLS,
        n_initial_points=N_INITIAL_POINTS,
        random_state=42,
        verbose=False
    )

    best_params = result.x
    best_score = -result.fun
    best_params_clean = {PARAM_NAMES[i]: int(best_params[i]) for i in range(len(best_params))}
    print(f"  Best Score: {best_score:.3f}")
    print(f"  Best Params: {best_params_clean}")

    # Build regime dict
    regime_dict = {
        'bull': {
            'lookback_days': int(best_params[0]),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': int(best_params[1]),
            'rebalance_max_days': int(best_params[2]),
        },
        'bear': {
            'lookback_days': int(best_params[3]),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': int(best_params[4]),
            'rebalance_max_days': int(best_params[5]),
        },
        'crash': {
            'lookback_days': int(best_params[6]),
            'drift_threshold': FIXED_DRIFT,
            'rebalance_min_days': int(best_params[7]),
            'rebalance_max_days': int(best_params[8]),
        }
    }

    return regime_dict, best_score

# ---------------------------------------------------------------------------
# Run optimisation on past (training) period
# ---------------------------------------------------------------------------

past_regime, past_score = run_optimisation(train_data, "Past (Training)", spy_aligned)

# ---------------------------------------------------------------------------
# Run optimisation on future (test) period - this is cheating (look-ahead bias)
# ---------------------------------------------------------------------------

future_regime, future_score = run_optimisation(test_data, "Future (Test - CHEATING)", spy_aligned)

# ---------------------------------------------------------------------------
# Test on training period with past-optimised parameters (in-sample)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("TESTING PAST-OPTIMISED PARAMETERS ON PAST PERIOD (IN-SAMPLE)")
print("=" * 70)

past_train_metrics, past_train_trades, past_train_wealth = run_hmm_backtest(
    OPTIMAL_ETHICAL_PORTFOLIO,
    TRAIN_START,
    TRAIN_END,
    past_regime,
    spy_aligned,
    verbose=False
)

print(f"Return: {past_train_metrics['total_return']*100:.2f}%")
print(f"Sharpe: {past_train_metrics['sharpe_ratio']:.3f}")
print(f"Drawdown: {past_train_metrics['max_drawdown']*100:.2f}%")
print(f"Trades: {past_train_trades}")

# ---------------------------------------------------------------------------
# Test past-optimised parameters on test period (out-of-sample)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("TESTING PAST-OPTIMISED PARAMETERS ON TEST PERIOD (OUT-OF-SAMPLE)")
print("=" * 70)

past_test_metrics, past_test_trades, past_test_wealth = run_hmm_backtest(
    OPTIMAL_ETHICAL_PORTFOLIO,
    TEST_START,
    TEST_END,
    past_regime,
    spy_aligned,
    verbose=False
)

print(f"Return: {past_test_metrics['total_return']*100:.2f}%")
print(f"Sharpe: {past_test_metrics['sharpe_ratio']:.3f}")
print(f"Drawdown: {past_test_metrics['max_drawdown']*100:.2f}%")
print(f"Trades: {past_test_trades}")

# ---------------------------------------------------------------------------
# Test future-optimised parameters on test period (cheating, look-ahead)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("TESTING FUTURE-OPTIMISED PARAMETERS ON TEST PERIOD (CHEATING - LOOK-AHEAD)")
print("=" * 70)

future_test_metrics, future_test_trades, future_test_wealth = run_hmm_backtest(
    OPTIMAL_ETHICAL_PORTFOLIO,
    TEST_START,
    TEST_END,
    future_regime,
    spy_aligned,
    verbose=False
)

print(f"Return: {future_test_metrics['total_return']*100:.2f}%")
print(f"Sharpe: {future_test_metrics['sharpe_ratio']:.3f}")
print(f"Drawdown: {future_test_metrics['max_drawdown']*100:.2f}%")
print(f"Trades: {future_test_trades}")

# ---------------------------------------------------------------------------
# Run baseline on test period for comparison
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE ON TEST PERIOD (FOR COMPARISON)")
print("=" * 70)

# Use the Phase 5 fixed parameters
baseline_regime = {
    'bull': {
        'lookback_days': 405,
        'drift_threshold': 0.085,
        'rebalance_min_days': 110,
        'rebalance_max_days': 130,
    },
    'bear': {
        'lookback_days': 405,
        'drift_threshold': 0.085,
        'rebalance_min_days': 110,
        'rebalance_max_days': 130,
    },
    'crash': {
        'lookback_days': 405,
        'drift_threshold': 0.085,
        'rebalance_min_days': 110,
        'rebalance_max_days': 130,
    }
}

baseline_metrics, baseline_trades, baseline_wealth = run_hmm_backtest(
    OPTIMAL_ETHICAL_PORTFOLIO,
    TEST_START,
    TEST_END,
    baseline_regime,
    spy_aligned,
    verbose=False
)

print(f"Return: {baseline_metrics['total_return']*100:.2f}%")
print(f"Sharpe: {baseline_metrics['sharpe_ratio']:.3f}")
print(f"Drawdown: {baseline_metrics['max_drawdown']*100:.2f}%")
print(f"Trades: {baseline_trades}")

# ---------------------------------------------------------------------------
# Plot all equity curves on the test period
# ---------------------------------------------------------------------------

print("\nGenerating equity curve plot...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot baseline
ax.plot(baseline_wealth.index, baseline_wealth, label='Baseline (Phase 5)', color='#1f77b4', linewidth=2)

# Plot past-optimised on test period
ax.plot(past_test_wealth.index, past_test_wealth, label='Past-Optimised (1993-2020)', color='#ff7f0e', linewidth=2, linestyle='--')

# Plot future-optimised on test period (cheating)
ax.plot(future_test_wealth.index, future_test_wealth, label='Future-Optimised (2020-2026 - CHEATING)', color='#2ca02c', linewidth=2, linestyle='-.')

ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (£)')
ax.set_title(f'Overfitting Demonstration: Past vs Future Optimisation\nTest Period: 2020-{TEST_END[:4]}')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Add text box with metrics
text = (
    f"Baseline: Return={baseline_metrics['total_return']*100:.1f}%, Sharpe={baseline_metrics['sharpe_ratio']:.3f}\n"
    f"Past-Opt: Return={past_test_metrics['total_return']*100:.1f}%, Sharpe={past_test_metrics['sharpe_ratio']:.3f}\n"
    f"Future-Opt: Return={future_test_metrics['total_return']*100:.1f}%, Sharpe={future_test_metrics['sharpe_ratio']:.3f}"
)
ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'overfit_demo.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"Plot saved to {FIGURES_DIR}/overfit_demo.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'Metric':<20} {'Past-Opt (In-Sample)':<20} {'Past-Opt (Out-of-Sample)':<20} {'Future-Opt (Cheating)':<20} {'Baseline':<15}")
print("-" * 95)
print(f"{'Return':<20} {past_train_metrics['total_return']*100:<20.2f}% {past_test_metrics['total_return']*100:<20.2f}% {future_test_metrics['total_return']*100:<20.2f}% {baseline_metrics['total_return']*100:<15.2f}%")
print(f"{'Sharpe':<20} {past_train_metrics['sharpe_ratio']:<20.3f} {past_test_metrics['sharpe_ratio']:<20.3f} {future_test_metrics['sharpe_ratio']:<20.3f} {baseline_metrics['sharpe_ratio']:<15.3f}")
print(f"{'Drawdown':<20} {past_train_metrics['max_drawdown']*100:<20.2f}% {past_test_metrics['max_drawdown']*100:<20.2f}% {future_test_metrics['max_drawdown']*100:<20.2f}% {baseline_metrics['max_drawdown']*100:<15.2f}%")
print(f"{'Trades':<20} {past_train_trades:<20} {past_test_trades:<20} {future_test_trades:<20} {baseline_trades:<15}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print(f"\nPast-optimised parameters (1993-2020) achieved {past_train_metrics['total_return']*100:.2f}% return in-sample")
print(f"   but only {past_test_metrics['total_return']*100:.2f}% return out-of-sample on 2020-{TEST_END[:4]}.")
print(f"   Drop of {past_train_metrics['total_return']*100 - past_test_metrics['total_return']*100:.2f} percentage points.")

print(f"\nFuture-optimised parameters (2020-{TEST_END[:4]}) achieved {future_test_metrics['total_return']*100:.2f}% return")
print(f"   on the same period, {future_test_metrics['total_return']*100 - past_test_metrics['total_return']*100:.2f} percentage points higher.")

print(f"\nThe baseline achieved {baseline_metrics['total_return']*100:.2f}% return, outperforming the past-optimised strategy.")
print(f"   Baseline is {baseline_metrics['total_return']*100 - past_test_metrics['total_return']*100:.2f} percentage points better.")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("\nThe future-optimised parameters (which we couldn't have known in advance)")
print("outperform the past-optimised parameters on the test period.")
print("\nThis demonstrates the fundamental challenge of regime-switching strategies:")
print("  - Parameters that work well on historical data may not work well in the future.")
print("  - The regime state alone does not predict which parameters will be optimal.")
print("  - Without knowing the future market regime, we cannot select the right parameters.")
print("\nThis is why the regime-switching approach has underperformed the fixed baseline.")
print("The baseline (Phase 5 parameters) is a better choice because it is:")
print("  1. Simpler and more robust")
print("  2. Not dependent on regime detection accuracy")
print("  3. Optimised over a longer period with more market cycles")
print("=" * 70)

print("\nTo change the training period, edit TRAIN_START and TRAIN_END at the top of this file.")
print("Complete.")