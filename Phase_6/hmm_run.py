"""
hmm_run.py - Regime-weighted backtest with Kelly cash allocation (Phase 5).
Uses the unified HMMEngineWithKelly from hmm_engine.py.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from io import StringIO
import contextlib

# ----- Paths -----
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, os.path.join(parent_dir, "Phase_5"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))
sys.path.insert(0, script_dir)

# ----- Imports -----
import hmm_config
from hmm_config import *
from backtest_engine import BacktestEngine
from ethical_backtest import BacktestEngineWithKelly
from hmm_engine import HMMEngineWithKelly
from performance_metrics import calculate_metrics

PERMANENT_REGIME_PARAMS = REGIME_PARAMS.copy()

print(f"DEBUG: Imported hmm_config from: {hmm_config.__file__}")
print(f"DEBUG: PERMANENT_REGIME_PARAMS bull = {PERMANENT_REGIME_PARAMS['bull']}")
print(f"DEBUG: PERMANENT_REGIME_PARAMS bear = {PERMANENT_REGIME_PARAMS['bear']}")
print(f"DEBUG: PERMANENT_REGIME_PARAMS crash = {PERMANENT_REGIME_PARAMS['crash']}")
print(f"DEBUG: HMMEngineWithKelly imported from: {HMMEngineWithKelly.__module__}")


# ---------------------------------------------------------------------------
# Suppress stdout (for GARCH)
# ---------------------------------------------------------------------------

def silence_stdout(func):
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(StringIO()):
            return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Baseline - uses Phase 5 Kelly engine (with take_profit=0)
# ---------------------------------------------------------------------------

class SilentBacktestEngineWithKelly(BacktestEngineWithKelly):
    @silence_stdout
    def _calc_cash_allocation(self, returns):
        return super()._calc_cash_allocation(returns)


def run_baseline():
    print("\nRunning baseline (Phase 5 No Rules with Kelly)...")
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        engine = SilentBacktestEngineWithKelly(
            tickers=OPTIMAL_ETHICAL_PORTFOLIO,
            start_date=TEST_START,
            end_date=TEST_END,
            initial_capital=INITIAL_CAPITAL,
            lookback_days=OPTIMAL_ETHICAL_PARAMS['lookback_days'],
            rebalance_min_days=OPTIMAL_ETHICAL_PARAMS['rebalance_min_days'],
            rebalance_max_days=OPTIMAL_ETHICAL_PARAMS['rebalance_max_days'],
            drift_threshold=OPTIMAL_ETHICAL_PARAMS['drift_threshold'],
            take_profit_pct=TAKE_PROFIT_PCT,
            risk_free_rate=RISK_FREE_RATE,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            cash_interest_rate=CASH_INTEREST_RATE,
            cash_min_volatility=CASH_MIN_VOLATILITY,
            cash_max_volatility=CASH_MAX_VOLATILITY,
            cash_max_allocation=OPTIMAL_ETHICAL_PARAMS['cash_max_allocation'],
            kelly_lookback=OPTIMAL_ETHICAL_PARAMS.get('kelly_lookback', 165),
            kelly_base_cap=KELLY_BASE_CAP,
            kelly_max_cap=KELLY_MAX_CAP,
        )
        results = engine.run()
        wealth = results['wealth_curve']['value']
        metrics = calculate_metrics(wealth, INITIAL_CAPITAL, RISK_FREE_RATE)

    finally:
        sys.stdout = old_stdout

    return wealth, metrics, len(results.get('trades', []))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("REGIME-WEIGHTED BACKTEST (Smooth Probabilities + Kelly)")
    print("=" * 70)

    baseline_wealth, baseline_metrics, baseline_trades = run_baseline()

    print("\nDownloading SPY data...")
    spy_data = yf.download("SPY", start=TEST_START, end=TEST_END, progress=False)["Close"]
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()

    print("\nRunning regime-weighted backtest (with Kelly)...")
    engine = HMMEngineWithKelly(
        tickers=OPTIMAL_ETHICAL_PORTFOLIO,
        start_date=TEST_START,
        end_date=TEST_END,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=OPTIMAL_ETHICAL_PARAMS['lookback_days'],
        rebalance_min_days=OPTIMAL_ETHICAL_PARAMS['rebalance_min_days'],
        rebalance_max_days=OPTIMAL_ETHICAL_PARAMS['rebalance_max_days'],
        drift_threshold=OPTIMAL_ETHICAL_PARAMS['drift_threshold'],
        take_profit_pct=TAKE_PROFIT_PCT,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=OPTIMAL_ETHICAL_PARAMS['cash_max_allocation'],
        kelly_lookback=OPTIMAL_ETHICAL_PARAMS.get('kelly_lookback', 165),
        kelly_base_cap=KELLY_BASE_CAP,
        kelly_max_cap=KELLY_MAX_CAP,
        spy_data=spy_data,
        regime_params=PERMANENT_REGIME_PARAMS,
        verbose=True,
        debug_probs=True,
    )
    results = engine.run()
    hmm_wealth = results['wealth_curve']['value']
    hmm_metrics = calculate_metrics(hmm_wealth, INITIAL_CAPITAL, RISK_FREE_RATE)
    hmm_trades = len(results.get('trades', []))

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("\n| Method | Sharpe | Return | Max DD | Trades |")
    print("|--------|--------|--------|--------|--------|")
    print(f"| Baseline | {baseline_metrics['sharpe_ratio']:.4f} | {baseline_metrics['total_return']*100:.2f}% | {baseline_metrics['max_drawdown']*100:.2f}% | {baseline_trades} |")
    print(f"| Regime  | {hmm_metrics['sharpe_ratio']:.4f} | {hmm_metrics['total_return']*100:.2f}% | {hmm_metrics['max_drawdown']*100:.2f}% | {hmm_trades} |")

    base = baseline_metrics
    hmm = hmm_metrics
    print("\nRegime vs Baseline:")
    print(f"  Sharpe:  {hmm['sharpe_ratio'] - base['sharpe_ratio']:+.4f}")
    print(f"  Return:  {(hmm['total_return'] - base['total_return']) * 100:+.2f}%")
    print(f"  Max DD:  {(hmm['max_drawdown'] - base['max_drawdown']) * 100:+.2f}%")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(baseline_wealth.index, baseline_wealth, label='Baseline (Phase 5 No Rules + Kelly)', color='#1f77b4')
    ax.plot(hmm_wealth.index, hmm_wealth, label='Regime-Weighted (Smooth + Kelly)', color='#ff7f0e')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (£)')
    ax.set_title('Equity Curves (No Rules)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hmm_backtest_comparison.png'), dpi=150)
    plt.show()
    print(f"\nPlot saved to {FIGURES_DIR}/hmm_backtest_comparison.png")


if __name__ == "__main__":
    main()