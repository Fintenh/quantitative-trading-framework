"""
run_robust_optimisation.py - Run the complete robust optimisation pipeline.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from config_optimised import *
import yfinance as yf
import pandas as pd
from backtest_engine import BacktestEngine
from robust_optimiser import RobustOptimiser
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Logs directory: {logs_dir}")
print(f"Figures directory: {figures_dir}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Parameter search space
PARAM_SPACE = {
    'lookback_days': [200, 300],
    'rebalance_min_days': [20, 60],
    'rebalance_max_days': [60, 120],
    'drift_threshold': [0.01, 0.06],
    'take_profit_pct': [0.15, 0.45],
}

# Optimisation settings
N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15
N_SPLITS = 5
POLYNOMIAL_DEGREE = 4
SMOOTHING_WINDOW = 11

# ---------------------------------------------------------------------------
# Download data
# ---------------------------------------------------------------------------

print("\nDownloading data...")
data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]
print(f"Downloaded {len(data)} days from {data.index[0]} to {data.index[-1]}")

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING ROBUST OPTIMISATION")
print("=" * 70)

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

results = optimiser.run(data, TICKERS, n_splits=N_SPLITS)

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

print("\nSaving results...")

# Fold results
fold_df = optimiser.get_results_dataframe()
fold_df.to_csv(os.path.join(logs_dir, "optimisation_fold_results.csv"), index=False)
print(f"Fold results: optimisation_fold_results.csv")

# Robust parameters
params_df = pd.DataFrame([results['robust_params']])
params_df.to_csv(os.path.join(logs_dir, "robust_parameters.csv"), index=False)
print(f"Robust parameters: robust_parameters.csv")

# Final performance
perf_df = pd.DataFrame([results['final_performance']])
perf_df.to_csv(os.path.join(logs_dir, "final_performance.csv"), index=False)
print(f"Final performance: final_performance.csv")

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print("\nRobust Parameters:")
for key, val in results['robust_params'].items():
    if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
        print(f"  {key}: {val} days")
    elif key == 'drift_threshold':
        print(f"  {key}: {val*100:.2f}%")
    else:
        print(f"  {key}: {val*100:.1f}%")

print(f"\nRobustness Score: {results['robustness_score']:.1%}")

print("\nFinal Performance:")
perf = results['final_performance']
print(f"  Sharpe: {perf['sharpe_ratio']:.3f}")
print(f"  Return: {perf['total_return']*100:.2f}%")
print(f"  Drawdown: {perf['max_drawdown']*100:.2f}%")
print(f"  Volatility: {perf['annualised_volatility']*100:.2f}%")
print(f"  Trades: {perf['num_trades']}")

print("\nFold Results:")
for fold in results['fold_results']:
    print(f"  Fold {fold['fold']}: Sharpe={fold['test_sharpe']:.3f}, "
          f"Return={fold['test_return']*100:.2f}%, "
          f"Drawdown={fold['test_drawdown']*100:.2f}%")

# ---------------------------------------------------------------------------
# Generate summary plot
# ---------------------------------------------------------------------------

print("\nGenerating summary plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: fold Sharpe ratios
ax = axes[0, 0]
ax.bar(fold_df['fold'], fold_df['sharpe'], color='blue', alpha=0.7)
ax.axhline(y=0, color='black', lw=0.5)
ax.set_xlabel('Fold')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Out-of-Sample Sharpe by Fold')
ax.grid(True, alpha=0.3)

# Plot 2: fold returns
ax = axes[0, 1]
ax.bar(fold_df['fold'], fold_df['return']*100, color='green', alpha=0.7)
ax.axhline(y=0, color='black', lw=0.5)
ax.set_xlabel('Fold')
ax.set_ylabel('Return (%)')
ax.set_title('Out-of-Sample Return by Fold')
ax.grid(True, alpha=0.3)

# Plot 3: parameter stability
ax = axes[1, 0]
for param in ['lookback', 'rebalance_min', 'rebalance_max']:
    ax.plot(fold_df['fold'], fold_df[param], 'o-', label=param.replace('_', ' ').title())
ax.set_xlabel('Fold')
ax.set_ylabel('Days')
ax.set_title('Parameter Stability Across Folds')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: parameter consistency (normalised)
ax = axes[1, 1]
robust = results['robust_params']
param_names = ['lookback_days', 'drift_threshold', 'take_profit_pct']
display_names = ['Lookback', 'Drift', 'Take Profit']

for i, param in enumerate(param_names):
    values = [r['optimal_params'][param] for r in results['fold_results']]
    robust_val = robust[param]
    values_norm = [v / robust_val if robust_val > 0 else 1 for v in values]
    ax.plot(fold_df['fold'], values_norm, 'o-', label=display_names[i])

ax.axhline(y=1.0, color='red', ls='--', lw=1, label='Robust Value')
ax.set_xlabel('Fold')
ax.set_ylabel('Parameter / Robust Value')
ax.set_title('Parameter Consistency (Normalised)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "optimisation_summary.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print(f"Summary plot: optimisation_summary.png")

# ---------------------------------------------------------------------------
# Generate config file
# ---------------------------------------------------------------------------

print("\nGenerating config file...")

config_content = f'''"""
config_optimised.py - Configuration generated by robust optimisation.
These are the optimal parameters found through walk-forward validation.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Asset universe
TICKERS = {TICKERS}

# Data period
START_DATE = "{START_DATE}"
END_DATE = "{END_DATE}"
INITIAL_CAPITAL = 100.0

# ---------------------------------------------------------------------------
# Optimal parameters (from robust optimisation)
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = {results['robust_params']['lookback_days']}
REBALANCE_MIN_DAYS = {results['robust_params']['rebalance_min_days']}
REBALANCE_MAX_DAYS = {results['robust_params']['rebalance_max_days']}
DRIFT_THRESHOLD = {results['robust_params']['drift_threshold']}
RELATIVE_TAKE_PROFIT_PCT = {results['robust_params']['take_profit_pct']}

RISK_FREE_RATE = {RISK_FREE_RATE}
TRANSACTION_COST_PCT = {TRANSACTION_COST_PCT}
CASH_MIN_VOLATILITY = {CASH_MIN_VOLATILITY}
CASH_MAX_VOLATILITY = {CASH_MAX_VOLATILITY}
CASH_MAX_ALLOCATION = {CASH_MAX_ALLOCATION}
DATA_BUFFER_DAYS = {DATA_BUFFER_DAYS}
CASH_INTEREST_RATE = {CASH_INTEREST_RATE}

# Logging
PORTFOLIO_LOG_FILE = "portfolio_log.csv"
REBALANCE_LOG_FILE = "rebalance_log.csv"
LAST_REBALANCE_FILE = "last_rebalance.txt"
REBALANCE_DECISIONS_FILE = "rebalance_decisions.csv"

# ---------------------------------------------------------------------------
# Optimisation metadata
# ---------------------------------------------------------------------------

# Parameters found using:
# - Walk-Forward Cross-Validation ({N_SPLITS} folds)
# - Bayesian Optimisation ({N_BAYESIAN_CALLS} calls per fold)
# - Polynomial Fitting (degree {POLYNOMIAL_DEGREE})
# - Robustness Testing (score: {results['robustness_score']:.1%})
# - Cash Interest Rate: {CASH_INTEREST_RATE*100:.2f}% AER (compounded daily)

# Fold Results:
'''

for fold in results['fold_results']:
    config_content += f"#   Fold {fold['fold']}: Sharpe={fold['test_sharpe']:.3f}, Return={fold['test_return']*100:.2f}%, DD={fold['test_drawdown']*100:.2f}%\n"

config_path = os.path.join(script_dir, "config_optimised.py")
with open(config_path, "w") as f:
    f.write(config_content)

print(f"Config file: config_optimised.py")

# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("OPTIMISATION COMPLETE")
print("=" * 70)
print("\nTo use these parameters, copy config_optimised.py to config.py")
print("or import from config_optimised.py in your backtest scripts.")