"""
ethical_config.py
-----------------
Configuration for ethical portfolio backtesting and optimisation.
Parameters for Original_18, Optimal_Standard (11 assets), and Optimal_Ethical (11 assets).
"""

import os
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# BACKTEST PERIOD
# ============================================================================

START_DATE = "2010-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")
HOLDOUT_START = "2020-01-01"
HOLDOUT_END = datetime.now().strftime("%Y-%m-%d")

# ============================================================================
# GENERAL PARAMETERS
# ============================================================================

INITIAL_CAPITAL = 100.0
RISK_FREE_RATE = 0.045
TRANSACTION_COST_PCT = 0.004
CASH_INTEREST_RATE = 0.0525
CASH_MIN_VOLATILITY = 0.25
CASH_MAX_VOLATILITY = 0.50
CASH_MAX_ALLOCATION = 0.20

# ============================================================================
# KELLY PARAMETERS
# ============================================================================

# Original_18 Kelly lookback (from Phase 4 kelly_lookback_optimiser.py)
KELLY_LOOKBACK_ORIGINAL = 126

# Optimal_Standard and Optimal_Ethical Kelly lookback (from kelly_lookback_optimiser_ethical.py)
KELLY_LOOKBACK_OPTIMAL = 280

# Default Kelly parameters
KELLY_BASE_CAP = 0.20
KELLY_MAX_CAP = 1.00
KELLY_FRACTION = 1.00

# ============================================================================
# OPTIMISED PARAMETERS FOR EACH UNIVERSE
# ============================================================================

# Original_18 (from Phase 3 config_optimised.py)
ORIGINAL_PARAMS = {
    'lookback_days': 290,
    'rebalance_min_days': 55,
    'rebalance_max_days': 85,
    'drift_threshold': 0.035,
    'take_profit_pct': 0.174,
    'cash_max_allocation': 0.20,
    'kelly_lookback': KELLY_LOOKBACK_ORIGINAL,  # 126 days
}

# Optimal_Standard (11 assets - from Growth-Focused Selection)
OPTIMAL_STANDARD_PARAMS = {
    'lookback_days': 275,
    'rebalance_min_days': 110,
    'rebalance_max_days': 155,
    'drift_threshold': 0.078,
    'take_profit_pct': 0.303,
    'cash_max_allocation': 0.20,
    'kelly_lookback': KELLY_LOOKBACK_OPTIMAL,  # 280 days
}

# Optimal_Ethical (11 assets - from Growth-Focused Selection)
OPTIMAL_ETHICAL_PARAMS = {
    'lookback_days': 300,
    'rebalance_min_days': 120,
    'rebalance_max_days': 135,
    'drift_threshold': 0.005,
    'take_profit_pct': 0.256,
    'cash_max_allocation': 0.20,
    'kelly_lookback': KELLY_LOOKBACK_OPTIMAL,  # 280 days
}

# ============================================================================
# OPTIMISATION PARAMETER SPACE
# ============================================================================

# Expanded parameter space for 11-asset portfolios
PARAM_SPACE = {
    'lookback_days': [200, 500],
    'rebalance_min_days': [30, 120],
    'rebalance_max_days': [120, 200],
    'drift_threshold': [0.005, 0.10],
    'take_profit_pct': [0.15, 0.60],
}

# ============================================================================
# OPTIMISATION SETTINGS
# ============================================================================

N_BAYESIAN_CALLS = 80
N_INITIAL_POINTS = 15
N_SPLITS = 5
POLYNOMIAL_DEGREE = 4
SMOOTHING_WINDOW = 11

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Colour scheme for ethical comparison
COLORS = {
    'original': '#1f77b4',      # Blue
    'standard': '#ff7f0e',      # Orange
    'ethical': '#2ca02c',       # Green
    'spy': '#d62728',           # Red
}

# Universe names for display
UNIVERSE_NAMES = {
    'original': 'Original_18',
    'standard': 'Optimal_Standard',
    'ethical': 'Optimal_Ethical'
}

# ============================================================================
# LOGGING
# ============================================================================

PORTFOLIO_LOG_FILE = "portfolio_log.csv"
REBALANCE_LOG_FILE = "rebalance_log.csv"
LAST_REBALANCE_FILE = "last_rebalance.txt"
REBALANCE_DECISIONS_FILE = "rebalance_decisions.csv"
TAKE_PROFIT_LOG_FILE = "take_profit_log.csv"
LAST_UPDATE_DATE_FILE = "last_update_date.txt"