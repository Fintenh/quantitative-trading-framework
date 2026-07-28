"""
ethical_config.py
-----------------
Configuration for ethical portfolio backtesting.
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
# BACKTEST PERIODS
# ============================================================================

START_DATE = "2010-01-01"                              # Full backtest start
HOLDOUT_START = "2020-01-01"                          # Holdout period start
HOLDOUT_END = datetime.now().strftime("%Y-%m-%d")     # Both end here (today)

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

KELLY_BASE_CAP = 0.20    # Cash allocation when f* > 0 (positive signal)
KELLY_MAX_CAP = 1.00     # Cash allocation when f* <= 0 (negative signal)

# ============================================================================
# SELECTED PORTFOLIOS (From walk-forward validation)
# ============================================================================

OPTIMAL_STANDARD_PORTFOLIO = [
    "NEE",
    "NVDA",
    "LLY",
    "VWS.CO",
    "COST",
    "ADBE",
    "AVGO",
    "UNH",
    "AMZN",
    "NOW",
    "SHW",
    "BA",
    "NKE",
    "TSM",
    "GILD",
]

OPTIMAL_ETHICAL_PORTFOLIO = [
    "RSG",
    "NVDA",
    "VWS.CO",
    "ADBE",
    "LLY",
    "NEE",
    "NOW",
    "COST",
    "WM",
    "ENPH",
    "UNH",
    "FSLR",
    "MSFT",
    "GILD",
    "V",
]

# ============================================================================
# OPTIMISED PARAMETERS FOR EACH UNIVERSE
# ============================================================================

ORIGINAL_PARAMS = {
    'lookback_days': 290,
    'rebalance_min_days': 55,
    'rebalance_max_days': 85,
    'drift_threshold': 0.035,
    'take_profit_pct': 0.174,
    'cash_max_allocation': 0.20,
    'kelly_lookback': 126,
}

OPTIMAL_STANDARD_PARAMS = {
    'lookback_days': 450,
    'rebalance_min_days': 105,
    'rebalance_max_days': 135,
    'drift_threshold': 0.07,
    'take_profit_pct': 0.15,
    'cash_max_allocation': 0.2,
    'kelly_lookback': 150,
}

OPTIMAL_ETHICAL_PARAMS = {
    'lookback_days': 405,
    'rebalance_min_days': 110,
    'rebalance_max_days': 130,
    'drift_threshold': 0.085,
    'take_profit_pct': 0.203,
    'cash_max_allocation': 0.2,
    'kelly_lookback': 165,
}