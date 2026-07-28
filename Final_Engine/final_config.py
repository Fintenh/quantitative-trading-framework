"""
final_config.py - Final trading configuration for Optimal_Ethical portfolio.
All parameters optimised through Phase 5 walk-forward validation.
"""

import os
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================================
# ASSET UNIVERSE - OPTIMAL_ETHICAL (15 assets from ethical_config.py)
# ============================================================================

TICKERS = [
    "RSG",     # Waste Management - Republic Services
    "NVDA",    # Technology - AI leader
    "VWS.CO",  # Clean Energy - Vestas (Denmark)
    "ADBE",    # Technology - Adobe
    "LLY",     # Healthcare - Eli Lilly
    "NEE",     # Clean Energy - NextEra Energy
    "NOW",     # Technology - ServiceNow
    "COST",    # Consumer - Costco
    "WM",      # Waste Management - Waste Management
    "ENPH",    # Clean Energy - Enphase Energy
    "UNH",     # Healthcare - UnitedHealth
    "FSLR",    # Clean Energy - First Solar
    "MSFT",    # Technology - Microsoft
    "GILD",    # Healthcare - Gilead Sciences
    "V",       # Financials - Visa
]

# ============================================================================
# STRATEGY PARAMETERS (optimised via run_ethical_optimisation.py)
# ============================================================================

INITIAL_CAPITAL = 100.0
START_DATE = "2020-01-01"

# Optimised parameters for Optimal_Ethical (from ethical_config.py)
LOOKBACK_DAYS = 405
REBALANCE_MIN_DAYS = 110
REBALANCE_MAX_DAYS = 130
DRIFT_THRESHOLD = 0.085          # 8.5%
RELATIVE_TAKE_PROFIT_PCT = 0.203  # 20.3%

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

RISK_FREE_RATE = 0.045
CASH_MIN_VOLATILITY = 0.25
CASH_MAX_VOLATILITY = 0.50
CASH_MAX_ALLOCATION = 0.20
CASH_INTEREST_RATE = 0.0525

# ============================================================================
# KELLY PARAMETERS (optimised via kelly_lookback_optimiser_ethical.py)
# ============================================================================

KELLY_LOOKBACK = 320
KELLY_BASE_CAP = 0.20
KELLY_MAX_CAP = 1.00
KELLY_FRACTION = 1.00

# ============================================================================
# TRANSACTION COSTS (Trading 212)
# ============================================================================

FX_FEE_PCT = 0.0015
SPREAD_PCT = 0.0005
COMMISSION_PCT = 0.0
TRANSACTION_COST_PCT = FX_FEE_PCT + SPREAD_PCT + COMMISSION_PCT
ROUND_TRIP_COST_PCT = 2 * TRANSACTION_COST_PCT  # 0.4%

# ============================================================================
# LOGGING
# ============================================================================

PORTFOLIO_LOG_FILE = "portfolio_log.csv"
REBALANCE_LOG_FILE = "rebalance_log.csv"
LAST_REBALANCE_FILE = "last_rebalance.txt"
REBALANCE_DECISIONS_FILE = "rebalance_decisions.csv"
TAKE_PROFIT_LOG_FILE = "take_profit_log.csv"
LAST_UPDATE_DATE_FILE = "last_update_date.txt"