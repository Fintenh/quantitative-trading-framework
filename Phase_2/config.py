"""
config.py - Central configuration for the Phase 2 trading system.
Parameters are the optimised values from the Phase 3 walk-forward validation.
"""

import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Trading parameters
# ---------------------------------------------------------------------------

INITIAL_CAPITAL = 100.0
START_DATE = "2020-01-01"

# Asset universe (18 assets across sectors)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",  # Tech
    "JPM", "JNJ", "XOM",                                      # Financials/Healthcare/Energy
    "SPY", "QQQ", "EFA", "EEM",                               # ETFs
    "TLT", "LQD", "GLD", "DBC"                                # Bonds/Commodities
]

# ---------------------------------------------------------------------------
# Data & lookback
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 280
DATA_BUFFER_DAYS = 400

# ---------------------------------------------------------------------------
# Rebalancing (time + drift)
# ---------------------------------------------------------------------------

REBALANCE_MIN_DAYS = 60
REBALANCE_MAX_DAYS = 120
DRIFT_THRESHOLD = 0.025          # 2.5% drift triggers rebalance

# ---------------------------------------------------------------------------
# Risk management
# ---------------------------------------------------------------------------

RISK_FREE_RATE = 0.045           # 4.5% (current UK base rate)
RELATIVE_TAKE_PROFIT_PCT = 0.25  # 25% above SPY

# ---------------------------------------------------------------------------
# Cash management
# ---------------------------------------------------------------------------

CASH_MIN_VOLATILITY = 0.25       # Below this: 0% cash
CASH_MAX_VOLATILITY = 0.50       # Above this: max cash
CASH_MAX_ALLOCATION = 0.20       # Maximum cash allocation (20%)
CASH_INTEREST_RATE = 0.0525      # 5.25% AER (UK base rate)

# ---------------------------------------------------------------------------
# Kelly criterion (from Phase 4 optimisation)
# ---------------------------------------------------------------------------

KELLY_LOOKBACK = 126              # Optimal lookback for Kelly calculation
KELLY_BASE_CAP = 0.20             # 20% base cash cap (when f* > 0)
KELLY_MAX_CAP = 1.00              # 100% max cash cap (when f* <= 0)

# ---------------------------------------------------------------------------
# Transaction costs (Trading 212)
# ---------------------------------------------------------------------------

FX_FEE_PCT = 0.0015              # 0.15% FX fee
SPREAD_PCT = 0.0005              # 0.05% spread
COMMISSION_PCT = 0.0             # No commission
TRANSACTION_COST_PCT = FX_FEE_PCT + SPREAD_PCT + COMMISSION_PCT
ROUND_TRIP_COST_PCT = 2 * TRANSACTION_COST_PCT  # 0.4% round-trip

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

PORTFOLIO_LOG_FILE = "portfolio_log.csv"
REBALANCE_LOG_FILE = "rebalance_log.csv"
LAST_REBALANCE_FILE = "last_rebalance.txt"
REBALANCE_DECISIONS_FILE = "rebalance_decisions.csv"
TAKE_PROFIT_LOG_FILE = "take_profit_log.csv"
LAST_UPDATE_DATE_FILE = "last_update_date.txt"  # Tracks last portfolio update date