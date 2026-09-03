"""
hmm_config.py - Configuration for HMM probability-weighted strategy.
All regimes set to the general optimal parameters (Ethical baseline).
"""

import os
from datetime import datetime

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

HMM_MODEL_PATH = os.path.join(LOGS_DIR, "hmm_model.pkl")
HMM_STATE_MAP_PATH = os.path.join(LOGS_DIR, "hmm_state_map.pkl")

# --- Data periods ---
TRAIN_START = "1990-01-01"
TRAIN_END = "2015-12-31"
TEST_START = "2020-01-01"
TEST_END = datetime.now().strftime("%Y-%m-%d")
#TEST_START = "2010-01-04"
#TEST_END = "2018-10-16"

# --- HMM settings ---
N_STATES = 3
COVARIANCE_TYPE = "full"
N_ITER = 1000
RANDOM_STATE = 42

# --- Regime parameters (all identical to Phase 5 Ethical optimal) ---
REGIME_PARAMS = {   'bull': {   'lookback_days': 600,
                'rebalance_min_days': 150,
                'rebalance_max_days': 210,
                'drift_threshold': 0.085},
    'bear': {   'lookback_days': 440,
                'rebalance_min_days': 55,
                'rebalance_max_days': 85,
                'drift_threshold': 0.085},
    'crash': {   'lookback_days': 380,
                 'rebalance_min_days': 80,
                 'rebalance_max_days': 185,
                 'drift_threshold': 0.085}}

# --- Blending weight ---
ALPHA = 0.920

# --- Smoothing window ---
SMOOTH_WINDOW = 7

# --- Risk parameters (fixed, from Phase 5) ---
RISK_FREE_RATE = 0.045
TRANSACTION_COST_PCT = 0.004
CASH_INTEREST_RATE = 0.0525
CASH_MIN_VOLATILITY = 0.25
CASH_MAX_VOLATILITY = 0.50
CASH_MAX_ALLOCATION = 0.20
INITIAL_CAPITAL = 100.0
TAKE_PROFIT_PCT = 0.0
KELLY_BASE_CAP = 0.20
KELLY_MAX_CAP = 1.00

# --- Portfolio (import from Phase 5) ---
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, "Phase_5"))
from ethical_config import OPTIMAL_ETHICAL_PORTFOLIO, OPTIMAL_ETHICAL_PARAMS