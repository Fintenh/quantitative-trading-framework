"""
kelly_config.py
----------------
Kelly Criterion parameters from Phase 4 optimisation.
Optimal lookback found: 126 days
"""

# ============================================================================
# KELLY PARAMETERS (from Phase 4 optimisation)
# ============================================================================

KELLY_LOOKBACK = 126  # Optimal lookback for Kelly calculation
KELLY_BASE_CAP = 0.20          # 20% base cash cap
KELLY_MAX_CAP = 1.00           # 100% max cash cap when f* <= 0
KELLY_FRACTION = 1.00          # Full Kelly (0-1 for fractional)

# Kelly Cash Cap Rule:
#   If f* > 0:   cash_cap = KELLY_BASE_CAP (20%)
#   If f* <= 0:  cash_cap = KELLY_MAX_CAP (100%)

# Performance with this lookback:
#   Total Return: 1217.75%
#   Sharpe Ratio: 0.864
#   Max Drawdown: -26.76%