"""
fractional_kelly_config.py
---------------------------
Fractional Kelly parameters from Phase 4 optimisation.
Optimal fraction found: 100%
"""

# ============================================================================
# Fractional Kelly parameters (from Phase 4 optimisation)
# ============================================================================

KELLY_LOOKBACK = 135              # Optimal lookback for Kelly calculation
KELLY_FRACTION = 1.00  # Optimal Kelly fraction
KELLY_BASE_CAP = 0.2               # 20% base cash cap
KELLY_MAX_CAP = 1.00                           # 100% max cash cap when f* <= 0

# Kelly Cash Cap Rule (with Fractional Kelly):
#   If f* > 0:   cash_cap = KELLY_BASE_CAP (20%)
#   If f* <= 0:  cash_cap = KELLY_BASE_CAP + KELLY_FRACTION * (1.0 - KELLY_BASE_CAP)
#
#   With KELLY_FRACTION = 100%:
#   Cash cap = 100%

# Performance with this configuration:
#   Total Return: 1067.68%
#   Sharpe Ratio: 1.200
#   Max Drawdown: -27.87%
#   Trades: 400
#   Sign Changes: 4
