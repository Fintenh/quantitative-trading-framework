"""
ethical_universe.py
-------------------
Defines the standard, ethical, and original asset universes for comparison.
Original universe = 18 assets from Phases 1-4 (benchmark to verify code)
Standard and Ethical universes have been replaced with optimal 20-asset portfolios
selected using advanced mathematical methods (Efficient Frontier and Genetic Algorithm).
"""

# ============================================================================
# ORIGINAL ASSET UNIVERSE (18 assets from Phases 1-4)
# This is our benchmark to verify the code is working correctly
# ============================================================================

ORIGINAL_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    # Financials/Healthcare/Energy
    "JPM", "JNJ", "XOM",
    # ETFs
    "SPY", "QQQ", "EFA", "EEM",
    # Bonds/Commodities
    "TLT", "LQD", "GLD", "DBC"
]

# ============================================================================
# NEW OPTIMAL STANDARD UNIVERSE (11 assets from Growth-Focused Selection)
# Sharpe: 1.421 | Return: 27.1% | Volatility: 15.9%
# ============================================================================

STANDARD_UNIVERSE = [
    "AMD", "AAPL", "CAT", "JNJ", "UUP", 
    "NVDA", "GLD", "ABBV", "COST", "WMT", "NEE"
]

# ============================================================================
# NEW OPTIMAL ETHICAL UNIVERSE (11 assets from Growth-Focused Selection)
# Sharpe: 1.338 | Return: 31.9% | Volatility: 20.4%
# ============================================================================

ETHICAL_UNIVERSE = [
    "KO", "AAPL", "UNH", "JNJ", "NVDA", 
    "WMT", "TSLA", "ABBV", "ENPH", "COST", "TLT"
]

# ============================================================================
# SECTOR MAPPING (Updated for new tickers)
# ============================================================================

SECTOR_MAPPING = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "ORCL": "Technology", "IBM": "Technology", "AMD": "Technology",
    
    # Consumer
    "WMT": "Consumer", "COST": "Consumer", "KO": "Consumer",
    
    # Healthcare
    "ABBV": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare",
    "UNH": "Healthcare",
    
    # Industrials
    "CAT": "Industrials",
    
    # Clean Energy
    "ENPH": "Clean Energy", "FSLR": "Clean Energy", 
    "CWEN": "Clean Energy", "NEE": "Utilities",
    
    # Auto
    "TSLA": "Automotive",
    
    # Sustainable Agriculture
    "ADM": "Sustainable Agriculture",
    
    # Commodities
    "GLD": "Commodities", "DBC": "Commodities",
    
    # Currencies
    "UUP": "Currencies", "FXE": "Currencies", 
    "FXB": "Currencies", "FXY": "Currencies",
    
    # Bonds
    "TLT": "Bonds", "LQD": "Bonds", "BND": "Bonds", 
    "AGG": "Bonds", "GOVT": "Bonds",
}


def get_sector(ticker):
    """Get sector for a ticker, defaulting to 'Other'."""
    return SECTOR_MAPPING.get(ticker, "Other")