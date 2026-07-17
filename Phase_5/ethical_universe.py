"""
ethical_universe.py
-------------------
Defines the asset pools for portfolio selection.
"""

# ============================================================================
# ORIGINAL BENCHMARK (Fixed 18 assets from Phases 1-4)
# ============================================================================

ORIGINAL_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "JNJ", "XOM",
    "SPY", "QQQ", "EFA", "EEM",
    "TLT", "LQD", "GLD", "DBC"
]

# ============================================================================
# STANDARD POOL (US + Best-of-Breed International)
# ============================================================================

STANDARD_POOL = [
    # ===== US TECHNOLOGY =====
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "ORCL", "IBM",
    "ADBE", "CRM", "NOW", "INTU", "QCOM", "TXN", "AVGO",
    
    # ===== INTERNATIONAL TECHNOLOGY =====
    "ASML", "SAP", "TSM",
    
    # ===== US HEALTHCARE =====
    "JNJ", "UNH", "ABBV", "MRK", "PFE", "LLY", "BMY", "GILD", "AMGN",
    
    # ===== INTERNATIONAL HEALTHCARE =====
    "NVO",
    
    # ===== US CONSUMER =====
    "WMT", "COST", "KO", "PEP", "PG", "HD", "MCD", "NKE", "SBUX", "DIS",
    
    # ===== INTERNATIONAL CONSUMER =====
    "UL",
    
    # ===== US FINANCIALS =====
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
    
    # ===== INTERNATIONAL FINANCIALS =====
    "HSBC",
    
    # ===== US INDUSTRIALS =====
    "CAT", "GE", "BA", "MMM", "HON", "UNP", "UPS",
    
    # ===== US ENERGY =====
    "XOM", "CVX", "COP",
    
    # ===== INTERNATIONAL ENERGY =====
    "TTE", "SHEL",
    
    # ===== US UTILITIES =====
    "NEE", "DUK", "SO",
    
    # ===== US MATERIALS =====
    "LIN", "SHW", "APD",
    
    # ========================================================================
    # INTERNATIONAL GREEN ENERGY
    # ========================================================================
    "VWS.CO",  # Vestas - Denmark (World's largest wind turbine manufacturer)
    "IBE",     # Iberdrola - Spain (Global renewable utility)
    "CSIQ",    # Canadian Solar - Canada (Global solar manufacturer)
]

# ============================================================================
# ETHICAL POOL (US + Best-of-Breed International, ESG-screened)
# ============================================================================

ETHICAL_POOL = [
    # ===== US RENEWABLE ENERGY =====
    "ENPH",    # Enphase Energy - Solar
    "NEE",     # NextEra Energy - Renewable utility
    "FSLR",    # First Solar - Solar manufacturing
    "CWEN",    # Clearway Energy - Renewable energy
    "GEV",     # GE Vernova - Renewable energy solutions
    "HASI",    # Hannon Armstrong - Climate finance
    
    # ===== INTERNATIONAL RENEWABLE ENERGY  =====
    "VWS.CO",  # Vestas - Denmark (Wind turbine leader)
    "IBE",     # Iberdrola - Spain (Renewable utility)
    "BEP",     # Brookfield Renewable - Canada
    "NEL",     # Nel ASA - Norway (Hydrogen)
    "CSIQ",    # Canadian Solar - Canada (Solar)
    
    # ===== US TECHNOLOGY (ESG leaders) =====
    "MSFT", "CRM", "INTC", "ADBE", "NOW", "NVDA",
    
    # ===== INTERNATIONAL TECHNOLOGY (ESG leaders) =====
    "ASML", "SAP",
    
    # ===== US HEALTHCARE =====
    "GILD", "BMY", "MRK", "LLY", "UNH", "ABBV", "JNJ",
    
    # ===== INTERNATIONAL HEALTHCARE =====
    "NVO",
    
    # ===== US CONSUMER (Ethical) =====
    "COST", "PG", "WMT", "KO", "PEP",
    
    # ===== INTERNATIONAL CONSUMER (Ethical) =====
    "UL",
    
    # ===== US SUSTAINABLE AGRICULTURE =====
    "ADM", "NTR",
    
    # ===== US GREEN FINANCE =====
    "V", "MA",
    
    # ===== US SUSTAINABLE MATERIALS =====
    "LIN", "APD",
    
    # ===== US WATER =====
    "AWK", "WTRG",
    
    # ===== US WASTE MANAGEMENT =====
    "WM", "RSG",
    
    # ===== US ESG LEADERS =====
    "SCHW", "BLK",
]

# ============================================================================
# SECTOR MAPPING
# ============================================================================

SECTOR_MAPPING = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "AMZN": "Technology", "META": "Technology",
    "AMD": "Technology", "ORCL": "Technology", "IBM": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "NOW": "Technology",
    "INTU": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "AVGO": "Technology", "ASML": "Technology", "SAP": "Technology",
    "TSM": "Technology",
    
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "PFE": "Healthcare", "LLY": "Healthcare",
    "BMY": "Healthcare", "GILD": "Healthcare", "AMGN": "Healthcare",
    "NVO": "Healthcare",
    
    # Consumer
    "WMT": "Consumer", "COST": "Consumer", "KO": "Consumer",
    "PEP": "Consumer", "PG": "Consumer", "HD": "Consumer",
    "MCD": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "DIS": "Consumer", "UL": "Consumer",
    
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "V": "Financials",
    "MA": "Financials", "SCHW": "Financials", "BLK": "Financials",
    "HSBC": "Financials",
    
    # Industrials
    "CAT": "Industrials", "GE": "Industrials", "BA": "Industrials",
    "MMM": "Industrials", "HON": "Industrials", "UNP": "Industrials",
    "UPS": "Industrials",
    
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "TTE": "Energy", "SHEL": "Energy",
    
    # Clean Energy
    "ENPH": "Clean Energy", "NEE": "Clean Energy", "FSLR": "Clean Energy",
    "CWEN": "Clean Energy", "GEV": "Clean Energy", "HASI": "Clean Energy",
    "VWS.CO": "Clean Energy", "IBE": "Clean Energy",
    "BEP": "Clean Energy", "NEL": "Clean Energy", "CSIQ": "Clean Energy",
    
    # Utilities
    "DUK": "Utilities", "SO": "Utilities", "AWK": "Utilities",
    "WTRG": "Utilities",
    
    # Sustainable Agriculture
    "ADM": "Sustainable Agriculture", "NTR": "Sustainable Agriculture",
    
    # Waste Management
    "WM": "Waste Management", "RSG": "Waste Management",
    
    # Materials
    "LIN": "Materials", "SHW": "Materials", "APD": "Materials",
}


def get_sector(ticker):
    """Get sector for a ticker, defaulting to 'Other'."""
    return SECTOR_MAPPING.get(ticker, "Other")


# ============================================================================
# ETHICAL EXCLUSION CRITERIA
# ============================================================================

ETHICAL_EXCLUSIONS = {
    "TSLA": "Elon Musk controversies, labour violations",
    "XOM": "Fossil fuels, climate denial",
    "CVX": "Fossil fuels",
    "COP": "Fossil fuels",
    "DBC": "Commodities - includes fossil fuels",
    "JPM": "Fossil fuel financing",
    "BAC": "Fossil fuel financing",
    "WFC": "Fossil fuel financing",
    "GS": "Fossil fuel financing",
    "MS": "Fossil fuel financing",
    "HSBC": "Fossil fuel financing (mixed)",
}