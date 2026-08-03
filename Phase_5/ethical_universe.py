"""
ethical_universe.py
-------------------
Defines the asset pools for portfolio selection.
"""

# ---------------------------------------------------------------------------
# Original benchmark (fixed 18 assets from Phases 1-4)
# ---------------------------------------------------------------------------

ORIGINAL_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
    "JPM", "JNJ", "XOM",
    "SPY", "QQQ", "EFA", "EEM",
    "TLT", "LQD", "GLD", "DBC"
]

# ---------------------------------------------------------------------------
# Standard pool (US + best-of-breed international)
# ---------------------------------------------------------------------------

STANDARD_POOL = [
    # US technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "ORCL", "IBM",
    "ADBE", "CRM", "NOW", "INTU", "QCOM", "TXN", "AVGO",

    # International technology
    "ASML", "SAP", "TSM",

    # US healthcare
    "JNJ", "UNH", "ABBV", "MRK", "PFE", "LLY", "BMY", "GILD", "AMGN",

    # International healthcare
    "NVO",

    # US consumer
    "WMT", "COST", "KO", "PEP", "PG", "HD", "MCD", "NKE", "SBUX", "DIS",

    # International consumer
    "UL",

    # US financials
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",

    # International financials
    "HSBC",

    # US industrials
    "CAT", "GE", "BA", "MMM", "HON", "UNP", "UPS",

    # US energy
    "XOM", "CVX", "COP",

    # International energy
    "TTE", "SHEL",

    # US utilities
    "NEE", "DUK", "SO",

    # US materials
    "LIN", "SHW", "APD",

    # International green energy
    "VWS.CO",  # Vestas - Denmark (world's largest wind turbine manufacturer)
    "IBE",     # Iberdrola - Spain (global renewable utility)
    "CSIQ",    # Canadian Solar - Canada (global solar manufacturer)
]

# ---------------------------------------------------------------------------
# Ethical pool (US + best-of-breed international, ESG-screened)
# ---------------------------------------------------------------------------

ETHICAL_POOL = [
    # US renewable energy
    "ENPH",    # Enphase Energy - Solar
    "NEE",     # NextEra Energy - Renewable utility
    "FSLR",    # First Solar - Solar manufacturing
    "CWEN",    # Clearway Energy - Renewable energy
    "GEV",     # GE Vernova - Renewable energy solutions
    "HASI",    # Hannon Armstrong - Climate finance

    # International renewable energy
    "VWS.CO",  # Vestas - Denmark (wind turbine leader)
    "IBE",     # Iberdrola - Spain (renewable utility)
    "BEP",     # Brookfield Renewable - Canada
    "NEL",     # Nel ASA - Norway (hydrogen)
    "CSIQ",    # Canadian Solar - Canada (solar)

    # US technology (ESG leaders)
    "MSFT", "CRM", "INTC", "ADBE", "NOW", "NVDA",

    # International technology (ESG leaders)
    "ASML", "SAP",

    # US healthcare
    "GILD", "BMY", "MRK", "LLY", "UNH", "ABBV", "JNJ",

    # International healthcare
    "NVO",

    # US consumer (ethical)
    "COST", "PG", "WMT", "KO", "PEP",

    # International consumer (ethical)
    "UL",

    # US sustainable agriculture
    "ADM", "NTR",

    # US green finance
    "V", "MA",

    # US sustainable materials
    "LIN", "APD",

    # US water
    "AWK", "WTRG",

    # US waste management
    "WM", "RSG",

    # US ESG leaders
    "SCHW", "BLK",
]

# ---------------------------------------------------------------------------
# Sector mapping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ethical exclusion criteria
# ---------------------------------------------------------------------------

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