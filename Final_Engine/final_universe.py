"""
final_universe.py
-----------------
Defines the asset pool for the final portfolio selection.
This is the universe from which the selection algorithm will choose
the optimal 15 assets for the live trading engine.
"""

# ---------------------------------------------------------------------------
# Final universe pool (expanded set for selection)
# ---------------------------------------------------------------------------

FINAL_POOL = [
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
    "AMGN",

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
    # US renewable energy
    "ENPH": "Clean Energy",
    "NEE": "Clean Energy",
    "FSLR": "Clean Energy",
    "CWEN": "Clean Energy",
    "GEV": "Clean Energy",
    "HASI": "Clean Energy",

    # International renewable energy
    "VWS.CO": "Clean Energy",
    "IBE": "Clean Energy",
    "BEP": "Clean Energy",
    "NEL": "Clean Energy",
    "CSIQ": "Clean Energy",

    # US technology
    "MSFT": "Technology",
    "CRM": "Technology",
    "INTC": "Technology",
    "ADBE": "Technology",
    "NOW": "Technology",
    "NVDA": "Technology",

    # International technology
    "ASML": "Technology",
    "SAP": "Technology",

    # US healthcare
    "GILD": "Healthcare",
    "BMY": "Healthcare",
    "MRK": "Healthcare",
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "ABBV": "Healthcare",
    "JNJ": "Healthcare",
    "AMGN": "Healthcare",

    # International healthcare
    "NVO": "Healthcare",

    # US consumer
    "COST": "Consumer",
    "PG": "Consumer",
    "WMT": "Consumer",
    "KO": "Consumer",
    "PEP": "Consumer",

    # International consumer
    "UL": "Consumer",

    # US sustainable agriculture
    "ADM": "Sustainable Agriculture",
    "NTR": "Sustainable Agriculture",

    # US green finance
    "V": "Financials",
    "MA": "Financials",

    # US sustainable materials
    "LIN": "Materials",
    "APD": "Materials",

    # US water
    "AWK": "Utilities",
    "WTRG": "Utilities",

    # US waste management
    "WM": "Waste Management",
    "RSG": "Waste Management",

    # US ESG leaders
    "SCHW": "Financials",
    "BLK": "Financials",
}

# ---------------------------------------------------------------------------
# Country mapping
# ---------------------------------------------------------------------------

COUNTRY_MAPPING = {
    # US
    "ENPH": "US",
    "NEE": "US",
    "FSLR": "US",
    "CWEN": "US",
    "GEV": "US",
    "HASI": "US",
    "MSFT": "US",
    "CRM": "US",
    "INTC": "US",
    "ADBE": "US",
    "NOW": "US",
    "NVDA": "US",
    "GILD": "US",
    "BMY": "US",
    "MRK": "US",
    "LLY": "US",
    "UNH": "US",
    "ABBV": "US",
    "JNJ": "US",
    "AMGN": "US",
    "COST": "US",
    "PG": "US",
    "WMT": "US",
    "KO": "US",
    "PEP": "US",
    "ADM": "US",
    "NTR": "US",
    "V": "US",
    "MA": "US",
    "LIN": "US",
    "APD": "US",
    "AWK": "US",
    "WTRG": "US",
    "WM": "US",
    "RSG": "US",
    "SCHW": "US",
    "BLK": "US",

    # Denmark
    "VWS.CO": "Denmark",
    "NVO": "Denmark",

    # Netherlands
    "ASML": "Netherlands",

    # Germany
    "SAP": "Germany",

    # Spain
    "IBE": "Spain",

    # Canada
    "BEP": "Canada",
    "CSIQ": "Canada",

    # Norway
    "NEL": "Norway",

    # United Kingdom
    "UL": "United Kingdom",
}

# ---------------------------------------------------------------------------
# Company description mapping (for preserving comments in final_config.py)
# ---------------------------------------------------------------------------

COMPANY_DESCRIPTION = {
    # US renewable energy
    "ENPH": "Enphase Energy - Solar",
    "NEE": "NextEra Energy - Renewable utility",
    "FSLR": "First Solar - Solar manufacturing",
    "CWEN": "Clearway Energy - Renewable energy",
    "GEV": "GE Vernova - Renewable energy solutions",
    "HASI": "Hannon Armstrong - Climate finance",

    # International renewable energy
    "VWS.CO": "Vestas - Denmark (wind turbine leader)",
    "IBE": "Iberdrola - Spain (renewable utility)",
    "BEP": "Brookfield Renewable - Canada",
    "NEL": "Nel ASA - Norway (hydrogen)",
    "CSIQ": "Canadian Solar - Canada (solar)",

    # US technology
    "MSFT": "Microsoft - Enterprise software",
    "CRM": "Salesforce - CRM software",
    "INTC": "Intel - Semiconductors",
    "ADBE": "Adobe - Creative software",
    "NOW": "ServiceNow - Enterprise software",
    "NVDA": "NVIDIA - AI leader",

    # International technology
    "ASML": "ASML - Semiconductor equipment (Netherlands)",
    "SAP": "SAP - Enterprise software (Germany)",

    # US healthcare
    "GILD": "Gilead Sciences - Pharmaceuticals",
    "BMY": "Bristol-Myers Squibb - Pharmaceuticals",
    "MRK": "Merck - Pharmaceuticals",
    "LLY": "Eli Lilly - Pharmaceuticals",
    "UNH": "UnitedHealth - Health insurance",
    "ABBV": "AbbVie - Pharmaceuticals",
    "JNJ": "Johnson & Johnson - Healthcare",
    "AMGN": "Amgen - Biotech",

    # International healthcare
    "NVO": "Novo Nordisk - Pharmaceuticals (Denmark)",

    # US consumer
    "COST": "Costco - Retail with strong labour practices",
    "PG": "Procter & Gamble - Consumer goods",
    "WMT": "Walmart - Retail",
    "KO": "Coca-Cola - Beverages",
    "PEP": "PepsiCo - Beverages",

    # International consumer
    "UL": "Unilever - Consumer goods (United Kingdom)",

    # US sustainable agriculture
    "ADM": "Archer-Daniels-Midland - Agriculture",
    "NTR": "Nutrien - Fertilizers",

    # US green finance
    "V": "Visa - Financial inclusion",
    "MA": "Mastercard - Financial inclusion",

    # US sustainable materials
    "LIN": "Linde - Industrial gases",
    "APD": "Air Products - Hydrogen/clean energy",

    # US water
    "AWK": "American Water Works - Water utility",
    "WTRG": "Essential Utilities - Water utility",

    # US waste management
    "WM": "Waste Management - Waste management leader",
    "RSG": "Republic Services - Waste management leader",

    # US ESG leaders
    "SCHW": "Charles Schwab - Sustainable investing",
    "BLK": "BlackRock - ESG investing",
}