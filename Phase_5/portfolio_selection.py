"""
portfolio_selection.py
----------------------
DYNAMIC UNIVERSE SELECTION - NO FIXED SIZE
Assets are added greedily until adding more no longer improves performance.
AUTO-UPDATES ethical_config.py with selected assets.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize

from ethical_universe import STANDARD_POOL, ETHICAL_POOL, ORIGINAL_UNIVERSE

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("DYNAMIC UNIVERSE SELECTION - NO FIXED SIZE")
print("=" * 70)

# ============================================================================
# PARAMETERS
# ============================================================================

START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
RISK_FREE = 0.045

# Scoring weights
MOMENTUM_WEIGHT = 0.35
GROWTH_WEIGHT = 0.25
QUALITY_WEIGHT = 0.20
STABILITY_WEIGHT = 0.10
DIVERSIFICATION_WEIGHT = 0.10

# Greedy selection parameters
MIN_ASSETS = 5
MAX_ASSETS = 25
IMPROVEMENT_THRESHOLD = 0.01  # 1% improvement needed

# Re-evaluation frequency
RE_EVALUATE_DAYS = 63

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(tickers, start_date, end_date):
    """Download and process data."""
    print(f"\n📊 Loading data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    valid_tickers = []
    for ticker in tickers:
        if ticker in data.columns:
            available_pct = data[ticker].notna().sum() / len(data)
            if available_pct >= 0.8:
                valid_tickers.append(ticker)
    
    data = data[valid_tickers]
    returns = data.pct_change().dropna()
    
    print(f"   ✅ {len(valid_tickers)} assets with sufficient data")
    print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    return data, returns, valid_tickers

# ============================================================================
# SECTOR MAPPING
# ============================================================================

def get_sector_mapping():
    """Return sector mapping for all assets."""
    return {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
        'ADBE': 'Technology', 'CRM': 'Technology', 'NOW': 'Technology',
        'INTC': 'Technology', 'AMD': 'Technology', 'ORCL': 'Technology',
        'IBM': 'Technology', 'INTU': 'Technology', 'QCOM': 'Technology',
        'TXN': 'Technology', 'AVGO': 'Technology', 'ASML': 'Technology',
        
        # Healthcare
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
        'MRK': 'Healthcare', 'PFE': 'Healthcare', 'LLY': 'Healthcare',
        'BMY': 'Healthcare', 'GILD': 'Healthcare', 'AMGN': 'Healthcare',
        
        # Consumer
        'WMT': 'Consumer', 'COST': 'Consumer', 'KO': 'Consumer',
        'PEP': 'Consumer', 'PG': 'Consumer', 'HD': 'Consumer',
        'MCD': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',
        'DIS': 'Consumer',
        
        # Financials
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'GS': 'Financials', 'MS': 'Financials', 'V': 'Financials',
        'MA': 'Financials', 'SCHW': 'Financials', 'BLK': 'Financials',
        
        # Industrials
        'CAT': 'Industrials', 'GE': 'Industrials', 'BA': 'Industrials',
        'MMM': 'Industrials', 'HON': 'Industrials', 'UNP': 'Industrials',
        'UPS': 'Industrials',
        
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        
        # Clean Energy
        'ENPH': 'Clean Energy', 'NEE': 'Clean Energy', 'FSLR': 'Clean Energy',
        'CWEN': 'Clean Energy', 'BEP': 'Clean Energy', 'GEV': 'Clean Energy',
        'HASI': 'Clean Energy',
        
        # Utilities
        'DUK': 'Utilities', 'SO': 'Utilities', 'AWK': 'Utilities',
        'WTRG': 'Utilities',
        
        # Sustainable Agriculture
        'ADM': 'Sustainable Agriculture', 'NTR': 'Sustainable Agriculture',
        
        # Waste Management
        'WM': 'Waste Management', 'RSG': 'Waste Management',
        
        # Materials
        'LIN': 'Materials', 'SHW': 'Materials', 'APD': 'Materials',
        
        # Bonds
        'TLT': 'Bonds', 'LQD': 'Bonds',
        
        # Commodities
        'GLD': 'Commodities', 'DBC': 'Commodities',
        
        # ETFs
        'SPY': 'ETF', 'QQQ': 'ETF', 'EFA': 'ETF', 'EEM': 'ETF',
    }

def get_sector(ticker, sectors=None):
    if sectors is None:
        sectors = get_sector_mapping()
    return sectors.get(ticker, 'Other')

# ============================================================================
# ASSET SCORING
# ============================================================================

def score_assets(returns, date=None):
    """Score all assets based on recent performance."""
    if date is not None:
        returns = returns[returns.index <= date]
    
    if len(returns) < 63:
        return pd.Series(0.5, index=returns.columns)
    
    # 1. Momentum
    periods = [21, 63, 126, 252]
    period_weights = [0.4, 0.3, 0.2, 0.1]
    
    momentum_scores = []
    for period, weight in zip(periods, period_weights):
        if len(returns) >= period:
            period_returns = returns.iloc[-period:].mean() * 252
            period_vol = returns.iloc[-period:].std() * np.sqrt(252)
            momentum_scores.append(weight * period_returns / (period_vol + 0.01))
        else:
            momentum_scores.append(0)
    
    momentum = pd.Series(sum(momentum_scores), index=returns.columns)
    
    # 2. Growth
    if len(returns) >= 252:
        recent = returns.iloc[-126:].mean() * 252
        long_term = returns.iloc[-252:].mean() * 252
        growth = (recent - long_term) / (long_term.abs() + 0.01)
        growth = growth.clip(-1, 1)
    else:
        growth = pd.Series(0, index=returns.columns)
    
    # 3. Quality
    exp_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = (exp_ret - RISK_FREE) / (vol + 0.01)
    
    if len(returns) >= 12:
        monthly_returns = returns.resample('ME').mean()
        positive_months = (monthly_returns > 0).sum()
        pos_ratio = positive_months / len(monthly_returns)
    else:
        pos_ratio = 0.5
    
    quality = 0.6 * sharpe.clip(-1, 1) + 0.4 * pos_ratio
    
    # 4. Stability
    rolling_vol = returns.rolling(63).std() * np.sqrt(252)
    avg_vol = rolling_vol.mean()
    stability = 1 / (1 + avg_vol)
    stability = (stability - stability.min()) / (stability.max() - stability.min() + 0.01)
    
    # 5. Diversification
    sectors = get_sector_mapping()
    sector_counts = {}
    for ticker in returns.columns:
        sector = get_sector(ticker, sectors)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    diversification = pd.Series(index=returns.columns)
    for ticker in returns.columns:
        sector = get_sector(ticker, sectors)
        count = sector_counts.get(sector, 1)
        diversification[ticker] = 1 / count
    
    # Normalise
    def normalise(x):
        return (x - x.min()) / (x.max() - x.min() + 0.01)
    
    momentum_norm = normalise(momentum)
    growth_norm = normalise(growth)
    quality_norm = normalise(quality)
    stability_norm = normalise(stability)
    diversification_norm = normalise(diversification)
    
    composite_score = (
        MOMENTUM_WEIGHT * momentum_norm +
        GROWTH_WEIGHT * growth_norm +
        QUALITY_WEIGHT * quality_norm +
        STABILITY_WEIGHT * stability_norm +
        DIVERSIFICATION_WEIGHT * diversification_norm
    )
    
    return composite_score

# ============================================================================
# OPTIMISE PORTFOLIO
# ============================================================================

def calculate_portfolio_sharpe(returns, weights=None):
    """
    Calculate Sharpe ratio for a portfolio.
    """
    if len(returns.columns) == 0:
        return -np.inf
    
    if weights is None:
        weights = np.ones(len(returns.columns)) / len(returns.columns)
    
    port_returns = (returns * weights).sum(axis=1)
    mean_return = port_returns.mean() * 252
    std_return = port_returns.std() * np.sqrt(252)
    
    if std_return < 1e-10:
        return -np.inf
    
    return (mean_return - RISK_FREE) / std_return

def optimize_portfolio_simple(returns):
    """
    Simple portfolio optimisation with fallback to equal weights.
    """
    n = len(returns.columns)
    
    if n == 0:
        return None
    
    if n == 1:
        return np.array([1.0])
    
    try:
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        
        def neg_sharpe(w):
            w = np.array(w)
            ret = np.sum(exp_ret * w)
            vol = np.sqrt(w.T @ cov @ w)
            return -(ret - RISK_FREE) / vol if vol > 1e-10 else 999
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.02, 0.25) for _ in range(n))
        
        result = minimize(
            neg_sharpe,
            np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
    except:
        pass
    
    return np.ones(n) / n

# ============================================================================
# GREEDY SELECTION
# ============================================================================

def greedy_select_assets(returns, date=None, max_assets=25, min_assets=5):
    """
    Greedily select assets until adding more doesn't improve performance.
    """
    if date is not None:
        returns = returns[returns.index <= date]
    
    if len(returns) < 126:
        scores = score_assets(returns)
        return scores.nlargest(10).index.tolist()
    
    scores = score_assets(returns)
    
    top_assets = scores.nlargest(3).index.tolist()
    selected = top_assets.copy()
    remaining = scores.drop(selected).sort_values(ascending=False)
    
    test_returns = returns[selected]
    weights = optimize_portfolio_simple(test_returns)
    current_sharpe = calculate_portfolio_sharpe(test_returns, weights)
    
    sharpe_history = [current_sharpe]
    selected_history = [selected.copy()]
    
    for i in range(len(selected), max_assets):
        if len(remaining) == 0:
            break
        
        best_new_sharpe = -np.inf
        best_new_asset = None
        
        for asset in remaining.head(15).index:
            test_selected = selected + [asset]
            
            try:
                test_returns = returns[test_selected]
                weights = optimize_portfolio_simple(test_returns)
                sharpe = calculate_portfolio_sharpe(test_returns, weights)
                
                if sharpe > best_new_sharpe:
                    best_new_sharpe = sharpe
                    best_new_asset = asset
            except:
                pass
        
        if best_new_asset is None:
            break
        
        improvement = best_new_sharpe - current_sharpe
        
        if len(selected) >= min_assets and improvement < IMPROVEMENT_THRESHOLD:
            break
        
        selected.append(best_new_asset)
        remaining = remaining.drop(best_new_asset)
        current_sharpe = best_new_sharpe
        sharpe_history.append(current_sharpe)
        selected_history.append(selected.copy())
    
    optimal_idx = np.argmax(sharpe_history)
    optimal_selection = selected_history[optimal_idx]
    optimal_sharpe = sharpe_history[optimal_idx]
    
    return optimal_selection

# ============================================================================
# DYNAMIC UNIVERSE SELECTION
# ============================================================================

def select_dynamic_universe(returns, current_date, pool_tickers):
    """Select the best assets from the pool at a given date."""
    historical = returns[returns.index <= current_date]
    
    if len(historical) < 126:
        scores = score_assets(returns)
        return scores.nlargest(10).index.tolist()
    
    return greedy_select_assets(historical, max_assets=20, min_assets=5)

# ============================================================================
# RUN DYNAMIC SELECTION
# ============================================================================

def run_dynamic_selection(pool_tickers, pool_name):
    """Run dynamic universe selection over the entire period."""
    print("\n" + "=" * 70)
    print(f"DYNAMIC SELECTION: {pool_name}")
    print("=" * 70)
    
    data, returns, valid_tickers = load_data(pool_tickers, START_DATE, END_DATE)
    
    if len(valid_tickers) < 15:
        print(f"❌ Only {len(valid_tickers)} assets available.")
        return None, None
    
    start_idx = 252
    dates = []
    current_date = returns.index[start_idx]
    
    while current_date < returns.index[-1]:
        dates.append(current_date)
        current_date += timedelta(days=RE_EVALUATE_DAYS)
        if current_date > returns.index[-1]:
            break
    
    print(f"   Re-evaluating on {len(dates)} dates")
    
    selection_history = []
    size_history = []
    selection_counts = {ticker: 0 for ticker in valid_tickers}
    
    for i, date in enumerate(dates):
        selected = select_dynamic_universe(returns, date, valid_tickers)
        
        selection_history.append({
            'date': date,
            'selected': selected,
            'count': len(selected)
        })
        size_history.append(len(selected))
        
        for ticker in selected:
            selection_counts[ticker] += 1
        
        if i % 10 == 0:
            print(f"   Date: {date.strftime('%Y-%m-%d')}, Assets: {len(selected)}")
    
    avg_size = np.mean(size_history)
    print(f"\n   Statistics: Avg={avg_size:.1f}, Min={np.min(size_history)}, Max={np.max(size_history)}")
    
    sorted_counts = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   Most frequently selected:")
    for ticker, count in sorted_counts[:10]:
        freq = count / len(dates) * 100
        print(f"      {ticker}: {freq:.1f}%")
    
    return sorted_counts, selection_history

# ============================================================================
# AUTO-UPDATE CONFIG FILE
# ============================================================================

def update_config_file(standard_assets, ethical_assets):
    """
    Automatically update ethical_config.py with selected assets.
    """
    print("\n" + "=" * 70)
    print("UPDATING CONFIG FILE")
    print("=" * 70)
    
    config_path = os.path.join(script_dir, "ethical_config.py")
    
    # Read the current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the portfolio sections
    new_lines = []
    in_standard = False
    in_ethical = False
    
    for line in lines:
        # Check for standard portfolio section
        if 'OPTIMAL_STANDARD_PORTFOLIO = [' in line:
            in_standard = True
            new_lines.append('OPTIMAL_STANDARD_PORTFOLIO = [\n')
            for asset in standard_assets:
                new_lines.append(f'    "{asset}",\n')
            new_lines.append(']\n')
            continue
            
        # Check for ethical portfolio section
        elif 'OPTIMAL_ETHICAL_PORTFOLIO = [' in line:
            in_ethical = True
            new_lines.append('OPTIMAL_ETHICAL_PORTFOLIO = [\n')
            for asset in ethical_assets:
                new_lines.append(f'    "{asset}",\n')
            new_lines.append(']\n')
            continue
            
        # Skip old portfolio content
        elif in_standard and ']' in line and 'OPTIMAL_STANDARD_PORTFOLIO' not in line:
            in_standard = False
            continue
        elif in_ethical and ']' in line and 'OPTIMAL_ETHICAL_PORTFOLIO' not in line:
            in_ethical = False
            continue
        elif in_standard or in_ethical:
            continue
        
        # Keep everything else
        new_lines.append(line)
    
    # Write the updated config
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"   ✅ Updated: {config_path}")
    print(f"   Standard assets: {len(standard_assets)}")
    print(f"   Ethical assets: {len(ethical_assets)}")
    
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the dynamic universe selection."""
    
    print(f"\n📊 Starting dynamic universe selection...")
    print(f"   Min assets: {MIN_ASSETS}, Max: {MAX_ASSETS}")
    print(f"   Improvement threshold: {IMPROVEMENT_THRESHOLD*100:.1f}%")
    
    # ========================================================================
    # STANDARD POOL
    # ========================================================================
    
    standard_counts, standard_history = run_dynamic_selection(
        STANDARD_POOL, "Standard_Pool"
    )
    
    # ========================================================================
    # ETHICAL POOL
    # ========================================================================
    
    ethical_counts, ethical_history = run_dynamic_selection(
        ETHICAL_POOL, "Ethical_Pool"
    )
    
    # ========================================================================
    # SELECT TOP 15 ASSETS FROM EACH POOL
    # ========================================================================
    
    standard_assets = [t for t, c in standard_counts[:15]] if standard_counts else []
    ethical_assets = [t for t, c in ethical_counts[:15]] if ethical_counts else []
    
    # ========================================================================
    # AUTO-UPDATE CONFIG
    # ========================================================================
    
    if standard_assets and ethical_assets:
        update_config_file(standard_assets, ethical_assets)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if standard_assets:
        print(f"\n📊 Standard Pool (top {len(standard_assets)} assets):")
        print(f"   {standard_assets}")
        
        # Save to CSV
        pd.DataFrame({'Ticker': standard_assets}).to_csv(
            os.path.join(logs_dir, "optimal_standard_portfolio.csv"), index=False
        )
    
    if ethical_assets:
        print(f"\n📊 Ethical Pool (top {len(ethical_assets)} assets):")
        print(f"   {ethical_assets}")
        
        pd.DataFrame({'Ticker': ethical_assets}).to_csv(
            os.path.join(logs_dir, "optimal_ethical_portfolio.csv"), index=False
        )
    
    print(f"\n✅ Results saved to: {logs_dir}/")
    print(f"✅ Config auto-updated: ethical_config.py")

if __name__ == "__main__":
    main()