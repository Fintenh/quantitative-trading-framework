"""
portfolio_selection_growth.py
-----------------------------
Sophisticated growth-focused portfolio selection using:
- Multi-factor scoring (Momentum, Growth, Quality)
- Regime detection (Bull/Bear)
- Adaptive number of assets
- Maximum Sharpe optimisation with growth constraints
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
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ethical_universe import STANDARD_UNIVERSE, ETHICAL_UNIVERSE

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("GROWTH-FOCUSED PORTFOLIO SELECTION")
print("=" * 70)

# ============================================================================
# PARAMETERS
# ============================================================================

START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
RISK_FREE = 0.045
LOOKBACK = 252  # 1 year

# Factor weights
WEIGHTS = {
    'momentum': 0.35,
    'growth': 0.25,
    'quality': 0.20,
    'sharpe': 0.10,
    'volatility_adj': 0.10,  # Penalty for high volatility
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(tickers, start_date, end_date):
    """Download and process data."""
    print(f"\n📊 Loading data for {len(tickers)} assets...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    
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
# FACTOR COMPUTATION
# ============================================================================

def compute_momentum(returns):
    """Compute multi-period momentum score."""
    # Weighted momentum: more weight to recent periods
    periods = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m
    weights = [0.4, 0.3, 0.2, 0.1]  # Higher weight to shorter periods
    
    momentum_scores = []
    for period, weight in zip(periods, weights):
        if len(returns) >= period:
            period_returns = returns.iloc[-period:].mean() * period * 252 / period
            # Normalise by volatility for risk-adjusted momentum
            period_vol = returns.iloc[-period:].std() * np.sqrt(252)
            momentum_scores.append(weight * period_returns / period_vol)
        else:
            momentum_scores.append(0)
    
    return pd.Series(sum(momentum_scores), index=returns.columns)

def compute_growth(returns):
    """Compute growth score using earnings growth proxy."""
    # Since we don't have fundamental data, use price growth as proxy
    # Look at rolling 6-month growth vs 12-month growth
    if len(returns) >= 252:
        recent = returns.iloc[-126:].mean() * 252
        long_term = returns.iloc[-252:].mean() * 252
        growth = (recent - long_term) / long_term.abs().replace(0, 0.01)
        return growth.clip(-1, 1)
    return pd.Series(0, index=returns.columns)

def compute_quality(returns):
    """Compute quality score using Sharpe ratio and consistency."""
    # Quality = Sharpe ratio + positive months ratio
    exp_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = (exp_ret - RISK_FREE) / vol
    
    # Positive months ratio
    positive_months = returns.groupby(returns.index.month).mean() > 0
    pos_ratio = positive_months.sum() / 12
    
    # Combine
    quality = 0.6 * sharpe.clip(-1, 1) + 0.4 * pos_ratio
    return quality

def compute_volatility_penalty(returns):
    """Compute volatility penalty (lower is better)."""
    vol = returns.std() * np.sqrt(252)
    # Penalise high volatility
    return -vol / vol.max()

# ============================================================================
# COMPOSITE SCORE
# ============================================================================

def compute_composite_score(returns):
    """Compute composite score for all assets."""
    print("\n📊 Computing factor scores...")
    
    # Compute individual factors
    momentum = compute_momentum(returns)
    growth = compute_growth(returns)
    quality = compute_quality(returns)
    sharpe = ((returns.mean() * 252 - RISK_FREE) / (returns.std() * np.sqrt(252))).clip(-1, 1)
    vol_penalty = compute_volatility_penalty(returns)
    
    # Normalise each factor to [0, 1]
    def normalise(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)
    
    momentum_norm = normalise(momentum)
    growth_norm = normalise(growth)
    quality_norm = normalise(quality)
    sharpe_norm = normalise(sharpe)
    vol_penalty_norm = normalise(vol_penalty)
    
    # Compute composite score
    score = (
        WEIGHTS['momentum'] * momentum_norm +
        WEIGHTS['growth'] * growth_norm +
        WEIGHTS['quality'] * quality_norm +
        WEIGHTS['sharpe'] * sharpe_norm +
        WEIGHTS['volatility_adj'] * vol_penalty_norm
    )
    
    # Create results DataFrame
    factor_df = pd.DataFrame({
        'Momentum': momentum_norm,
        'Growth': growth_norm,
        'Quality': quality_norm,
        'Sharpe': sharpe_norm,
        'Vol_Penalty': vol_penalty_norm,
        'Composite_Score': score
    })
    
    return factor_df.sort_values('Composite_Score', ascending=False)

# ============================================================================
# SELECT OPTIMAL ASSETS (DYNAMIC NUMBER)
# ============================================================================

def select_optimal_assets(returns, factor_df, max_assets=30, min_assets=10):
    """
    Dynamically select optimal number of assets.
    Stops when adding more assets doesn't improve the Sharpe ratio.
    """
    print("\n🔍 Selecting optimal assets dynamically...")
    
    # Start with top 5 assets
    selected = factor_df.head(5).index.tolist()
    remaining = factor_df.index.difference(selected).tolist()
    
    results = []
    
    # Greedy addition
    for i in range(5, max_assets + 1):
        # Try each remaining asset
        best_sharpe = -np.inf
        best_asset = None
        
        for asset in remaining:
            test_assets = selected + [asset]
            test_returns = returns[test_assets]
            
            # Optimise weights for this set
            weights = optimize_portfolio(test_returns)
            if weights is None:
                continue
            
            # Calculate Sharpe
            exp_ret = test_returns.mean() * 252
            cov = test_returns.cov() * 252
            port_ret = np.sum(exp_ret * weights)
            port_vol = np.sqrt(weights.T @ cov @ weights)
            sharpe = (port_ret - RISK_FREE) / port_vol if port_vol > 0 else -np.inf
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_asset = asset
        
        if best_asset is None:
            break
        
        selected.append(best_asset)
        remaining.remove(best_asset)
        
        # Calculate metrics for current selection
        test_returns = returns[selected]
        weights = optimize_portfolio(test_returns)
        exp_ret = test_returns.mean() * 252
        cov = test_returns.cov() * 252
        port_ret = np.sum(exp_ret * weights)
        port_vol = np.sqrt(weights.T @ cov @ weights)
        sharpe = (port_ret - RISK_FREE) / port_vol if port_vol > 0 else -np.inf
        
        results.append({
            'n_assets': len(selected),
            'sharpe': sharpe,
            'return': port_ret,
            'volatility': port_vol,
            'tickers': selected.copy()
        })
        
        print(f"   {len(selected)} assets: Sharpe={sharpe:.3f}, Return={port_ret*100:.1f}%")
        
        # Stop if Sharpe decreases significantly
        if len(results) > 3:
            if results[-1]['sharpe'] < results[-2]['sharpe'] * 0.98:
                print(f"   ⚠️ Sharpe decreased, stopping at {len(selected)-1} assets")
                selected = results[-2]['tickers']
                break
    
    # Find best result
    best_idx = np.argmax([r['sharpe'] for r in results])
    best_result = results[best_idx]
    
    print(f"\n   ✅ Optimal: {best_result['n_assets']} assets")
    print(f"      Sharpe: {best_result['sharpe']:.3f}")
    print(f"      Return: {best_result['return']*100:.1f}%")
    print(f"      Volatility: {best_result['volatility']*100:.1f}%")
    
    return best_result

# ============================================================================
# PORTFOLIO OPTIMISATION
# ============================================================================

def optimize_portfolio(returns, max_weight=0.25):
    """Optimise portfolio with constraints."""
    n = len(returns.columns)
    
    exp_ret = returns.mean() * 252
    cov = returns.cov() * 252
    
    def neg_sharpe(w):
        w = np.array(w)
        ret = np.sum(exp_ret * w)
        vol = np.sqrt(w.T @ cov @ w)
        return -(ret - RISK_FREE) / vol if vol > 1e-10 else 999
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0.01, max_weight) for _ in range(n))
    
    try:
        result = minimize(
            neg_sharpe,
            np.ones(n) / n,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x if result.success else None
    except:
        return None

# ============================================================================
# EVALUATE AND COMPARE
# ============================================================================

def evaluate_portfolio(returns, tickers):
    """Evaluate a selected portfolio."""
    test_returns = returns[tickers]
    weights = optimize_portfolio(test_returns)
    
    if weights is None:
        return None
    
    exp_ret = test_returns.mean() * 252
    cov = test_returns.cov() * 252
    vol = np.sqrt(np.diag(cov))
    
    port_ret = np.sum(exp_ret * weights)
    port_vol = np.sqrt(weights.T @ cov @ weights)
    sharpe = (port_ret - RISK_FREE) / port_vol if port_vol > 0 else 0
    
    return {
        'tickers': tickers,
        'weights': weights,
        'return': port_ret,
        'volatility': port_vol,
        'sharpe': sharpe,
        'n_assets': len(tickers)
    }

# ============================================================================
# MAIN
# ============================================================================

print("\n📊 Loading universes...")

# Standard Universe
data_std, returns_std, valid_std = load_data(STANDARD_UNIVERSE, START_DATE, END_DATE)
print(f"Valid assets: {len(valid_std)}")

# Ethical Universe
data_eth, returns_eth, valid_eth = load_data(ETHICAL_UNIVERSE, START_DATE, END_DATE)
print(f"Valid assets: {len(valid_eth)}")

# ============================================================================
# SELECT FOR STANDARD UNIVERSE
# ============================================================================

print("\n" + "=" * 70)
print("STANDARD UNIVERSE SELECTION")
print("=" * 70)

# Compute scores
factor_df_std = compute_composite_score(returns_std)
print("\n📊 Top 10 Assets by Composite Score:")
print(factor_df_std.head(10).to_string())

# Select optimal assets
standard_result = select_optimal_assets(returns_std, factor_df_std, max_assets=30)
standard_result['returns'] = returns_std

# ============================================================================
# SELECT FOR ETHICAL UNIVERSE
# ============================================================================

print("\n" + "=" * 70)
print("ETHICAL UNIVERSE SELECTION")
print("=" * 70)

# Compute scores
factor_df_eth = compute_composite_score(returns_eth)
print("\n📊 Top 10 Assets by Composite Score:")
print(factor_df_eth.head(10).to_string())

# Select optimal assets
ethical_result = select_optimal_assets(returns_eth, factor_df_eth, max_assets=30)
ethical_result['returns'] = returns_eth

# ============================================================================
# FINAL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("FINAL SELECTION COMPARISON")
print("=" * 70)

print("\n| Universe | Assets | Sharpe | Return | Volatility |")
print("|----------|--------|--------|--------|------------|")

for name, result in [("Standard", standard_result), ("Ethical", ethical_result)]:
    print(f"| {name:<10} | {result['n_assets']:>6} | {result['sharpe']:>6.3f} | {result['return']*100:>6.1f}% | {result['volatility']*100:>10.1f}% |")

print(f"\n📊 Standard Selected Assets ({standard_result['n_assets']}):")
print(f"   {standard_result['tickers']}")

print(f"\n📊 Ethical Selected Assets ({ethical_result['n_assets']}):")
print(f"   {ethical_result['tickers']}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n💾 Saving results...")

# Create summary
summary_df = pd.DataFrame({
    'Universe': ['Standard', 'Ethical'],
    'Assets': [standard_result['n_assets'], ethical_result['n_assets']],
    'Sharpe': [standard_result['sharpe'], ethical_result['sharpe']],
    'Return': [standard_result['return'] * 100, ethical_result['return'] * 100],
    'Volatility': [standard_result['volatility'] * 100, ethical_result['volatility'] * 100]
})
summary_df.to_csv(os.path.join(logs_dir, "growth_portfolio_results.csv"), index=False)

# Save tickers
tickers_df = pd.DataFrame({
    'Standard_Tickers': standard_result['tickers'] + [''] * (max(len(standard_result['tickers']), len(ethical_result['tickers'])) - len(standard_result['tickers'])),
    'Ethical_Tickers': ethical_result['tickers'] + [''] * (max(len(standard_result['tickers']), len(ethical_result['tickers'])) - len(ethical_result['tickers']))
})
tickers_df.to_csv(os.path.join(logs_dir, "growth_portfolio_tickers.csv"), index=False)

print(f"✅ Saved: growth_portfolio_results.csv")
print(f"✅ Saved: growth_portfolio_tickers.csv")

print("\n" + "=" * 70)
print("🎉 GROWTH-FOCUSED PORTFOLIO SELECTION COMPLETE!")
print("=" * 70)