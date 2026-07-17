"""
portfolio.py
------------
Phase 1.3: Linear Algebra and Portfolio Risk.
Downloads multiple stocks, computes covariance matrix, performs Cholesky decomposition,
and constructs the Efficient Frontier to find optimal portfolios.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

figures_dir = os.path.join(script_dir, "figures")
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

print(f"📁 Working directory: {os.getcwd()}")

# ============================================================================
# CONSTANTS
# ============================================================================

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
RISK_FREE = 0.045
TRADING_DAYS = 252

# ============================================================================
# 1. DOWNLOAD DATA
# ============================================================================

print("\n" + "=" * 60)
print("PHASE 1.3: LINEAR ALGEBRA AND PORTFOLIO RISK")
print("=" * 60)

print(f"\nDownloading: {TICKERS}")

price_data = yf.download(TICKERS, start="2020-01-01", end="2025-01-01")["Close"]
price_data.columns = TICKERS

daily_returns = price_data.pct_change().dropna()
print(f"✅ Downloaded {len(daily_returns)} days of data")

# ============================================================================
# 2. COVARIANCE AND CORRELATION MATRICES
# ============================================================================

print("\n--- 2. Covariance Matrix ---")

# Annualise
cov_matrix = daily_returns.cov() * TRADING_DAYS
corr_matrix = daily_returns.corr()

print("\nCovariance Matrix (Annualised):")
print(cov_matrix.round(4))

print("\nCorrelation Matrix:")
print(corr_matrix.round(4))

# ============================================================================
# 3. CHOLESKY DECOMPOSITION
# ============================================================================

print("\n--- 3. Cholesky Decomposition ---")
print("(Used for generating correlated random variables)")

L = cholesky(cov_matrix, lower=True)
print("\nLower Triangular Matrix (L):")
print(L.round(4))

# Verify
reconstructed = L @ L.T
max_diff = np.max(np.abs(cov_matrix.values - reconstructed))
print(f"\nReconstruction error: {max_diff:.10f}")
print("✓ Cholesky decomposition verified")

# ============================================================================
# 4. INDIVIDUAL STOCK STATISTICS
# ============================================================================

print("\n--- 4. Individual Stock Statistics ---")

expected_returns = daily_returns.mean() * TRADING_DAYS
volatilities = np.sqrt(np.diag(cov_matrix))

print("\nAnnualised Returns:")
for ticker, ret in expected_returns.items():
    print(f"  {ticker}: {ret * 100:.2f}%")

print("\nAnnualised Volatilities:")
for ticker, vol in zip(TICKERS, volatilities):
    print(f"  {ticker}: {vol * 100:.2f}%")

# ============================================================================
# 5. PORTFOLIO OPTIMISATION
# ============================================================================

print("\n--- 5. Efficient Frontier ---")

def portfolio_stats(weights):
    """Calculate return and volatility for a given portfolio."""
    weights = np.array(weights)
    ret = np.sum(expected_returns * weights)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return ret, vol

def neg_sharpe(weights):
    """Negative Sharpe ratio (for minimisation)."""
    ret, vol = portfolio_stats(weights)
    return -(ret - RISK_FREE) / vol if vol > 0 else 999

def portfolio_vol(weights):
    """Portfolio volatility only."""
    _, vol = portfolio_stats(weights)
    return vol

# Constraints and bounds
n_assets = len(expected_returns)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_assets))
initial_weights = np.ones(n_assets) / n_assets

print("Solving for optimal portfolios...")

# Minimum Variance Portfolio
mvp_result = minimize(portfolio_vol, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
mvp_weights = mvp_result.x
mvp_ret, mvp_vol = portfolio_stats(mvp_weights)

# Maximum Sharpe Ratio Portfolio
msr_result = minimize(neg_sharpe, initial_weights,
                     method='SLSQP', bounds=bounds, constraints=constraints)
msr_weights = msr_result.x
msr_ret, msr_vol = portfolio_stats(msr_weights)

print("\n--- Minimum Variance Portfolio ---")
for ticker, w in zip(TICKERS, mvp_weights):
    if w > 0.01:
        print(f"  {ticker}: {w * 100:.1f}%")
print(f"  Expected Return: {mvp_ret * 100:.2f}%")
print(f"  Volatility: {mvp_vol * 100:.2f}%")

print("\n--- Maximum Sharpe Ratio Portfolio ---")
for ticker, w in zip(TICKERS, msr_weights):
    if w > 0.01:
        print(f"  {ticker}: {w * 100:.1f}%")
print(f"  Expected Return: {msr_ret * 100:.2f}%")
print(f"  Volatility: {msr_vol * 100:.2f}%")
print(f"  Sharpe Ratio: {(msr_ret - RISK_FREE) / msr_vol:.3f}")

# ============================================================================
# 6. EFFICIENT FRONTIER PLOT
# ============================================================================

print("\n--- 6. Generating Efficient Frontier Plot ---")

# Generate random portfolios
N_PORTFOLIOS = 5000
random_weights = np.random.dirichlet(np.ones(n_assets), N_PORTFOLIOS)

random_returns = []
random_vols = []
random_sharpes = []

for w in random_weights:
    ret, vol = portfolio_stats(w)
    random_returns.append(ret)
    random_vols.append(vol)
    random_sharpes.append((ret - RISK_FREE) / vol)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(random_vols, random_returns, 
                    c=random_sharpes, cmap='viridis',
                    marker='o', s=10, alpha=0.3)

ax.set_xlabel('Volatility (Risk)')
ax.set_ylabel('Expected Return')
ax.set_title('Efficient Frontier: Apple, Microsoft, Google, Amazon, Tesla')
ax.grid(True, alpha=0.3)

# Colour bar
cbar = plt.colorbar(scatter)
cbar.set_label('Sharpe Ratio')

# Mark MVP
ax.scatter(mvp_vol, mvp_ret, color='red', marker='*', s=200, label='Min Variance')
ax.annotate('MVP', (mvp_vol * 1.02, mvp_ret * 0.98))

# Mark MSR
ax.scatter(msr_vol, msr_ret, color='gold', marker='*', s=200, label='Max Sharpe')
ax.annotate('Max Sharpe', (msr_vol * 1.02, msr_ret * 0.98))

# Mark individual stocks
for ticker, vol, ret in zip(TICKERS, volatilities, expected_returns):
    ax.scatter(vol, ret, s=100, label=ticker)

ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "efficient_frontier.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✅ Saved: efficient_frontier.png")

# ============================================================================
# COMPLETE
# ============================================================================

print("\n" + "=" * 60)
print("✅ PHASE 1.3 COMPLETE!")
print("=" * 60)