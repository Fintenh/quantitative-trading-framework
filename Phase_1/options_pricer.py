"""
options_pricer.py - GARCH volatility forecasting and option pricing via Monte Carlo.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Setup
os.chdir(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(os.getcwd(), "figures")
LOGS = os.path.join(os.getcwd(), "logs")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# Constants
RF = 0.045
DAYS = 252
N_SIMS = 10000

print("=" * 60)
print("PHASE 1.4: OPTION PRICING")
print("=" * 60)

# ----------------------------------------------------------------------------
# 1. LOAD DATA & FIT GARCH
# ----------------------------------------------------------------------------

print("\nLoading data & fitting GARCH...")

data = pd.read_csv(os.path.join(LOGS, "apple_data.csv"), index_col="Date", parse_dates=True)
returns = data['Returns'].dropna()

# Fit GARCH on scaled returns
model = arch_model(returns.values * 100, vol='Garch', p=1, q=1)
fit = model.fit(disp='off')

# Get daily volatility forecast
forecast = fit.forecast(horizon=DAYS)
vol_daily = np.sqrt(forecast.variance.iloc[-1].values / 10000)

S0 = float(data['Close'].iloc[-1])
mu_hist = returns.mean() * DAYS
vol_ann = np.mean(vol_daily) * np.sqrt(DAYS)

print(f"  Current Price (S0): ${S0:.2f}")
print(f"  Volatility: {vol_ann*100:.1f}%")
print(f"  Historical drift: {mu_hist*100:.1f}%")
print(f"  Risk-free rate: {RF*100:.1f}%")

# ----------------------------------------------------------------------------
# 2. SIMULATE
# ----------------------------------------------------------------------------

def simulate_historical(drift, S0, vol, steps=DAYS, n=N_SIMS):
    """Original GBM (unchanged)."""
    dt = 1 / steps
    paths = np.zeros((n, steps + 1))
    paths[:, 0] = S0
    shocks = np.random.normal(0, 1, (n, steps))
    drift_daily = drift * dt
    
    for t in range(1, steps + 1):
        sigma = vol[t-1] if t <= len(vol) else vol[-1]
        paths[:, t] = paths[:, t-1] * np.exp(
            (drift_daily - 0.5 * sigma**2) + sigma * shocks[:, t-1]
        )
    return paths

def simulate_risk_neutral(drift, S0, vol, steps=DAYS, n=N_SIMS):
    """Corrected GBM for risk-neutral."""
    dt = 1 / steps
    paths = np.zeros((n, steps + 1))
    paths[:, 0] = S0
    shocks = np.random.normal(0, 1, (n, steps))
    
    for t in range(1, steps + 1):
        sigma = vol[t-1] if t <= len(vol) else vol[-1]
        paths[:, t] = paths[:, t-1] * np.exp(
            (drift * dt - 0.5 * sigma**2 * dt) + sigma * np.sqrt(dt) * shocks[:, t-1]
        )
    return paths

print("\nSimulating...")
paths_hist = simulate_historical(mu_hist, S0, vol_daily)
paths_rn = simulate_risk_neutral(RF, S0, vol_daily)

print(f"  Historical mean terminal: ${np.mean(paths_hist[:,-1]):.2f}")
print(f"  Risk-neutral mean terminal: ${np.mean(paths_rn[:,-1]):.2f}")

# ----------------------------------------------------------------------------
# 3. PRICE OPTIONS (USING $250 STRIKE)
# ----------------------------------------------------------------------------

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def mc_price(paths, K, r, T=1.0):
    payoffs = np.maximum(paths[:, -1] - K, 0)
    return np.exp(-r*T) * np.mean(payoffs), np.mean(paths[:, -1] > K) * 100

# Use $250 strike (closest available to Apple's current price)
K = 250.00

print(f"\n--- Pricing (Strike: ${K:.2f}) ---")
print(f"{'Drift':<15} {'MC Price':<12} {'ITM %':<10}")
print("-" * 40)

# Historical (real-world)
mc_hist, itm_hist = mc_price(paths_hist, K, RF)
print(f"{'Historical':<15} ${mc_hist:<11.2f} {itm_hist:<9.1f}%")

# Risk-neutral
mc_rn, itm_rn = mc_price(paths_rn, K, RF)
print(f"{'Risk-Neutral':<15} ${mc_rn:<11.2f} {itm_rn:<9.1f}%")

# Black-Scholes
bs = black_scholes(S0, K, 1.0, RF, vol_ann)
print(f"{'Black-Scholes':<15} ${bs:<11.2f} {'--':<9}")

print(f"\n--- Comparison ---")
print(f"  Historical vs Risk-Neutral: ${mc_hist - mc_rn:.2f}")
print(f"  Historical vs Black-Scholes: ${mc_hist - bs:.2f}")
print(f"  Risk-Neutral vs Black-Scholes: ${mc_rn - bs:.2f}")

# ----------------------------------------------------------------------------
# 4. PLOT PATHS
# ----------------------------------------------------------------------------

def plot_paths(paths, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in np.random.choice(len(paths), 100, replace=False):
        ax.plot(paths[i], alpha=0.3, lw=0.5)
    mean_path = paths.mean(axis=0)
    ax.plot(mean_path, 'r', lw=2, label='Mean')
    ax.fill_between(range(DAYS+1), 
                   np.percentile(paths, 5, axis=0),
                   np.percentile(paths, 95, axis=0),
                   color='r', alpha=0.1, label='90% CI')
    ax.axhline(y=K, color='black', ls='--', lw=1.5, alpha=0.7, label=f'Strike ${K:.0f}')
    ax.set_title(title)
    ax.set_xlabel('Time Steps (Days)')
    ax.set_ylabel('Stock Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, fname), dpi=300)
    plt.show()
    plt.close()

print("\n--- Path Plots ---")
plot_paths(paths_hist, f'Monte Carlo: Historical Drift (Strike ${K:.0f})', 'mc_paths_hist.png')
plot_paths(paths_rn, f'Monte Carlo: Risk-Neutral (Strike ${K:.0f})', 'mc_paths_rn.png')
print("  ✅ Saved and displayed path plots")

# ----------------------------------------------------------------------------
# 5. PAYOFF PLOTS (HISTORICAL VS RISK-NEUTRAL SIDE BY SIDE)
# ----------------------------------------------------------------------------

def plot_payoff_comparison(paths_hist, paths_rn, K, fname):
    """Compare historical vs risk-neutral payoffs side by side."""
    terminal_hist = paths_hist[:, -1]
    terminal_rn = paths_rn[:, -1]
    
    payoffs_hist = np.maximum(terminal_hist - K, 0)
    payoffs_rn = np.maximum(terminal_rn - K, 0)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Row 1: Historical (Real-World)
    ax = axes[0, 0]
    ax.hist(terminal_hist, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(K, color='red', ls='--', lw=2, label=f'Strike ${K:.2f}')
    ax.axvline(S0, color='green', ls='--', lw=2, label=f'Current ${S0:.2f}')
    ax.set_title(f'Historical: Terminal Prices')
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(payoffs_hist[payoffs_hist > 0], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(mc_hist, color='red', lw=2, label=f'Price ${mc_hist:.2f}')
    zero_pct = np.mean(payoffs_hist == 0) * 100
    # FIXED: fontsize goes OUTSIDE the bbox dict
    ax.text(0.95, 0.95, f'Zero: {zero_pct:.1f}%', transform=ax.transAxes, 
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_title(f'Historical: Option Payoffs')
    ax.set_xlabel('Payoff ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Risk-Neutral
    ax = axes[1, 0]
    ax.hist(terminal_rn, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(K, color='red', ls='--', lw=2, label=f'Strike ${K:.2f}')
    ax.axvline(S0, color='green', ls='--', lw=2, label=f'Current ${S0:.2f}')
    ax.set_title(f'Risk-Neutral: Terminal Prices')
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.hist(payoffs_rn[payoffs_rn > 0], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(mc_rn, color='red', lw=2, label=f'Price ${mc_rn:.2f}')
    zero_pct = np.mean(payoffs_rn == 0) * 100
    # FIXED: fontsize goes OUTSIDE the bbox dict
    ax.text(0.95, 0.95, f'Zero: {zero_pct:.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_title(f'Risk-Neutral: Option Payoffs')
    ax.set_xlabel('Payoff ($)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Payoff Comparison: Historical ({mu_hist*100:.1f}%) vs Risk-Neutral ({RF*100:.1f}%)\nStrike: ${K:.2f} (closest exchange strike)', 
                 fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, fname), dpi=300)
    plt.show()
    plt.close()

print("\n--- Payoff Comparison Plots ---")
plot_payoff_comparison(paths_hist, paths_rn, K, 'payoff_comparison.png')
print("  ✅ Saved and displayed payoff comparison plot")

# ----------------------------------------------------------------------------
# 6. SUMMARY STATISTICS
# ----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nStrike Price: ${K:.2f} (closest to current ${S0:.2f})")
print(f"Historical Drift: {mu_hist*100:.1f}%")
print(f"Risk-Free Rate: {RF*100:.1f}%")

print(f"\n{'Metric':<25} {'Historical':<15} {'Risk-Neutral':<15}")
print("-" * 55)

print(f"{'Mean Terminal Price':<25} ${np.mean(paths_hist[:,-1]):<14.2f} ${np.mean(paths_rn[:,-1]):<14.2f}")
print(f"{'ITM Probability':<25} {itm_hist:<14.1f}% {itm_rn:<14.1f}%")
print(f"{'Option Price':<25} ${mc_hist:<14.2f} ${mc_rn:<14.2f}")

print(f"\nKey Insight:")
print(f"  Historical drift overstates option value by ${mc_hist - mc_rn:.2f} ({((mc_hist - mc_rn)/mc_rn)*100:.1f}%)")
print(f"  Professional options use risk-neutral pricing ({RF*100:.1f}%), not historical returns ({mu_hist*100:.1f}%)")

# ----------------------------------------------------------------------------
# 7. GREEKS
# ----------------------------------------------------------------------------

def greeks(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return {
        'delta': norm.cdf(d1),
        'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        'vega': S * norm.pdf(d1) * np.sqrt(T),
        'theta': -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2),
        'rho': K*T*np.exp(-r*T)*norm.cdf(d2)
    }

g = greeks(S0, K, 1.0, RF, vol_ann)
print(f"\n--- Greeks (Strike ${K:.2f}) ---")
print(f"  Delta: {g['delta']:.3f}  (price sensitivity to $1 stock move)")
print(f"  Gamma: {g['gamma']:.4f}  (delta sensitivity to $1 stock move)")
print(f"  Vega:  {g['vega']:.1f}   (price sensitivity to 1% volatility change)")
print(f"  Theta: {g['theta']:.4f}  (daily time decay)")
print(f"  Rho:   {g['rho']:.4f}    (sensitivity to 1% interest rate change)")

print("\n" + "=" * 60)
print("✅ OPTION PRICING COMPLETE!")
print("=" * 60)