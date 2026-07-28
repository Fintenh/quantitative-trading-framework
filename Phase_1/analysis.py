"""
analysis.py
------------
Loads the saved Apple data and performs the Phase 1.1 (probability and
statistics) and Phase 1.2 (time series analysis) work: moments, normality
testing, stationarity testing, ACF/PACF, ARIMA for the conditional mean,
and GARCH for the conditional variance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)

print(f"Working directory: {os.getcwd()}")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("\nLoading saved data...")

csv_path = os.path.join(logs_dir, "apple_data.csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(script_dir, "apple_data.csv")
    if not os.path.exists(csv_path):
        print("Error: apple_data.csv not found. Run data_loader.py first.")
        exit(1)

price_data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
returns = price_data['Returns'].dropna()
returns_array = returns.values

print(f"Loaded {len(returns_array)} days of return data")

# ---------------------------------------------------------------------------
# Phase 1.1: Probability & statistics
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 1.1: STATISTICAL MOMENTS")
print("=" * 60)

# Sample moments
mean_return = np.mean(returns_array)
std_dev = np.std(returns_array, ddof=1)
skewness = skew(returns_array)
excess_kurt = kurtosis(returns_array)

print(f"Mean (mu):              {mean_return:.6f}")
print(f"Standard Deviation (sd): {std_dev:.6f}")
print(f"Skewness:               {skewness:.6f}")
print(f"Excess Kurtosis:        {excess_kurt:.6f}")

# Jarque-Bera test
jb_stat, jb_p = jarque_bera(returns_array)
print(f"\nJarque-Bera p-value: {jb_p:.6f}")
if jb_p < 0.05:
    print("Reject H0: returns are NOT normally distributed")
    print("(Justifies using GARCH models for volatility.)")
else:
    print("Fail to reject H0: returns could be normally distributed")

# Annualised metrics
TRADING_DAYS = 252
annual_return = mean_return * TRADING_DAYS
annual_vol = std_dev * np.sqrt(TRADING_DAYS)

print(f"\nAnnualised Return:     {annual_return * 100:.2f}%")
print(f"Annualised Volatility: {annual_vol * 100:.2f}%")

# Sharpe ratio
risk_free = 0.045
sharpe = (annual_return - risk_free) / annual_vol
print(f"Sharpe Ratio:          {sharpe:.3f}")

# ---------------------------------------------------------------------------
# Phase 1.2: Time series analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 1.2: TIME SERIES ANALYSIS")
print("=" * 60)

# --- 1.2.1: Stationarity testing (ADF) --------------------------------------

print("\n--- 1.2.1 Augmented Dickey-Fuller (ADF) Test ---")
print("H0: series has a unit root (non-stationary)")
print("H1: series is stationary")

adf_result = adfuller(returns_array)
print(f"\nADF Test Statistic: {adf_result[0]:.4f}")
print(f"p-value:            {adf_result[1]:.6f}")

if adf_result[1] < 0.05:
    print("Reject H0: returns are STATIONARY")
    print("(Validates using ARIMA and GARCH models.)")
else:
    print("Fail to reject H0: returns are NON-STATIONARY")

# --- 1.2.2: ACF and PACF plots ----------------------------------------------

print("\n--- 1.2.2 Autocorrelation (ACF) and PACF Plots ---")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(returns_array, lags=20, ax=ax1)
plot_pacf(returns_array, lags=20, ax=ax2)

ax1.set_title('Autocorrelation Function (ACF) of Returns')
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')

ax2.set_title('Partial Autocorrelation Function (PACF) of Returns')
ax2.set_xlabel('Lag')
ax2.set_ylabel('Partial Autocorrelation')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "acf_pacf.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: acf_pacf.png")

# --- 1.2.3: ARIMA model ------------------------------------------------------

print("\n--- 1.2.3 ARIMA Model for Conditional Mean ---")

arima_model = ARIMA(returns_array, order=(1, 0, 1))
arima_fit = arima_model.fit()

print(f"ARIMA(1,0,1) Summary:")
print(f"  AIC:              {arima_fit.aic:.2f}")
print(f"  BIC:              {arima_fit.bic:.2f}")
print(f"  Log-Likelihood:   {arima_fit.llf:.2f}")
print(f"  AR(1) coefficient: {arima_fit.params[1]:.4f}")
print(f"  MA(1) coefficient: {arima_fit.params[2]:.4f}")

# --- 1.2.4: GARCH model ------------------------------------------------------

print("\n--- 1.2.4 GARCH(1,1) Model for Conditional Variance ---")

# Scale returns by 100 for numerical stability
returns_scaled = returns_array * 100
garch_model = arch_model(returns_scaled, vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Extract parameters (converted back from scaled)
gamma_0 = garch_fit.params['omega']          # baseline variance
alpha_1 = garch_fit.params['alpha[1]']       # ARCH term
beta_1 = garch_fit.params['beta[1]']         # GARCH term
persistence = alpha_1 + beta_1

print(f"GARCH(1,1) Parameter Estimates:")
print(f"  Omega (baseline variance): {gamma_0:.6f}")
print(f"  Alpha (reaction to shocks): {alpha_1:.6f}")
print(f"  Beta (persistence):        {beta_1:.6f}")
print(f"  Alpha + Beta (total persistence): {persistence:.6f}")

if persistence < 1:
    print("  Stationarity constraint satisfied")
else:
    print("  WARNING: alpha + beta >= 1 (model may be non-stationary)")

# --- 1.2.5: GARCH volatility plot -------------------------------------------

print("\n--- 1.2.5 GARCH Conditional Volatility Plot ---")

# Convert back from percentage scale
cond_vol = garch_fit.conditional_volatility / 100

plt.figure(figsize=(12, 6))
plt.plot(price_data.index[1:], cond_vol, color='red', linewidth=1.5)
plt.title("GARCH(1,1) Conditional Volatility (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Volatility (Standard Deviation)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "garch_volatility.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: garch_volatility.png")

# --- 1.2.6: Volatility forecast ----------------------------------------------

print("\n--- 1.2.6 30-Day Volatility Forecast ---")

forecast = garch_fit.forecast(horizon=30)
forecast_var = forecast.variance.iloc[-1].values / 10000  # convert back
forecast_vol = np.sqrt(forecast_var)

print("\n30-Day Volatility Forecast (Annualised %):")
for i in range(0, 30, 5):
    print(f"  Day {i+1}:  {forecast_vol[i] * 100:.2f}%")
print(f"  Day 30: {forecast_vol[29] * 100:.2f}%")

# --- 1.2.7: Interpretation ---------------------------------------------------

print("\n--- 1.2.7 Interpretation ---")
print(f"  Baseline variance (omega): {gamma_0:.6f}")
print(f"  Daily baseline volatility: {np.sqrt(gamma_0 / 10000):.4f}")

print(f"\n  Reaction to shocks (alpha): {alpha_1:.4f}")
print(f"  {alpha_1 * 100:.2f}% of new shocks immediately affect volatility")

print(f"\n  Persistence (beta): {beta_1:.4f}")
print(f"  {beta_1 * 100:.2f}% of yesterday's volatility carries over")

print(f"\n  Total persistence (alpha + beta): {persistence:.4f}")
if persistence < 1:
    half_life = -np.log(2) / np.log(persistence)
    print(f"  Half-life of volatility shocks: {half_life:.1f} days")

# ---------------------------------------------------------------------------
# Complete
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)