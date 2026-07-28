"""
data_loader.py
---------------
Downloads Apple stock data and saves it to CSV, along with a few
exploratory plots (price, returns, returns distribution).
"""

import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Logs directory: {logs_dir}")
print(f"Figures directory: {figures_dir}")

# ---------------------------------------------------------------------------
# Download data
# ---------------------------------------------------------------------------

print("\nDownloading Apple stock data...")

ticker = "AAPL"
price_data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Flatten MultiIndex columns
price_data.columns = price_data.columns.droplevel(1)

print(f"Downloaded {len(price_data)} days of data")

# ---------------------------------------------------------------------------
# Calculate returns
# ---------------------------------------------------------------------------

price_data['Returns'] = price_data['Close'].pct_change()
returns = price_data['Returns'].dropna()

print(f"Returns calculated: {len(returns)} observations")

# ---------------------------------------------------------------------------
# Save data
# ---------------------------------------------------------------------------

csv_path = os.path.join(logs_dir, "apple_data.csv")
price_data.to_csv(csv_path)
print(f"Data saved to {csv_path}")

# ---------------------------------------------------------------------------
# Plot 1: closing price
# ---------------------------------------------------------------------------

print("\nGenerating price plot...")

plt.figure(figsize=(10, 5))
plt.plot(price_data['Close'], color='blue', linewidth=1.5)
plt.title("Apple Closing Price (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "apple_price.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: apple_price.png")

# ---------------------------------------------------------------------------
# Plot 2: daily returns
# ---------------------------------------------------------------------------

print("\nGenerating returns plot...")

plt.figure(figsize=(10, 5))
plt.plot(returns, color='red', linewidth=0.8)
plt.title("Apple Daily Returns (2020-2025)")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True, alpha=0.4)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "apple_returns.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: apple_returns.png")

# ---------------------------------------------------------------------------
# Plot 3: returns histogram
# ---------------------------------------------------------------------------

print("\nGenerating returns histogram...")

plt.figure(figsize=(10, 5))
plt.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='blue')
plt.title("Distribution of Apple Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "returns_histogram.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("Saved: returns_histogram.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA LOADING COMPLETE")
print("=" * 60)
print(f"Data saved to: {csv_path}")
print(f"Figures saved to: {figures_dir}")
print(f"Total observations: {len(returns)}")
print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
print("=" * 60)