"""
ethical_pie_chart.py
--------------------
Generates pie charts showing actual portfolio allocation weights from optimisation.
- Plot 1: Original universe (18 assets) - benchmark
- Plot 2: Optimal Standard vs Optimal Ethical side by side (20 assets each)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from ethical_universe import ORIGINAL_UNIVERSE, STANDARD_UNIVERSE, ETHICAL_UNIVERSE
from ethical_config import *

# Import from Phase 2 for optimisation
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))
from data_fetcher import fetch_price_data, calculate_returns, calculate_annualised_stats
from portfolio_optimiser import optimise_portfolios
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility, determine_cash_allocation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# LOOKBACK DAYS (from ethical_config)
# ============================================================================

# Use the lookback from ORIGINAL_PARAMS (290 days)
LOOKBACK_DAYS = ORIGINAL_PARAMS['lookback_days']

# ============================================================================
# COLOUR MAP FOR TICKERS
# ============================================================================

def get_colour_for_ticker(ticker, idx):
    """Get a consistent colour for a ticker."""
    colours = plt.cm.tab20(np.linspace(0, 1, 20))
    return colours[idx % 20]


def get_portfolio_weights(tickers, lookback_days=LOOKBACK_DAYS):
    """
    Get optimised portfolio weights for a given universe.
    Uses the same method as Phases 1-4.
    """
    print(f"   Optimising {len(tickers)} assets...")
    
    # Fetch data
    prices = fetch_price_data(tickers, lookback_days)
    returns = calculate_returns(prices)
    exp_ret, cov, _ = calculate_annualised_stats(returns)
    
    # Optimise portfolio (Maximum Sharpe Ratio)
    opt_results = optimise_portfolios(exp_ret, cov, RISK_FREE_RATE)
    weights = opt_results['msr_weights']
    
    # Calculate cash allocation from GARCH
    try:
        models, _, _ = fit_garch_for_assets(returns)
        vols = get_latest_volatility(models, returns)
        avg_vol = get_average_volatility(vols)
        cash_pct = determine_cash_allocation(avg_vol, CASH_MIN_VOLATILITY, CASH_MAX_VOLATILITY, CASH_MAX_ALLOCATION)
    except:
        cash_pct = 0.0
    
    # Adjust weights for cash
    weights = np.array(weights) * (1 - cash_pct)
    
    # Create dictionary of ticker -> weight
    weight_dict = {ticker: weights[i] for i, ticker in enumerate(tickers) if weights[i] > 0.001}
    weight_dict['Cash'] = cash_pct
    
    # Sort by weight descending
    weight_dict = dict(sorted(weight_dict.items(), key=lambda x: x[1], reverse=True))
    
    return weight_dict, opt_results, cash_pct


def create_pie_chart(weight_dict, title, filename, max_labels=20):
    """
    Create a pie chart from a weight dictionary.
    Groups small weights (< 1%) into 'Other'.
    """
    # Group small weights
    threshold = 0.01  # 1% threshold
    main_items = {}
    other_weight = 0.0
    other_count = 0
    
    for ticker, weight in weight_dict.items():
        if weight >= threshold:
            main_items[ticker] = weight
        else:
            other_weight += weight
            other_count += 1
    
    if other_weight > 0.001:
        main_items[f'Other ({other_count} assets)'] = other_weight
    
    # Prepare data
    labels = list(main_items.keys())
    values = list(main_items.values())
    
    # Cap number of labels shown
    if len(labels) > max_labels:
        # Keep top max_labels-1, group rest
        top_labels = labels[:max_labels-1]
        top_values = values[:max_labels-1]
        remaining_weight = sum(values[max_labels-1:])
        top_labels.append(f'Other ({len(labels) - max_labels + 1} assets)')
        top_values.append(remaining_weight)
        labels = top_labels
        values = top_values
    
    # Create colours
    colours = []
    for i, label in enumerate(labels):
        if 'Other' in label:
            colours.append('#cccccc')
        elif label == 'Cash':
            colours.append('#000000')
        else:
            colours.append(get_colour_for_ticker(label, i))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 1.5 else '',
        startangle=90,
        colors=colours,
        textprops={'fontsize': 9},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Make autotexts smaller
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved: {filename}")
    
    return main_items


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("ETHICAL PORTFOLIO PIE CHARTS (ACTUAL WEIGHTS)")
print("=" * 70)
print(f"📁 Lookback Period: {LOOKBACK_DAYS} days")
print(f"📁 Original Universe: {len(ORIGINAL_UNIVERSE)} assets")
print(f"📁 Optimal Standard Universe: {len(STANDARD_UNIVERSE)} assets")
print(f"📁 Optimal Ethical Universe: {len(ETHICAL_UNIVERSE)} assets")
print("=" * 70)

# ----------------------------------------------------------------------------
# GET WEIGHTS FOR EACH UNIVERSE
# ----------------------------------------------------------------------------

print("\n📊 Calculating portfolio weights...")

# Original Universe
print("\n--- Original Universe (18 assets) ---")
original_weights, original_opt, original_cash = get_portfolio_weights(ORIGINAL_UNIVERSE)

# Optimal Standard Universe (20 assets)
print("\n--- Optimal Standard Universe (20 assets) ---")
standard_weights, standard_opt, standard_cash = get_portfolio_weights(STANDARD_UNIVERSE)

# Optimal Ethical Universe (20 assets)
print("\n--- Optimal Ethical Universe (20 assets) ---")
ethical_weights, ethical_opt, ethical_cash = get_portfolio_weights(ETHICAL_UNIVERSE)

# ----------------------------------------------------------------------------
# PLOT 1: Original Universe (Benchmark)
# ----------------------------------------------------------------------------

print("\n📊 Generating Original Universe pie chart...")

create_pie_chart(
    original_weights,
    f'Original Portfolio (18 assets)\nCash: {original_cash*100:.1f}%',
    'original_pie_chart.png'
)

print("\nOriginal Universe Top Holdings:")
for ticker, weight in list(original_weights.items())[:10]:
    print(f"  {ticker}: {weight*100:.2f}%")

# ----------------------------------------------------------------------------
# PLOT 2: Optimal Standard vs Optimal Ethical Side by Side
# ----------------------------------------------------------------------------

print("\n📊 Generating Optimal Standard vs Optimal Ethical side-by-side pie charts...")

fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Helper to plot on an axis
def plot_pie_on_axis(ax, weight_dict, title, max_labels=20):
    """Plot a pie chart on a given axis."""
    # Group small weights
    threshold = 0.01
    main_items = {}
    other_weight = 0.0
    other_count = 0
    
    for ticker, weight in weight_dict.items():
        if weight >= threshold:
            main_items[ticker] = weight
        else:
            other_weight += weight
            other_count += 1
    
    if other_weight > 0.001:
        main_items[f'Other ({other_count} assets)'] = other_weight
    
    labels = list(main_items.keys())
    values = list(main_items.values())
    
    # Cap labels
    if len(labels) > max_labels:
        top_labels = labels[:max_labels-1]
        top_values = values[:max_labels-1]
        remaining_weight = sum(values[max_labels-1:])
        top_labels.append(f'Other ({len(labels) - max_labels + 1} assets)')
        top_values.append(remaining_weight)
        labels = top_labels
        values = top_values
    
    # Colours
    colours = []
    for i, label in enumerate(labels):
        if 'Other' in label:
            colours.append('#cccccc')
        elif label == 'Cash':
            colours.append('#000000')
        else:
            colours.append(get_colour_for_ticker(label, i))
    
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 1.5 else '',
        startangle=90,
        colors=colours,
        textprops={'fontsize': 8},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(7)
    
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.axis('equal')
    
    return main_items

# Optimal Standard
plot_pie_on_axis(
    axes[0],
    standard_weights,
    f'Optimal Standard Portfolio (20 assets)\nCash: {standard_cash*100:.1f}%'
)

# Optimal Ethical
plot_pie_on_axis(
    axes[1],
    ethical_weights,
    f'Optimal Ethical Portfolio (20 assets)\nCash: {ethical_cash*100:.1f}%'
)

plt.suptitle('Optimal Standard vs Optimal Ethical Portfolio Allocation (Optimised Weights)', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "optimal_standard_vs_ethical_pie_charts.png"), dpi=150, bbox_inches='tight')
plt.show()
plt.close()

print(f"✅ Saved: optimal_standard_vs_ethical_pie_charts.png")

# ----------------------------------------------------------------------------
# PRINT SUMMARIES
# ----------------------------------------------------------------------------

print("\n" + "=" * 70)
print("PORTFOLIO WEIGHTS SUMMARY")
print("=" * 70)

print(f"\nOriginal Universe (18 assets) - Cash: {original_cash*100:.1f}%")
print("Top 10 holdings:")
for ticker, weight in list(original_weights.items())[:10]:
    print(f"  {ticker}: {weight*100:.2f}%")

print(f"\nOptimal Standard Universe ({len(STANDARD_UNIVERSE)} assets) - Cash: {standard_cash*100:.1f}%")
print("Top 10 holdings:")
for ticker, weight in list(standard_weights.items())[:10]:
    print(f"  {ticker}: {weight*100:.2f}%")

print(f"\nOptimal Ethical Universe ({len(ETHICAL_UNIVERSE)} assets) - Cash: {ethical_cash*100:.1f}%")
print("Top 10 holdings:")
for ticker, weight in list(ethical_weights.items())[:10]:
    print(f"  {ticker}: {weight*100:.2f}%")

print("\n" + "=" * 70)
print("🎉 PIE CHARTS COMPLETE!")
print("=" * 70)