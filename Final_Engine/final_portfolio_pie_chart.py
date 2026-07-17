"""
final_portfolio_pie_chart.py
----------------------------
Generates a pie chart showing the current portfolio allocation weights.
Reads from portfolio_log.csv to show the LATEST portfolio allocation.
EXCLUDES metadata columns like total_cost from the pie chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
import numpy as np

# ============================================================================
# SETUP
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📁 Working directory: {os.getcwd()}")

THRESHOLD_PCT = 1.0  # Group assets below 1%


def parse_weight(val):
    """
    Convert weight to float.
    Weights are stored as decimals (0.196 = 19.6%).
    Returns the weight as a percentage (19.6).
    """
    if isinstance(val, (int, float)):
        if 0 < val <= 1:
            return val * 100
        elif val > 1:
            return val
        else:
            return 0.0
    if isinstance(val, str):
        val = val.replace('%', '').strip()
        val = re.sub(r'[^0-9.]', '', val)
        if val:
            float_val = float(val)
            if 0 < float_val <= 1:
                return float_val * 100
            return float_val
    return 0.0


def get_colour_for_ticker(idx, total):
    """
    Generate a colour using idx to ensure each slice is unique.
    Uses different colour maps based on total number of slices.
    """
    if total <= 1:
        return plt.cm.Set3(0)
    elif total <= 12:
        cmap = plt.cm.Set3  # 12 distinct colours
    elif total <= 20:
        cmap = plt.cm.tab20  # 20 distinct colours
    elif total <= 30:
        cmap = plt.cm.tab20c  # 20 colours (different palette)
    else:
        cmap = plt.cm.viridis
    
    # Use idx to ensure each slice gets a different colour
    return cmap(idx / max(1, total - 1))


# ============================================================================
# LOAD DATA
# ============================================================================

def load_portfolio_data():
    """
    Load current portfolio weights from portfolio_log.csv.
    Uses the MOST RECENT entry.
    """
    log_file = os.path.join(script_dir, "logs", "portfolio_log.csv")
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("   Run final_trading_engine.py first to generate portfolio data.")
        sys.exit(1)
    
    df = pd.read_csv(log_file)
    latest = df.iloc[-1]  # Most recent entry
    
    # Exclude metadata columns
    exclude = {
        'date', 'portfolio_value', 'cash_allocation', 'realised_profit', 
        'total_wealth', 'total_cost', 'rebalance_reason', 'Unnamed: 0'
    }
    
    # Find all ticker columns (end with _weight)
    asset_cols = []
    for col in df.columns:
        if col not in exclude and not col.startswith('Unnamed'):
            if col.endswith('_weight'):
                asset_cols.append(col)
    
    # Extract weights
    weights = []
    tickers = []
    for col in asset_cols:
        ticker = col.replace('_weight', '')
        tickers.append(ticker)
        weights.append(parse_weight(latest[col]))
    
    # Get cash allocation - stored as decimal (0.068 = 6.8%)
    cash_val = parse_weight(latest.get('cash_allocation', 0))
    cash_allocation = cash_val
    
    return tickers, weights, cash_allocation, latest.get('portfolio_value', 0)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate the pie chart."""
    try:
        tickers, weights, cash_allocation, portfolio_value = load_portfolio_data()
    except Exception as e:
        print(f"❌ Error loading portfolio data: {e}")
        sys.exit(1)
    
    # Add cash as a slice
    tickers.append('Cash')
    weights.append(cash_allocation)
    
    print(f"💰 Portfolio Value: £{portfolio_value:.2f}")
    print(f"💰 Cash allocation: {cash_allocation:.2f}%")
    print(f"📊 Total: {sum(weights):.2f}%")
    
    # Check if weights are valid
    if sum(weights) < 0.1 or sum(weights) > 100.1:
        print(f"⚠️ Warning: Weights sum to {sum(weights):.2f}% - this seems wrong!")
        print("   Check the log file format.")
        return
    
    # ------------------------------------------------------------------------
    # GROUP SMALL ASSETS
    # ------------------------------------------------------------------------
    main_tickers, main_weights = [], []
    other_weight, other_count = 0.0, 0
    
    for ticker, weight in zip(tickers, weights):
        if weight >= THRESHOLD_PCT:
            main_tickers.append(ticker)
            main_weights.append(weight)
        else:
            other_weight += weight
            other_count += 1
    
    if other_weight > 0.01:
        main_tickers.append('Other')
        main_weights.append(other_weight)
    
    # ------------------------------------------------------------------------
    # PRINT DETAILED BREAKDOWN
    # ------------------------------------------------------------------------
    print("\n📊 Detailed Portfolio Breakdown:")
    sorted_items = sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_items:
        if weight > 0.01:
            print(f"  {ticker}: {weight:.2f}%")
    
    # ------------------------------------------------------------------------
    # CREATE PIE CHART - WITH UNIQUE DYNAMIC COLOURS
    # ------------------------------------------------------------------------
    labels = [f'{ticker}\n({weight:.1f}%)' for ticker, weight in zip(main_tickers, main_weights)]
    
    # Generate colours - each slice gets a unique colour
    total_items = len(main_tickers)
    colours = []
    
    for i, ticker in enumerate(main_tickers):
        if ticker == 'Other':
            colours.append('#cccccc')  # Grey for 'Other'
        elif ticker == 'Cash':
            colours.append('#000000')  # Black for 'Cash'
        else:
            # Use idx to ensure each slice gets a unique colour
            colours.append(get_colour_for_ticker(i, total_items))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        main_weights,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 0.5 else '',
        startangle=90,
        colors=colours,
        textprops={'fontsize': 11, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    
    for text in texts:
        text.set_fontsize(11)
        text.set_weight('bold')
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
    
    total_assets = len([a for a in tickers if a != 'Cash'])
    title = f'Current Portfolio Allocation\n({total_assets} Assets'
    if other_count > 0 and 'Other' in main_tickers:
        title += f', {other_count} grouped as "Other"'
    title += f')\nValue: £{portfolio_value:.2f}'
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.axis('equal')
    
    plt.figtext(0.5, 0.01,
                f'Cash: {cash_allocation:.1f}% | Current Portfolio Allocation',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/final_portfolio_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Pie chart saved to: figures/final_portfolio_pie_chart.png")


if __name__ == "__main__":
    main()