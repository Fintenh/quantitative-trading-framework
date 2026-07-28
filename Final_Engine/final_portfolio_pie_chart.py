"""
final_portfolio_pie_chart.py
----------------------------
Generates a pie chart showing the current portfolio allocation weights.
Reads from portfolio_log.csv to show the LATEST portfolio allocation.
EXCLUDES metadata columns like total_cost from the pie chart.
UPDATED: Works with pound values (_value columns) instead of percentages (_weight columns).
UPDATED: Shows both pounds and percentages in output.
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


def parse_value(val):
    """
    Convert value to float.
    Values are stored as pounds (e.g., 23.51 = £23.51).
    Returns the value as a float.
    """
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.replace('£', '').replace(',', '').strip()
        val = re.sub(r'[^0-9.]', '', val)
        if val:
            return float(val)
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
    
    return cmap(idx / max(1, total - 1))


# ============================================================================
# LOAD DATA
# ============================================================================

def load_portfolio_data():
    """
    Load current portfolio data from portfolio_log.csv.
    Uses the MOST RECENT entry.
    Works with both old (_weight) and new (_value) column formats.
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
        'date', 'portfolio_value', 'cash_allocation', 'cash_pounds', 
        'realised_profit', 'total_wealth', 'total_cost', 'rebalance_reason', 
        'Unnamed: 0'
    }
    
    # Find all ticker columns - try _value first, then fall back to _weight
    asset_cols = []
    for col in df.columns:
        if col not in exclude and not col.startswith('Unnamed'):
            if col.endswith('_value') or col.endswith('_weight'):
                asset_cols.append(col)
    
    if not asset_cols:
        print("❌ No asset columns found in the log file.")
        print(f"   Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Determine format: _value (pounds) or _weight (percentage)
    is_value_format = any(col.endswith('_value') for col in asset_cols)
    
    # Extract values/weights
    values = []
    tickers = []
    portfolio_value = float(latest.get('portfolio_value', 0))
    
    for col in asset_cols:
        ticker = col.replace('_value', '').replace('_weight', '')
        tickers.append(ticker)
        val = parse_value(latest[col])
        values.append(val)
    
    # Get cash - try cash_pounds first, then cash_allocation
    cash_value = 0.0
    if 'cash_pounds' in latest:
        cash_value = parse_value(latest['cash_pounds'])
    elif 'cash_allocation' in latest:
        cash_value = parse_value(latest['cash_allocation']) * portfolio_value / 100
    
    # Store the pound values
    pound_values = values.copy()
    cash_pounds = cash_value
    
    # If we have pound values, convert to percentages
    if is_value_format:
        total_value = sum(values) + cash_value
        if total_value > 0:
            weights = [(v / total_value) * 100 for v in values]
            cash_allocation = (cash_value / total_value) * 100
        else:
            weights = [0.0] * len(values)
            cash_allocation = 0.0
    else:
        # Already percentages
        weights = values
        cash_allocation = cash_value
        # Convert percentages back to pounds for display
        pound_values = [(w / 100) * portfolio_value for w in weights]
        cash_pounds = (cash_allocation / 100) * portfolio_value
    
    return tickers, weights, pound_values, cash_allocation, cash_pounds, portfolio_value


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate the pie chart."""
    try:
        tickers, weights, pound_values, cash_allocation, cash_pounds, portfolio_value = load_portfolio_data()
    except Exception as e:
        print(f"❌ Error loading portfolio data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Add cash as a slice
    tickers.append('Cash')
    weights.append(cash_allocation)
    pound_values.append(cash_pounds)
    
    print(f"💰 Portfolio Value: £{portfolio_value:.2f}")
    print(f"💰 Cash: £{cash_pounds:.2f} ({cash_allocation:.2f}%)")
    print(f"📊 Total: {sum(weights):.2f}%")
    
    # ------------------------------------------------------------------------
    # PRINT DETAILED BREAKDOWN - SHOWING BOTH POUNDS AND PERCENTAGES
    # ------------------------------------------------------------------------
    print("\n📊 Detailed Portfolio Breakdown:")
    sorted_items = sorted(
        zip(tickers, weights, pound_values), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Print header
    print(f"{'Ticker':<12} {'Value (£)':<12} {'Allocation':<12}")
    print("-" * 40)
    
    for ticker, weight, value in sorted_items:
        if weight > 0.01:
            print(f"{ticker:<12} £{value:>10.2f} {weight:>10.2f}%")
    
    # ------------------------------------------------------------------------
    # GROUP SMALL ASSETS FOR PIE CHART
    # ------------------------------------------------------------------------
    main_tickers, main_weights, main_values = [], [], []
    other_weight, other_value, other_count = 0.0, 0.0, 0
    
    for ticker, weight, value in zip(tickers, weights, pound_values):
        if ticker == 'Cash':
            # Always include cash as a main slice
            main_tickers.append(ticker)
            main_weights.append(weight)
            main_values.append(value)
        elif weight >= THRESHOLD_PCT:
            main_tickers.append(ticker)
            main_weights.append(weight)
            main_values.append(value)
        else:
            other_weight += weight
            other_value += value
            other_count += 1
    
    if other_weight > 0.01:
        main_tickers.append('Other')
        main_weights.append(other_weight)
        main_values.append(other_value)
    
    # ------------------------------------------------------------------------
    # CREATE PIE CHART - WITH UNIQUE DYNAMIC COLOURS
    # ------------------------------------------------------------------------
    # Create labels with both percentage and pound value
    labels = []
    for ticker, weight, value in zip(main_tickers, main_weights, main_values):
        labels.append(f'{ticker}\n£{value:.2f}\n({weight:.1f}%)')
    
    # Generate colours - each slice gets a unique colour
    total_items = len(main_tickers)
    colours = []
    
    for i, ticker in enumerate(main_tickers):
        if ticker == 'Other':
            colours.append('#cccccc')  # Grey for 'Other'
        elif ticker == 'Cash':
            colours.append('#000000')  # Black for 'Cash'
        else:
            colours.append(get_colour_for_ticker(i, total_items))
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    wedges, texts, autotexts = ax.pie(
        main_weights,
        labels=labels,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 0.5 else '',
        startangle=90,
        colors=colours,
        textprops={'fontsize': 10, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    
    for text in texts:
        text.set_fontsize(10)
        text.set_weight('bold')
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
    
    total_assets = len([a for a in tickers if a != 'Cash'])
    title = f'Current Portfolio Allocation\n({total_assets} Assets'
    if other_count > 0 and 'Other' in main_tickers:
        title += f', {other_count} grouped as "Other"'
    title += f')\nTotal Value: £{portfolio_value:.2f}'
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.axis('equal')
    
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/final_portfolio_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"\n✅ Pie chart saved to: figures/final_portfolio_pie_chart.png")


if __name__ == "__main__":
    main()