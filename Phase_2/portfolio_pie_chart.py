"""
portfolio_pie_chart.py
----------------------
Generate a publication-quality pie chart for the Overleaf report.
Reads from rebalance_log.csv to show the TARGET portfolio allocation.
EXCLUDES metadata columns like total_cost from the pie chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"📁 Working directory: {os.getcwd()}")

THRESHOLD_PCT = 1.0  # 1%

# Colour map for assets
COLOURS = {
    'AAPL': '#1f77b4', 'MSFT': '#ff7f0e', 'GOOGL': '#2ca02c',
    'AMZN': '#d62728', 'TSLA': '#9467bd', 'META': '#8c564b',
    'NVDA': '#e377c2', 'JPM': '#7f7f7f', 'JNJ': '#bcbd22',
    'XOM': '#17becf', 'SPY': '#1f77b4', 'QQQ': '#ff7f0e',
    'EFA': '#2ca02c', 'EEM': '#d62728', 'TLT': '#9467bd',
    'LQD': '#8c564b', 'GLD': '#e377c2', 'DBC': '#7f7f7f',
    'Cash': '#000000'
}


def parse_weight(val):
    """Convert weight to float, handling strings with % signs."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.replace('%', '').strip()
        val = re.sub(r'[^0-9.]', '', val)
        return float(val) if val else 0.0
    return 0.0


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------

def load_portfolio_data():
    """
    Load target portfolio weights from rebalance_log.csv.
    Falls back to portfolio_log.csv if rebalance log doesn't exist.
    EXCLUDES metadata columns.
    """
    # Try rebalance_log first (target weights)
    log_file = "logs/rebalance_log.csv"
    
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        latest = df.iloc[-1]
        
        # Exclude ALL metadata columns
        exclude = {
            'date', 
            'cash_allocation', 
            'portfolio_value', 
            'total_cost',
            'rebalance_reason',  # ← ADD THIS!
            'Unnamed: 0'         # ← ADD THIS (if it exists)
        }
        
        # Only include columns that are actual asset tickers
        asset_cols = [col for col in df.columns if col not in exclude and not col.startswith('Unnamed')]
        
        assets = asset_cols
        weights = [parse_weight(latest[col]) * 100 for col in asset_cols]
        cash_allocation = parse_weight(latest['cash_allocation']) * 100 if 'cash_allocation' in df.columns else 0.0
        
        return assets, weights, cash_allocation
    
    # Fallback: portfolio_log.csv (current weights, need to scale)
    log_file = "logs/portfolio_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        latest = df.iloc[-1]
        
        # Exclude metadata columns
        exclude = {'date', 'portfolio_value', 'cash_allocation', 'Unnamed: 0'}
        asset_cols = [col for col in df.columns if col.endswith('_weight') and col not in exclude]
        assets = [col.replace('_weight', '') for col in asset_cols]
        weights = [parse_weight(latest[col]) for col in asset_cols]
        
        cash_allocation = parse_weight(latest['cash_allocation']) if 'cash_allocation' in df.columns else 0.0
        
        # Scale to pre-cash (assets sum to 100% - cash)
        equity_pct = sum(weights)
        if equity_pct > 0:
            scaling_factor = 100 / equity_pct
            weights = [w * scaling_factor for w in weights]
        
        return assets, weights, cash_allocation
    
    raise FileNotFoundError("No log file found. Run trading_engine.py first.")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    """Generate the pie chart."""
    try:
        assets, weights, cash_allocation = load_portfolio_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Add cash as a slice
    assets.append('Cash')
    weights.append(cash_allocation)
    
    print(f"💰 Cash allocation: {cash_allocation:.2f}%")
    print(f"📊 Total: {sum(weights):.2f}%")
    
    # Check if weights are valid
    if sum(weights) < 0.1 or sum(weights) > 100.1:
        print(f"⚠️ Warning: Weights sum to {sum(weights):.2f}% - this seems wrong!")
        print("   Check the log file format.")
        print(f"   Asset columns: {assets}")
        print(f"   Weights: {weights}")
        return
    
    # ------------------------------------------------------------------------
    # GROUP SMALL ASSETS
    # ------------------------------------------------------------------------
    main_assets, main_weights = [], []
    other_weight, other_count = 0.0, 0
    
    for asset, weight in zip(assets, weights):
        if weight >= THRESHOLD_PCT:
            main_assets.append(asset)
            main_weights.append(weight)
        else:
            other_weight += weight
            other_count += 1
    
    if other_weight > 0.01:
        main_assets.append('Other')
        main_weights.append(other_weight)
    
    # ------------------------------------------------------------------------
    # CREATE PIE CHART
    # ------------------------------------------------------------------------
    labels = [f'{asset}\n({weight:.1f}%)' for asset, weight in zip(main_assets, main_weights)]
    
    colours = []
    for asset in main_assets:
        if asset == 'Other':
            colours.append('#cccccc')
        elif asset == 'Cash':
            colours.append('#000000')
        elif asset in COLOURS:
            colours.append(COLOURS[asset])
        else:
            colours.append('#cccccc')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts = ax.pie(
        main_weights,
        labels=labels,
        autopct=None,
        startangle=90,
        colors=colours,
        textprops={'fontsize': 11, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    
    for text in texts:
        text.set_fontsize(11)
        text.set_weight('bold')
    
    total_assets = len([a for a in assets if a != 'Cash'])
    title = f'Phase 2 Target Portfolio Allocation\n({total_assets} Assets'
    if other_count > 0 and 'Other' in main_assets:
        title += f', {other_count} grouped as "Other"'
    title += ')'
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.axis('equal')
    
    plt.figtext(0.5, 0.01,
                f'Cash: {cash_allocation:.1f}% | Target Allocation',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/phase2_pie_chart.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Target Portfolio Allocation:")
    for asset, weight in zip(main_assets, main_weights):
        print(f"  {asset}: {weight:.1f}%")
    if other_count > 0:
        print(f"  ({other_count} other assets grouped as 'Other')")
    print(f"\n💰 Cash Allocation: {cash_allocation:.2f}%")


if __name__ == "__main__":
    main()