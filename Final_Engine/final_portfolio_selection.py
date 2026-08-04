"""
final_portfolio_selection.py
----------------------------
Selects the optimal assets from FINAL_POOL using the same greedy algorithm as Phase 5.
Automatically updates final_config.py with the selected TICKERS, preserving the
original-style inline comments.
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
import re

from final_universe import FINAL_POOL, SECTOR_MAPPING, COMPANY_DESCRIPTION

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("FINAL PORTFOLIO SELECTION (Phase 5 Algorithm)")
print("=" * 70)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

START_DATE = "2010-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
RISK_FREE = 0.045

MOMENTUM_WEIGHT = 0.35
GROWTH_WEIGHT = 0.25
QUALITY_WEIGHT = 0.20
STABILITY_WEIGHT = 0.10
DIVERSIFICATION_WEIGHT = 0.10

MIN_ASSETS = 5
MAX_ASSETS = 20
IMPROVEMENT_THRESHOLD = 0.01
RE_EVALUATE_DAYS = 63

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(tickers, start_date, end_date):
    print(f"\nLoading data for {len(tickers)} assets...")
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

    print(f"   {len(valid_tickers)} assets with sufficient data")
    print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")

    return data, returns, valid_tickers

# ---------------------------------------------------------------------------
# Sector helper
# ---------------------------------------------------------------------------

def get_sector(ticker):
    return SECTOR_MAPPING.get(ticker, "Other")

# ---------------------------------------------------------------------------
# Asset scoring
# ---------------------------------------------------------------------------

def score_assets(returns, date=None):
    if date is not None:
        returns = returns[returns.index <= date]
    if len(returns) < 63:
        return pd.Series(0.5, index=returns.columns)

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

    if len(returns) >= 252:
        recent = returns.iloc[-126:].mean() * 252
        long_term = returns.iloc[-252:].mean() * 252
        growth = (recent - long_term) / (long_term.abs() + 0.01)
        growth = growth.clip(-1, 1)
    else:
        growth = pd.Series(0, index=returns.columns)

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

    rolling_vol = returns.rolling(63).std() * np.sqrt(252)
    avg_vol = rolling_vol.mean()
    stability = 1 / (1 + avg_vol)
    stability = (stability - stability.min()) / (stability.max() - stability.min() + 0.01)

    sector_counts = {}
    for ticker in returns.columns:
        sector = get_sector(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    diversification = pd.Series(index=returns.columns)
    for ticker in returns.columns:
        sector = get_sector(ticker)
        count = sector_counts.get(sector, 1)
        diversification[ticker] = 1 / count

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

# ---------------------------------------------------------------------------
# Portfolio optimisation helpers
# ---------------------------------------------------------------------------

def calculate_portfolio_sharpe(returns, weights=None):
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
        result = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
    except:
        pass
    return np.ones(n) / n

# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

def greedy_select_assets(returns, date=None, max_assets=20, min_assets=5):
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
    return selected

# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def run_selection(pool_tickers):
    print("\n" + "=" * 70)
    print("FINAL PORTFOLIO SELECTION")
    print("=" * 70)
    data, returns, valid_tickers = load_data(pool_tickers, START_DATE, END_DATE)
    if len(valid_tickers) < 15:
        print(f"Only {len(valid_tickers)} assets available. Need at least 15.")
        return None
    start_idx = 252
    dates = []
    current_date = returns.index[start_idx]
    while current_date < returns.index[-1]:
        dates.append(current_date)
        current_date += timedelta(days=RE_EVALUATE_DAYS)
        if current_date > returns.index[-1]:
            break
    print(f"   Re-evaluating on {len(dates)} dates")
    selection_counts = {ticker: 0 for ticker in valid_tickers}
    for i, date in enumerate(dates):
        selected = greedy_select_assets(returns, date, max_assets=MAX_ASSETS, min_assets=MIN_ASSETS)
        for ticker in selected:
            selection_counts[ticker] += 1
        if i % 10 == 0:
            print(f"   Date: {date.strftime('%Y-%m-%d')}, Assets: {len(selected)}")
    sorted_counts = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   Most frequently selected:")
    for ticker, count in sorted_counts[:10]:
        freq = count / len(dates) * 100
        print(f"      {ticker}: {freq:.1f}%")
    selected_assets = [t for t, c in sorted_counts[:15]]
    return selected_assets

# ---------------------------------------------------------------------------
# Update final_config.py
# ---------------------------------------------------------------------------

def update_final_config(selected_assets):
    print("\n" + "=" * 70)
    print("UPDATING FINAL_CONFIG.PY")
    print("=" * 70)

    config_path = os.path.join(script_dir, "final_config.py")
    with open(config_path, 'r') as f:
        content = f.read()

    ticker_lines = []
    for asset in selected_assets:
        desc = COMPANY_DESCRIPTION.get(asset, SECTOR_MAPPING.get(asset, "Other"))
        ticker_lines.append(f'    "{asset}",    # {desc}')
    new_list_content = "[\n" + "\n".join(ticker_lines) + "\n]"

    pattern = r'(TICKERS\s*=\s*)\[[^\]]*\]'
    replacement = r'\1' + new_list_content
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    if new_content == content:
        print("   Regex replacement failed, trying manual method...")
        start_pattern = r'TICKERS\s*=\s*\['
        match_start = re.search(start_pattern, content)
        if not match_start:
            print("Could not find TICKERS list")
            return False
        start_idx = match_start.end()
        bracket_count = 0
        end_idx = None
        for i in range(start_idx, len(content)):
            if content[i] == '[':
                bracket_count += 1
            elif content[i] == ']':
                if bracket_count == 0:
                    end_idx = i
                    break
                else:
                    bracket_count -= 1
        if end_idx is None:
            print("Could not find closing bracket")
            return False
        prefix = content[:match_start.end()]
        suffix = content[end_idx+1:]
        new_content = prefix + new_list_content + suffix

    with open(config_path, 'w') as f:
        f.write(new_content)

    print(f"   Updated {config_path}")
    print(f"   Selected {len(selected_assets)} assets:")
    for asset in selected_assets:
        desc = COMPANY_DESCRIPTION.get(asset, SECTOR_MAPPING.get(asset, "Other"))
        print(f"      {asset} ({desc})")
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nStarting final portfolio selection...")
    print(f"   Pool size: {len(FINAL_POOL)} assets")
    print(f"   Min assets: {MIN_ASSETS}, Max: {MAX_ASSETS}")
    print(f"   Improvement threshold: {IMPROVEMENT_THRESHOLD*100:.1f}%")

    selected_assets = run_selection(FINAL_POOL)
    if selected_assets is None or len(selected_assets) < 5:
        print("Selection failed. Not enough valid assets.")
        return

    update_final_config(selected_assets)

    pd.DataFrame({'Ticker': selected_assets}).to_csv(
        os.path.join(logs_dir, "final_selected_portfolio.csv"), index=False
    )

    print("\n" + "=" * 70)
    print("FINAL PORTFOLIO SELECTION COMPLETE")
    print("=" * 70)
    print(f"Selected {len(selected_assets)} assets")
    print(f"Updated final_config.py (preserved original-style comments)")
    print(f"Saved to logs/final_selected_portfolio.csv")

if __name__ == "__main__":
    main()