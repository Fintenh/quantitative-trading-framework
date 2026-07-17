"""
fama_french_ethical.py
------------------------
Phase 4: Fama-French 5-Factor Analysis for Optimal Portfolios.
Compares factor exposures and alpha between Original_18, Optimal_Standard, and Optimal_Ethical.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
import urllib.request
import ssl
import zipfile
import io
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Import from existing modules
from config_optimised import *
from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics

# ============================================================================
# IMPORT FROM ETHICAL_CONFIG (FINAL PORTFOLIOS)
# ============================================================================

from ethical_universe import ORIGINAL_UNIVERSE
from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO,
    OPTIMAL_ETHICAL_PORTFOLIO,
    OPTIMAL_STANDARD_PARAMS,
    OPTIMAL_ETHICAL_PARAMS,
    ORIGINAL_PARAMS,
    RISK_FREE_RATE,
    TRANSACTION_COST_PCT,
    CASH_INTEREST_RATE,
    CASH_MIN_VOLATILITY,
    CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION,
    START_DATE,
    HOLDOUT_END,
)

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("PHASE 4: FAMA-FRENCH ANALYSIS (OPTIMAL PORTFOLIOS)")
print("=" * 70)
print(f"📁 Working directory: {script_dir}")
print(f"📁 Logs directory: {logs_dir}")
print(f"📁 Figures directory: {figures_dir}")

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKTEST_START = START_DATE
BACKTEST_END = HOLDOUT_END
INITIAL_CAPITAL = 100.0

# ============================================================================
# OPTIMISED PARAMETERS FOR EACH UNIVERSE
# ============================================================================

# Original_18 (from Phase 3 config_optimised.py)
ORIGINAL_BACKTEST_PARAMS = {
    'lookback_days': 290,
    'rebalance_min_days': 55,
    'rebalance_max_days': 85,
    'drift_threshold': 0.035,
    'take_profit_pct': 0.174,
    'cash_max_allocation': 0.20,
}

# Optimal_Standard (from ethical_config)
OPTIMAL_STANDARD_BACKTEST_PARAMS = {
    'lookback_days': OPTIMAL_STANDARD_PARAMS['lookback_days'],
    'rebalance_min_days': OPTIMAL_STANDARD_PARAMS['rebalance_min_days'],
    'rebalance_max_days': OPTIMAL_STANDARD_PARAMS['rebalance_max_days'],
    'drift_threshold': OPTIMAL_STANDARD_PARAMS['drift_threshold'],
    'take_profit_pct': OPTIMAL_STANDARD_PARAMS['take_profit_pct'],
    'cash_max_allocation': OPTIMAL_STANDARD_PARAMS['cash_max_allocation'],
}

# Optimal_Ethical (from ethical_config)
OPTIMAL_ETHICAL_BACKTEST_PARAMS = {
    'lookback_days': OPTIMAL_ETHICAL_PARAMS['lookback_days'],
    'rebalance_min_days': OPTIMAL_ETHICAL_PARAMS['rebalance_min_days'],
    'rebalance_max_days': OPTIMAL_ETHICAL_PARAMS['rebalance_max_days'],
    'drift_threshold': OPTIMAL_ETHICAL_PARAMS['drift_threshold'],
    'take_profit_pct': OPTIMAL_ETHICAL_PARAMS['take_profit_pct'],
    'cash_max_allocation': OPTIMAL_ETHICAL_PARAMS['cash_max_allocation'],
}

# ============================================================================
# DOWNLOAD FAMA-FRENCH FACTORS
# ============================================================================

def download_fama_french_factors():
    """Download Fama-French 5-Factor data."""
    print("\n📥 Downloading Fama-French 5-Factor data...")
    
    try:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(url, context=ssl_context) as response:
            zip_data = response.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
            csv_filename = [f for f in zip_file.namelist() if f.endswith('.csv')][0]
            with zip_file.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file, skiprows=6, header=0, index_col=0)
        
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df = df / 100
        df.columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        
        print(f"✅ Downloaded {len(df)} daily observations")
        print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        print(f"⚠️ Direct download failed: {e}")
        return build_approximate_factors()


def build_approximate_factors():
    """Build approximate factors using Yahoo Finance ETFs."""
    print("\n📥 Building approximate factors from Yahoo Finance...")
    
    try:
        tickers = ['SPY', 'IWM', 'IVE', 'IVW', 'QUAL', 'SHY']
        data = yf.download(tickers, start='2010-01-01', end=datetime.now(), progress=False)['Close']
        returns = data.pct_change().dropna()
        
        factors = pd.DataFrame(index=returns.index)
        rf_approx = returns['SHY'].fillna(0.02/252)
        factors['Mkt-RF'] = returns['SPY'] - rf_approx
        factors['SMB'] = returns['IWM'] - returns['SPY']
        factors['HML'] = returns['IVE'] - returns['IVW']
        factors['RMW'] = returns['QUAL'] - returns['SPY']
        factors['CMA'] = rf_approx - returns['SPY']
        factors['RF'] = rf_approx
        
        print(f"✅ Built {len(factors)} approximate factor observations")
        return factors
        
    except Exception as e:
        print(f"❌ Error building factors: {e}")
        return None


# ============================================================================
# GET PORTFOLIO RETURNS
# ============================================================================

def get_portfolio_returns(tickers, params, universe_name):
    """Run backtest and extract portfolio returns for a universe."""
    
    print(f"\n🚀 Running backtest for {universe_name}...")
    
    try:
        engine = BacktestEngine(
            tickers=tickers,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
            initial_capital=INITIAL_CAPITAL,
            lookback_days=params['lookback_days'],
            rebalance_min_days=params['rebalance_min_days'],
            rebalance_max_days=params['rebalance_max_days'],
            drift_threshold=params['drift_threshold'],
            take_profit_pct=params['take_profit_pct'],
            risk_free_rate=RISK_FREE_RATE,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            cash_interest_rate=CASH_INTEREST_RATE,
            cash_min_volatility=CASH_MIN_VOLATILITY,
            cash_max_volatility=CASH_MAX_VOLATILITY,
            cash_max_allocation=params['cash_max_allocation']
        )
        
        results = engine.run()
        wealth = results['wealth_curve']['value']
        returns = wealth.pct_change().dropna()
        
        print(f"✅ {universe_name}: {len(returns)} daily observations")
        print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
        
        return returns
        
    except Exception as e:
        print(f"⚠️ Backtest failed for {universe_name}: {e}")
        return None


# ============================================================================
# PERFORM FACTOR REGRESSION
# ============================================================================

def perform_factor_regression(portfolio_returns, factors, universe_name):
    """Perform Fama-French 5-Factor regression."""
    
    print(f"\n📊 Performing regression for {universe_name}...")
    
    if portfolio_returns is None or factors is None:
        return None, None
    
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns
    }).join(factors, how='inner')
    
    aligned['excess_return'] = aligned['portfolio'] - aligned['RF']
    aligned = aligned.dropna()
    
    if len(aligned) < 100:
        print(f"   ⚠️ Insufficient observations: {len(aligned)}")
        return None, None
    
    X = aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
    y = aligned['excess_return'].values
    
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    
    alpha_daily = model_sm.params[0]
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    
    betas = {
        'Mkt-RF': model_sm.params[1],
        'SMB': model_sm.params[2],
        'HML': model_sm.params[3],
        'RMW': model_sm.params[4],
        'CMA': model_sm.params[5]
    }
    
    p_values = {
        'alpha': model_sm.pvalues[0],
        'Mkt-RF': model_sm.pvalues[1],
        'SMB': model_sm.pvalues[2],
        'HML': model_sm.pvalues[3],
        'RMW': model_sm.pvalues[4],
        'CMA': model_sm.pvalues[5]
    }
    
    results = {
        'universe_name': universe_name,
        'alpha_daily': alpha_daily,
        'alpha_annual': alpha_annual,
        'alpha_pvalue': p_values['alpha'],
        'betas': betas,
        'p_values': p_values,
        'r_squared': model_sm.rsquared,
        'r_squared_adj': model_sm.rsquared_adj,
        'f_statistic': model_sm.fvalue,
        'f_pvalue': model_sm.f_pvalue,
        'n_observations': len(aligned),
        'model': model_sm
    }
    
    print(f"   R-squared: {results['r_squared']*100:.1f}%")
    print(f"   Alpha (annualised): {alpha_annual*100:.2f}%")
    print(f"   Alpha p-value: {results['alpha_pvalue']:.4f}")
    
    return results, aligned


# ============================================================================
# PLOT COMPARISON
# ============================================================================

def plot_factor_comparison(all_results):
    """Plot factor exposures comparison across universes."""
    
    print("\n📊 Generating factor comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'Original_18': '#1f77b4',
        'Optimal_Standard': '#ff7f0e',
        'Optimal_Ethical': '#2ca02c'
    }
    
    # Plot 1: Alpha comparison
    ax = axes[0, 0]
    names = []
    alphas = []
    p_values = []
    for name, results in all_results.items():
        if results is not None:
            names.append(name)
            alphas.append(results['alpha_annual'] * 100)
            p_values.append(results['alpha_pvalue'])
    
    bars = ax.bar(names, alphas, color=[colors.get(n, '#888888') for n in names], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Annualised Alpha (%)', fontsize=10)
    ax.set_title('Alpha Comparison Across Portfolios', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, p_val in zip(bars, p_values):
        sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'n.s.'
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.5 if bar.get_height() >= 0 else -0.5),
                f'p={p_val:.3f}\n{sig}', ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=8)
    
    # Plot 2: Factor Exposures
    ax = axes[0, 1]
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    x = np.arange(len(factors))
    width = 0.25
    
    for i, (name, results) in enumerate(all_results.items()):
        if results is not None:
            betas = [results['betas'][f] for f in factors]
            offset = (i - len(all_results)/2 + 0.5) * width
            ax.bar(x + offset, betas, width, label=name, color=colors.get(name, '#888888'), alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Factor', fontsize=10)
    ax.set_ylabel('Beta (Factor Exposure)', fontsize=10)
    ax.set_title('Factor Exposures Comparison', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: R-squared comparison
    ax = axes[1, 0]
    r2_values = []
    for name, results in all_results.items():
        if results is not None:
            r2_values.append({
                'Universe': name,
                'R-squared': results['r_squared'] * 100
            })
    
    if r2_values:
        r2_df = pd.DataFrame(r2_values)
        bars = ax.bar(r2_df['Universe'], r2_df['R-squared'], 
                     color=[colors.get(n, '#888888') for n in r2_df['Universe']], alpha=0.7)
        ax.set_ylabel('R-squared (%)', fontsize=10)
        ax.set_title('Model Fit (R-squared) Comparison', fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, r2_df['R-squared']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for name, results in all_results.items():
        if results is not None:
            row = [
                name,
                f"{results['alpha_annual']*100:.2f}%",
                f"{results['alpha_pvalue']:.4f}",
                f"{results['r_squared']*100:.1f}%"
            ]
            table_data.append(row)
    
    if table_data:
        columns = ['Portfolio', 'Alpha', 'p-value', 'R²']
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center',
                        colColours=['#f0f0f0']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('Factor Regression Summary', fontsize=12, weight='bold', pad=20)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(figures_dir, "fama_french_optimal_comparison.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved: fama_french_optimal_comparison.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run Fama-French analysis for all optimal portfolios."""
    
    factors = download_fama_french_factors()
    if factors is None:
        print("❌ Could not download factors. Exiting.")
        return
    
    universes = {
        'Original_18': (ORIGINAL_UNIVERSE, ORIGINAL_BACKTEST_PARAMS),
        'Optimal_Standard': (OPTIMAL_STANDARD_PORTFOLIO, OPTIMAL_STANDARD_BACKTEST_PARAMS),
        'Optimal_Ethical': (OPTIMAL_ETHICAL_PORTFOLIO, OPTIMAL_ETHICAL_BACKTEST_PARAMS)
    }
    
    all_results = {}
    all_aligned = {}
    
    for name, (tickers, params) in universes.items():
        returns = get_portfolio_returns(tickers, params, name)
        if returns is not None:
            results, aligned = perform_factor_regression(returns, factors, name)
            if results is not None:
                all_results[name] = results
                all_aligned[name] = aligned
    
    print("\n" + "=" * 70)
    print("FAMA-FRENCH FACTOR ANALYSIS SUMMARY (OPTIMAL PORTFOLIOS)")
    print("=" * 70)
    
    print("\n| Portfolio | Alpha (Ann.) | p-value | Mkt-RF | SMB | HML | RMW | CMA | R² |")
    print("|-----------|--------------|---------|--------|-----|-----|-----|-----|-----|")
    
    for name, results in all_results.items():
        b = results['betas']
        print(f"| {name:<10} | {results['alpha_annual']*100:>12.2f}% | {results['alpha_pvalue']:>7.4f} | "
              f"{b['Mkt-RF']:>6.3f} | {b['SMB']:>4.3f} | {b['HML']:>4.3f} | {b['RMW']:>4.3f} | {b['CMA']:>4.3f} | {results['r_squared']*100:>4.1f}% |")
    
    if len(all_results) > 0:
        plot_factor_comparison(all_results)
        
        results_df = []
        for name, results in all_results.items():
            results_df.append({
                'Portfolio': name,
                'Alpha_Annual': results['alpha_annual'] * 100,
                'Alpha_pvalue': results['alpha_pvalue'],
                'Mkt-RF': results['betas']['Mkt-RF'],
                'SMB': results['betas']['SMB'],
                'HML': results['betas']['HML'],
                'RMW': results['betas']['RMW'],
                'CMA': results['betas']['CMA'],
                'R_squared': results['r_squared'] * 100,
                'Observations': results['n_observations']
            })
        
        pd.DataFrame(results_df).to_csv(os.path.join(logs_dir, "fama_french_optimal_results.csv"), index=False)
        print(f"\n✅ Results saved: fama_french_optimal_results.csv")
    
    print("\n" + "=" * 70)
    print("🎉 FAMA-FRENCH OPTIMAL PORTFOLIO ANALYSIS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()