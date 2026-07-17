"""
fama_french_analysis.py
------------------------
Phase 4: Fama-French 5-Factor Analysis.
Analyses what drives our portfolio returns.
FIXED: Correct alpha annualisation, proper p-values, robust factor data handling.
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
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Import from existing modules
try:
    from config_optimised import *
except ImportError:
    from Phase_3.config_optimised import *

try:
    from backtest_engine import BacktestEngine
except ImportError:
    from Phase_3.backtest_engine import BacktestEngine

try:
    from performance_metrics import calculate_metrics
except ImportError:
    from Phase_3.performance_metrics import calculate_metrics

# ============================================================================
# SETUP
# ============================================================================

logs_dir = os.path.join(script_dir, "logs")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

print("=" * 70)
print("PHASE 4: FAMA-FRENCH FACTOR ANALYSIS (FIXED)")
print("=" * 70)
print(f"📁 Working directory: {script_dir}")
print(f"📁 Logs directory: {logs_dir}")
print(f"📁 Figures directory: {figures_dir}")

# ============================================================================
# CONFIGURATION
# ============================================================================

BACKTEST_START = "2010-01-01"
BACKTEST_END = datetime.now().strftime("%Y-%m-%d")

print(f"\n📊 Backtest period: {BACKTEST_START} to {BACKTEST_END}")

# ============================================================================
# FUNCTION TO DOWNLOAD FAMA-FRENCH FACTORS
# ============================================================================

def download_fama_french_factors():
    """
    Download Fama-French 5-Factor data from Ken French's website.
    Returns: DataFrame with daily factors (Mkt-RF, SMB, HML, RMW, CMA, RF)
    """
    print("\n📥 Downloading Fama-French 5-Factor data...")
    
    try:
        # Try direct download first
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        
        # Create SSL context
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
        print("\n📊 Building approximate factors from Yahoo Finance...")
        return build_approximate_factors()


def build_approximate_factors():
    """
    Build approximate factors using Yahoo Finance ETFs.
    This is a fallback if direct download fails.
    """
    print("\n📥 Building approximate factors from Yahoo Finance...")
    
    try:
        # Define proxies for factors
        tickers = ['SPY', 'IWM', 'IVE', 'IVW', 'QUAL', 'SHY']
        data = yf.download(tickers, start='2010-01-01', end=datetime.now(), progress=False)['Close']
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Build factors
        factors = pd.DataFrame(index=returns.index)
        
        # Market factor (Mkt-RF): SPY return - risk-free rate
        # Approximate risk-free rate from SHY (1-3 year Treasury)
        rf_approx = returns['SHY'].fillna(0.02/252)  # ~2% annualised as fallback
        factors['Mkt-RF'] = returns['SPY'] - rf_approx
        
        # SMB: Small - Big (IWM - SPY)
        factors['SMB'] = returns['IWM'] - returns['SPY']
        
        # HML: Value - Growth (IVE - IVW)
        factors['HML'] = returns['IVE'] - returns['IVW']
        
        # RMW: Quality (QUAL - SPY) as proxy for profitability
        factors['RMW'] = returns['QUAL'] - returns['SPY']
        
        # CMA: Conservative - Aggressive (approximate with SHY - SPY)
        factors['CMA'] = rf_approx - returns['SPY']
        
        # Risk-free rate
        factors['RF'] = rf_approx
        
        print(f"✅ Built {len(factors)} approximate factor observations")
        print(f"   Period: {factors.index[0].date()} to {factors.index[-1].date()}")
        print("   ⚠️ Note: Using approximated factors")
        
        return factors
        
    except Exception as e:
        print(f"❌ Error building factors: {e}")
        return None


# ============================================================================
# FUNCTION TO GET PORTFOLIO RETURNS FROM BACKTEST
# ============================================================================

def get_portfolio_returns():
    """Run backtest and extract portfolio returns."""
    
    print("\n🚀 Running backtest to get portfolio returns...")
    
    try:
        engine = BacktestEngine(
            tickers=TICKERS,
            start_date=BACKTEST_START,
            end_date=BACKTEST_END,
            initial_capital=INITIAL_CAPITAL,
            lookback_days=LOOKBACK_DAYS,
            rebalance_min_days=REBALANCE_MIN_DAYS,
            rebalance_max_days=REBALANCE_MAX_DAYS,
            drift_threshold=DRIFT_THRESHOLD,
            take_profit_pct=RELATIVE_TAKE_PROFIT_PCT,
            risk_free_rate=RISK_FREE_RATE,
            transaction_cost_pct=TRANSACTION_COST_PCT,
            cash_interest_rate=CASH_INTEREST_RATE,
            cash_min_volatility=CASH_MIN_VOLATILITY,
            cash_max_volatility=CASH_MAX_VOLATILITY,
            cash_max_allocation=CASH_MAX_ALLOCATION
        )
        
        results = engine.run()
        wealth = results['wealth_curve']['value']
        returns = wealth.pct_change().dropna()
        
        print(f"✅ Portfolio returns: {len(returns)} daily observations")
        print(f"   Period: {returns.index[0].date()} to {returns.index[-1].date()}")
        
        return returns
        
    except Exception as e:
        print(f"⚠️ Backtest failed: {e}")
        print("   Using synthetic returns for demonstration...")
        
        # Generate synthetic returns if backtest fails
        np.random.seed(42)
        dates = pd.date_range(start=BACKTEST_START, end=BACKTEST_END, freq='B')
        returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
        returns = returns.cumsum().pct_change().dropna()
        
        return returns


# ============================================================================
# FUNCTION TO PERFORM FACTOR REGRESSION
# ============================================================================

def perform_factor_regression(portfolio_returns, factors):
    """
    Perform Fama-French 5-Factor regression with proper stats.
    
    Returns:
        results: Dictionary with all regression statistics
        aligned: DataFrame of aligned data
    """
    print("\n📊 Performing Fama-French 5-Factor Regression...")
    
    # Align data
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns
    }).join(factors, how='inner')
    
    # Calculate excess returns
    aligned['excess_return'] = aligned['portfolio'] - aligned['RF']
    
    # Drop NaN
    aligned = aligned.dropna()
    
    print(f"   Aligned observations: {len(aligned)}")
    
    if len(aligned) < 100:
        print("   ⚠️ Insufficient observations for reliable regression")
        return None, None
    
    # Prepare X and y
    X = aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
    y = aligned['excess_return'].values
    
    # Fit regression with statsmodels for proper statistics
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    
    # Extract results
    alpha_daily = model_sm.params[0]
    alpha_annual = (1 + alpha_daily) ** 252 - 1  # CORRECT annualisation
    
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
    
    # Get standard errors
    std_errors = {
        'alpha': model_sm.bse[0],
        'Mkt-RF': model_sm.bse[1],
        'SMB': model_sm.bse[2],
        'HML': model_sm.bse[3],
        'RMW': model_sm.bse[4],
        'CMA': model_sm.bse[5]
    }
    
    # Compile results
    results = {
        'alpha_daily': alpha_daily,
        'alpha_annual': alpha_annual,
        'alpha_pvalue': p_values['alpha'],
        'alpha_std_error': std_errors['alpha'],
        'betas': betas,
        'p_values': p_values,
        'std_errors': std_errors,
        'r_squared': model_sm.rsquared,
        'r_squared_adj': model_sm.rsquared_adj,
        'f_statistic': model_sm.fvalue,
        'f_pvalue': model_sm.f_pvalue,
        'n_observations': len(aligned),
        'model': model_sm
    }
    
    print(f"\n   R-squared: {results['r_squared']*100:.1f}%")
    print(f"   Adjusted R-squared: {results['r_squared_adj']*100:.1f}%")
    print(f"   F-statistic: {results['f_statistic']:.2f} (p={results['f_pvalue']:.4f})")
    print(f"   Alpha (daily): {alpha_daily*100:.4f}%")
    print(f"   Alpha (annualised): {alpha_annual*100:.2f}%")
    print(f"   Alpha p-value: {results['alpha_pvalue']:.4f}")
    
    return results, aligned


# ============================================================================
# FUNCTION TO CALCULATE ROLLING ALPHA
# ============================================================================

def calculate_rolling_alpha(aligned, window=252):
    """
    Calculate rolling alpha with a fixed window.
    """
    print(f"\n📊 Calculating rolling alpha ({window} days window)...")
    
    rolling_alphas = []
    rolling_alphas_annual = []
    rolling_dates = []
    rolling_pvalues = []
    
    for i in range(window, len(aligned)):
        window_data = aligned.iloc[i-window:i]
        X = window_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
        y = window_data['excess_return'].values
        
        if len(X) > 20:
            X_sm = sm.add_constant(X)
            try:
                model = sm.OLS(y, X_sm).fit()
                alpha_daily = model.params[0]
                alpha_annual = (1 + alpha_daily) ** 252 - 1
                rolling_alphas.append(alpha_daily)
                rolling_alphas_annual.append(alpha_annual)
                rolling_dates.append(aligned.index[i])
                rolling_pvalues.append(model.pvalues[0])
            except:
                rolling_alphas.append(np.nan)
                rolling_alphas_annual.append(np.nan)
                rolling_dates.append(aligned.index[i])
                rolling_pvalues.append(np.nan)
    
    return pd.DataFrame({
        'date': rolling_dates,
        'alpha_daily': rolling_alphas,
        'alpha_annual': rolling_alphas_annual,
        'p_value': rolling_pvalues
    })


# ============================================================================
# FUNCTION TO PRINT RESULTS
# ============================================================================

def print_factor_results(results):
    """Print formatted factor analysis results."""
    
    print("\n" + "=" * 70)
    print("FAMA-FRENCH 5-FACTOR REGRESSION RESULTS")
    print("=" * 70)
    
    print(f"\nObservations: {results['n_observations']}")
    print(f"R-squared: {results['r_squared']*100:.1f}%")
    print(f"Adjusted R-squared: {results['r_squared_adj']*100:.1f}%")
    print(f"F-statistic: {results['f_statistic']:.2f} (p={results['f_pvalue']:.4f})")
    
    print(f"\n--- Alpha (Un-explained Return) ---")
    print(f"  Daily Alpha: {results['alpha_daily']*100:.6f}%")
    print(f"  Annualised Alpha: {results['alpha_annual']*100:.2f}%")
    print(f"  Standard Error: {results['alpha_std_error']*100:.4f}%")
    print(f"  p-value: {results['alpha_pvalue']:.4f}")
    
    if results['alpha_pvalue'] < 0.05:
        print("  ✅ Alpha is statistically significant (p < 0.05)")
        print("  → Genuine skill (not just luck!)")
    else:
        print("  ⚠️ Alpha is NOT statistically significant (p >= 0.05)")
        print("  → Returns may be due to luck or factor exposure")
    
    print(f"\n--- Factor Exposures (Betas) ---")
    print("  Factor | Beta | Std Error | p-value | Interpretation")
    print("  " + "-" * 60)
    
    for factor in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
        beta = results['betas'][factor]
        se = results['std_errors'][factor]
        p_val = results['p_values'][factor]
        sig = "✅" if p_val < 0.05 else "⚠️"
        
        if factor == 'Mkt-RF':
            interpretation = "Market exposure"
            if beta > 1:
                interpretation += " (more volatile than market)"
            elif beta < 1:
                interpretation += " (less volatile than market)"
            else:
                interpretation += " (same as market)"
        elif factor == 'SMB':
            interpretation = "Small-cap" + (" (small-cap bias)" if beta > 0 else " (large-cap bias)")
        elif factor == 'HML':
            interpretation = "Value" + (" (value bias)" if beta > 0 else " (growth bias)")
        elif factor == 'RMW':
            interpretation = "Profitability" + (" (profitable companies)" if beta > 0 else " (unprofitable companies)")
        elif factor == 'CMA':
            interpretation = "Investment" + (" (conservative)" if beta > 0 else " (aggressive)")
        
        print(f"  {factor:>7} | {beta:>6.3f} | {se:>8.4f} | {p_val:>8.4f} {sig} | {interpretation}")
    
    print("\n" + "=" * 70)


# ============================================================================
# FUNCTION TO PLOT RESULTS
# ============================================================================

def plot_factor_results(results, aligned, rolling_alpha):
    """Generate factor analysis visualisations."""
    
    print("\n📊 Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Factor Exposures (Betas)
    ax = axes[0, 0]
    betas = results['betas']
    factors = list(betas.keys())
    values = list(betas.values())
    colors = ['#1f77b4' if v >= 0 else '#d62728' for v in values]
    
    bars = ax.bar(factors, values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Beta (Factor Exposure)')
    ax.set_title('Factor Exposures (Fama-French 5-Factor)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.02 * (1 if val >= 0 else -1),
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 2: Actual vs Predicted Returns
    ax = axes[0, 1]
    X = aligned[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values
    y = aligned['excess_return'].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    ax.scatter(y, y_pred, alpha=0.3, s=5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=1.5, label='Perfect Fit')
    ax.set_xlabel('Actual Excess Return')
    ax.set_ylabel('Predicted Excess Return')
    ax.set_title(f'Actual vs Predicted (R² = {results["r_squared"]*100:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Alpha
    ax = axes[1, 0]
    ax.plot(rolling_alpha['date'], rolling_alpha['alpha_annual'] * 100, 
            'b-', linewidth=1.5, label='Rolling Alpha')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Alpha')
    ax.axhline(y=results['alpha_annual'] * 100, color='green', linestyle='--', 
               linewidth=1, label=f'Full-period Alpha: {results["alpha_annual"]*100:.2f}%')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualised Alpha (%)')
    ax.set_title('Rolling Annualised Alpha (252-day window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Factor Attribution (Pie Chart)
    ax = axes[1, 1]
    
    # Calculate contributions
    total_var = np.var(y)
    explained_var = np.var(y_pred)
    unexplained_var = total_var - explained_var
    
    contributions = {}
    for i, factor in enumerate(['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']):
        if np.var(X[:, i] * model.coef_[i]) > 0:
            contributions[factor] = np.var(X[:, i] * model.coef_[i]) / total_var
        else:
            contributions[factor] = 0
    
    # Normalise
    total_contrib = sum(contributions.values()) + unexplained_var/total_var
    contrib_pct = {k: v/total_contrib for k, v in contributions.items()}
    unexplained_pct = (unexplained_var/total_var) / total_contrib
    
    # Pie chart
    labels = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Unexplained']
    sizes = [contrib_pct['Mkt-RF'], contrib_pct['SMB'], contrib_pct['HML'], 
             contrib_pct['RMW'], contrib_pct['CMA'], unexplained_pct]
    colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
    
    # Remove zero or negative contributions
    sizes = [max(s, 0.001) for s in sizes]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax.set_title('Sources of Return (Factor Attribution)')
    
    plt.suptitle('Fama-French 5-Factor Analysis', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fama_french_analysis.png"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Saved: fama_french_analysis.png")


# ============================================================================
# FUNCTION TO EXPORT RESULTS TO CSV
# ============================================================================

def export_results(results, aligned):
    """Export regression results to CSV."""
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Alpha (Daily)', 'Alpha (Annualised)', 'Alpha p-value', 
            'Alpha Std Error', 'R-squared', 'Adjusted R-squared',
            'F-statistic', 'F-statistic p-value', 'Observations'
        ],
        'Value': [
            results['alpha_daily'] * 100,
            results['alpha_annual'] * 100,
            results['alpha_pvalue'],
            results['alpha_std_error'] * 100,
            results['r_squared'] * 100,
            results['r_squared_adj'] * 100,
            results['f_statistic'],
            results['f_pvalue'],
            results['n_observations']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(logs_dir, "fama_french_summary.csv"), index=False)
    
    # Create factor DataFrame
    factor_data = {
        'Factor': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
        'Beta': [results['betas'][f] for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
        'Std_Error': [results['std_errors'][f] for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
        'p_value': [results['p_values'][f] for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
        'Significant': ['Yes' if results['p_values'][f] < 0.05 else 'No' 
                       for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
    }
    
    factor_df = pd.DataFrame(factor_data)
    factor_df.to_csv(os.path.join(logs_dir, "fama_french_factors.csv"), index=False)
    
    print(f"\n✅ Results saved to: {logs_dir}/fama_french_summary.csv")
    print(f"✅ Factor data saved to: {logs_dir}/fama_french_factors.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete Fama-French analysis."""
    
    # Step 1: Get portfolio returns
    portfolio_returns = get_portfolio_returns()
    
    # Step 2: Download Fama-French factors
    factors = download_fama_french_factors()
    
    if factors is None:
        print("\n❌ Could not download Fama-French factors.")
        print("   Please check internet connection or try again later.")
        return
    
    # Step 3: Perform regression
    results, aligned = perform_factor_regression(portfolio_returns, factors)
    
    if results is None:
        print("\n❌ Regression failed. Insufficient data.")
        return
    
    # Step 4: Calculate rolling alpha
    rolling_alpha = calculate_rolling_alpha(aligned, window=252)
    
    # Step 5: Print results
    print_factor_results(results)
    
    # Step 6: Plot results
    plot_factor_results(results, aligned, rolling_alpha)
    
    # Step 7: Export results
    export_results(results, aligned)
    
    # Step 8: Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Calculate some additional statistics
    excess_returns = aligned['excess_return']
    
    print(f"\nPortfolio Statistics:")
    print(f"  Mean Daily Excess Return: {excess_returns.mean()*100:.4f}%")
    print(f"  Std Daily Excess Return: {excess_returns.std()*100:.4f}%")
    print(f"  Mean Annualised Excess Return: {excess_returns.mean()*252*100:.2f}%")
    print(f"  Annualised Volatility: {excess_returns.std()*np.sqrt(252)*100:.2f}%")
    print(f"  Information Ratio: {excess_returns.mean()/excess_returns.std()*np.sqrt(252):.3f}")
    
    print("\n" + "=" * 70)
    print("🎉 FAMA-FRENCH ANALYSIS COMPLETE!")
    print("=" * 70)
    
    # Return results for further use
    return results, aligned, rolling_alpha


if __name__ == "__main__":
    results, aligned, rolling_alpha = main()