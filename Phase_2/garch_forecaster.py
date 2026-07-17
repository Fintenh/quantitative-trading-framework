"""
garch_forecaster.py - Automatic GARCH model selection and volatility forecasting.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


def select_best_garch_model(returns, max_p=2, max_q=2):
    """
    Select best GARCH model by AIC.
    
    Tests: GARCH, GJR-GARCH, EGARCH with various orders.
    Returns: (model_name, fitted_model, aic)
    """
    # Build test specifications
    specs = []
    
    # Standard GARCH
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            specs.append(('Garch', p, q, False, f'GARCH({p},{q})'))
    
    # GJR-GARCH (leverage effect)
    for p in range(1, min(max_p, 2) + 1):
        for q in range(1, min(max_q, 2) + 1):
            specs.append(('GARCH', p, q, True, f'GJR-GARCH({p},{q})'))
    
    # EGARCH (exponential)
    for p in range(1, min(max_p, 2) + 1):
        for q in range(1, min(max_q, 2) + 1):
            specs.append(('EGARCH', p, q, False, f'EGARCH({p},{q})'))
    
    best_aic = float('inf')
    best_name = None
    best_model = None
    
    for vol_type, p, q, leverage, name in specs:
        try:
            model = arch_model(returns, vol=vol_type, p=p, q=q, leverage=leverage)
            fitted = model.fit(disp='off')
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_name = name
                best_model = fitted
        except:
            continue
    
    # Fallback to GARCH(1,1)
    if best_model is None:
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1)
            best_model = model.fit(disp='off')
            best_name = 'GARCH(1,1)'
            best_aic = best_model.aic
        except:
            return 'GARCH(1,1)', None, float('inf')
    
    return best_name, best_model, best_aic


def fit_garch_for_assets(returns):
    """
    Fit GARCH models for each asset, automatically selecting the best model.
    
    Returns:
        tuple: (models, model_names, aic_values)
    """
    models = {}
    model_names = {}
    aic_values = {}
    
    print("\n" + "-" * 40)
    print("AUTOMATIC GARCH MODEL SELECTION")
    print("-" * 40)
    
    for ticker in returns.columns:
        print(f"\n📊 {ticker}:")
        
        asset_returns = returns[ticker].dropna().values * 100
        
        try:
            best_name, fitted_model, best_aic = select_best_garch_model(asset_returns)
            
            if fitted_model is not None:
                models[ticker] = fitted_model
                model_names[ticker] = best_name
                aic_values[ticker] = best_aic
                print(f"  ✅ Selected: {best_name} (AIC = {best_aic:.2f})")
                
                if 'alpha[1]' in fitted_model.params:
                    alpha1 = fitted_model.params['alpha[1]']
                    beta1 = fitted_model.params['beta[1]']
                    persistence = alpha1 + beta1
                    print(f"     α₁ = {alpha1:.4f}, β₁ = {beta1:.4f}, α₁+β₁ = {persistence:.4f}")
            else:
                model = arch_model(asset_returns, vol='Garch', p=1, q=1)
                fitted = model.fit(disp='off')
                models[ticker] = fitted
                model_names[ticker] = 'GARCH(1,1)'
                aic_values[ticker] = fitted.aic
                print(f"  ⚠️ Fallback: GARCH(1,1) (AIC = {fitted.aic:.2f})")
                
        except Exception as e:
            try:
                model = arch_model(asset_returns, vol='Garch', p=1, q=1)
                fitted = model.fit(disp='off')
                models[ticker] = fitted
                model_names[ticker] = 'GARCH(1,1)'
                aic_values[ticker] = fitted.aic
                print(f"  ⚠️ Error fallback: GARCH(1,1)")
            except:
                models[ticker] = None
                model_names[ticker] = 'Failed'
                aic_values[ticker] = float('inf')
                print(f"  ❌ Model failed completely")
    
    return models, model_names, aic_values


def get_latest_volatility(models, returns):
    """
    Get latest annualised volatility for each asset.
    
    Returns:
        dict: Ticker -> annualised volatility
    """
    volatilities = {}
    
    for ticker, model in models.items():
        if model is not None:
            try:
                cond_vol = model.conditional_volatility
                if isinstance(cond_vol, np.ndarray):
                    cond_vol = pd.Series(cond_vol, index=returns.index)
                vol_daily = cond_vol.iloc[-1] / 100
                volatilities[ticker] = vol_daily * np.sqrt(252)
            except:
                volatilities[ticker] = returns[ticker].std() * np.sqrt(252)
        else:
            volatilities[ticker] = returns[ticker].std() * np.sqrt(252)
    
    return volatilities


def get_average_volatility(volatilities):
    """Average volatility across all assets."""
    return np.mean(list(volatilities.values()))


def determine_cash_allocation(avg_volatility, min_vol=0.20, max_vol=0.40, max_cash=0.50):
    """
    Linear cash allocation based on volatility.
    
    - Volatility ≤ min_vol → 0% cash
    - Volatility ≥ max_vol → max_cash% cash
    - Between → linear interpolation
    """
    if avg_volatility <= min_vol:
        return 0.0
    elif avg_volatility >= max_vol:
        return max_cash
    else:
        fraction = (avg_volatility - min_vol) / (max_vol - min_vol)
        return fraction * max_cash