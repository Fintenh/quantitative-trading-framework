"""
portfolio_optimiser.py - Portfolio optimisation using Modern Portfolio Theory.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_stats(weights, exp_ret, cov):
    """Calculate expected return and volatility for a portfolio."""
    weights = np.array(weights)
    ret = np.sum(exp_ret * weights)
    vol = np.sqrt(weights.T @ cov @ weights)
    return ret, vol


def negative_sharpe(weights, exp_ret, cov, risk_free=0.02):
    """Negative Sharpe ratio (for minimisation)."""
    ret, vol = portfolio_stats(weights, exp_ret, cov)
    return -(ret - risk_free) / vol if vol > 1e-10 else 999


def portfolio_volatility(weights, exp_ret, cov):
    """Portfolio volatility only (for minimisation)."""
    _, vol = portfolio_stats(weights, exp_ret, cov)
    return vol


def optimise_portfolios(exp_ret, cov, risk_free=0.02, max_weight=1.0):
    """
    Find Minimum Variance and Maximum Sharpe Ratio portfolios.
    
    Returns:
        dict: Contains weights, returns, volatility for both portfolios
    """
    n = len(exp_ret)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, max_weight) for _ in range(n))
    init = np.ones(n) / n
    
    # Minimum Variance Portfolio
    mvp = minimize(portfolio_volatility, init, args=(exp_ret, cov),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Maximum Sharpe Ratio Portfolio
    msr = minimize(negative_sharpe, init, args=(exp_ret, cov, risk_free),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    
    mvp_ret, mvp_vol = portfolio_stats(mvp.x, exp_ret, cov)
    msr_ret, msr_vol = portfolio_stats(msr.x, exp_ret, cov)
    
    return {
        'mvp_weights': mvp.x,
        'msr_weights': msr.x,
        'mvp_return': mvp_ret,
        'mvp_volatility': mvp_vol,
        'msr_return': msr_ret,
        'msr_volatility': msr_vol,
        'tickers': exp_ret.index.tolist()
    }


def generate_portfolio_summary(results):
    """Generate a readable summary of optimisation results."""
    tickers = results['tickers']
    
    summary = []
    summary.append("\n" + "=" * 60)
    summary.append("PORTFOLIO OPTIMISATION RESULTS")
    summary.append("=" * 60)
    
    summary.append("\n--- Minimum Variance Portfolio ---")
    for ticker, w in zip(tickers, results['mvp_weights']):
        if w > 0.01:
            summary.append(f"  {ticker}: {w*100:.1f}%")
    summary.append(f"  Expected Return: {results['mvp_return']*100:.2f}%")
    summary.append(f"  Volatility: {results['mvp_volatility']*100:.2f}%")
    
    summary.append("\n--- Maximum Sharpe Ratio Portfolio ---")
    for ticker, w in zip(tickers, results['msr_weights']):
        if w > 0.01:
            summary.append(f"  {ticker}: {w*100:.1f}%")
    summary.append(f"  Expected Return: {results['msr_return']*100:.2f}%")
    summary.append(f"  Volatility: {results['msr_volatility']*100:.2f}%")
    msr_sharpe = (results['msr_return'] - 0.02) / results['msr_volatility']
    summary.append(f"  Sharpe Ratio: {msr_sharpe:.3f}")
    
    return "\n".join(summary)