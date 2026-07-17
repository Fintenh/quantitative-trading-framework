"""
performance_metrics.py - Calculate performance metrics for backtest results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_metrics(
    equity_curve: pd.Series,
    initial_capital: float = 100.0,
    risk_free_rate: float = 0.02,
    trades: Optional[list] = None
) -> Dict:
    """
    Calculate comprehensive performance metrics from an equity curve.
    
    Returns:
        Dict containing all performance metrics
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'value': equity_curve.values}, index=equity_curve.index)
    df['daily_return'] = df['value'].pct_change()
    df = df.dropna()
    
    # Returns
    total_return = (df['value'].iloc[-1] / initial_capital - 1)
    years = len(df) / 252
    annualised_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Risk
    daily_vol = df['daily_return'].std()
    annualised_vol = daily_vol * np.sqrt(252)
    
    # Drawdown
    df['cumulative'] = (1 + df['daily_return']).cumprod()
    df['running_max'] = df['cumulative'].cummax()
    df['drawdown'] = df['cumulative'] / df['running_max'] - 1
    max_drawdown = df['drawdown'].min()
    avg_drawdown = df['drawdown'].mean()
    
    # Max drawdown duration
    max_duration = 0
    drawdown_start = None
    for i in range(len(df)):
        if df['drawdown'].iloc[i] < 0:
            if drawdown_start is None:
                drawdown_start = df.index[i]
        else:
            if drawdown_start is not None:
                duration = (df.index[i] - drawdown_start).days
                if duration > max_duration:
                    max_duration = duration
                drawdown_start = None
    
    # Risk-adjusted
    sharpe = (annualised_return - risk_free_rate) / annualised_vol if annualised_vol > 0 else 0
    
    downside = df['daily_return'][df['daily_return'] < 0]
    downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (annualised_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
    
    calmar = annualised_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # Trades
    if trades:
        num_trades = len(trades)
        total_costs = sum([t.get('cost', 0) for t in trades])
    else:
        num_trades = 0
        total_costs = 0.0
    
    return {
        # Returns
        'total_return': total_return,
        'annualised_return': annualised_return,
        
        # Risk
        'annualised_volatility': annualised_vol,
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'max_drawdown_duration_days': max_duration,
        
        # Risk-adjusted
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        
        # Trades
        'num_trades': num_trades,
        'total_costs': total_costs,
        
        # Other
        'alpha': annualised_return - risk_free_rate,
        'final_value': df['value'].iloc[-1],
        'equity_curve': df,
        'drawdown_series': df['drawdown'],
    }


def format_metrics_summary(metrics: Dict) -> str:
    """Format metrics as a readable summary."""
    lines = [
        "\n" + "=" * 60,
        "PERFORMANCE METRICS SUMMARY",
        "=" * 60,
        "",
        "--- RETURNS ---",
        f"  Total Return: {metrics['total_return'] * 100:.2f}%",
        f"  Annualised Return: {metrics['annualised_return'] * 100:.2f}%",
        "",
        "--- RISK ---",
        f"  Annualised Volatility: {metrics['annualised_volatility'] * 100:.2f}%",
        f"  Maximum Drawdown: {metrics['max_drawdown'] * 100:.2f}%",
        f"  Average Drawdown: {metrics['avg_drawdown'] * 100:.2f}%",
        f"  Max Drawdown Duration: {metrics['max_drawdown_duration_days']} days",
        "",
        "--- RISK-ADJUSTED ---",
        f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}",
        f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}",
        f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}",
        "",
        "--- TRADES ---",
        f"  Number of Trades: {metrics['num_trades']}",
        f"  Total Transaction Costs: £{metrics['total_costs']:.4f}",
        "",
        "--- OTHER ---",
        f"  Alpha: {metrics['alpha'] * 100:.2f}%",
        f"  Final Value: £{metrics['final_value']:.2f}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def compare_with_benchmark(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    initial_capital: float = 100.0,
    risk_free_rate: float = 0.02
) -> Dict:
    """Compare strategy performance against a benchmark (e.g., SPY)."""
    # Ensure 1D Series
    if isinstance(strategy_equity, pd.DataFrame):
        strategy_equity = strategy_equity.iloc[:, 0]
    if isinstance(benchmark_equity, pd.DataFrame):
        benchmark_equity = benchmark_equity.iloc[:, 0]
    
    # Calculate metrics
    strategy_metrics = calculate_metrics(strategy_equity, initial_capital, risk_free_rate)
    benchmark_metrics = calculate_metrics(benchmark_equity, initial_capital, risk_free_rate)
    
    # Beta (regression)
    strategy_returns = strategy_equity.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()
    
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) > 0:
        s_aligned = strategy_returns.loc[common_idx]
        b_aligned = benchmark_returns.loc[common_idx]
        beta = np.cov(s_aligned, b_aligned)[0, 1] / np.var(b_aligned)
    else:
        beta = None
    
    return {
        'strategy': strategy_metrics,
        'benchmark': benchmark_metrics,
        'outperformance': strategy_metrics['total_return'] - benchmark_metrics['total_return'],
        'outperformance_annualised': strategy_metrics['annualised_return'] - benchmark_metrics['annualised_return'],
        'beta': beta,
    }


def format_comparison_summary(comparison: Dict) -> str:
    """Format comparison results as a readable summary."""
    s = comparison['strategy']
    b = comparison['benchmark']
    
    lines = [
        "\n" + "=" * 60,
        "STRATEGY VS BENCHMARK (SPY) COMPARISON",
        "=" * 60,
        "",
        "--- RETURNS ---",
        f"  Strategy Total Return: {s['total_return'] * 100:.2f}%",
        f"  Benchmark Total Return: {b['total_return'] * 100:.2f}%",
        f"  Outperformance: {comparison['outperformance'] * 100:.2f}%",
        "",
        f"  Strategy Annualised: {s['annualised_return'] * 100:.2f}%",
        f"  Benchmark Annualised: {b['annualised_return'] * 100:.2f}%",
        f"  Outperformance (Ann.): {comparison['outperformance_annualised'] * 100:.2f}%",
        "",
        "--- RISK ---",
        f"  Strategy Volatility: {s['annualised_volatility'] * 100:.2f}%",
        f"  Benchmark Volatility: {b['annualised_volatility'] * 100:.2f}%",
        "",
        f"  Strategy Max DD: {s['max_drawdown'] * 100:.2f}%",
        f"  Benchmark Max DD: {b['max_drawdown'] * 100:.2f}%",
        "",
        "--- RISK-ADJUSTED ---",
        f"  Strategy Sharpe: {s['sharpe_ratio']:.3f}",
        f"  Benchmark Sharpe: {b['sharpe_ratio']:.3f}",
        "",
        "--- OTHER ---",
        f"  Beta: {comparison['beta']:.3f}" if comparison['beta'] else "  Beta: N/A",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)