"""
kelly_lookback_optimiser_ethical.py
------------------------------------
Finds the optimal Kelly lookback for Optimal_Standard and Optimal_Ethical.
Calculates the Kelly fraction from actual equal-weight portfolio returns:
f* = (portfolio_return - risk_free_return) / volatility^2
Sign changes trigger early rebalances.
"""

import os
import sys
import warnings
import re
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics
from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility

from ethical_config import (
    OPTIMAL_STANDARD_PORTFOLIO,
    OPTIMAL_ETHICAL_PORTFOLIO,
    OPTIMAL_STANDARD_PARAMS,
    OPTIMAL_ETHICAL_PARAMS,
    RISK_FREE_RATE,
    TRANSACTION_COST_PCT,
    CASH_INTEREST_RATE,
    CASH_MIN_VOLATILITY,
    CASH_MAX_VOLATILITY,
    CASH_MAX_ALLOCATION,
    START_DATE,
    HOLDOUT_END,
    LOG_DIR,
    FIGURES_DIR,
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 5: KELLY LOOKBACK OPTIMISATION - ETHICAL PORTFOLIOS")
print("=" * 70)
print("Kelly fraction is calculated from actual portfolio returns")
print("f* = (portfolio_return - risk_free_return) / volatility^2")
print("Sign changes trigger early rebalances")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Kelly lookbacks to test (extended range for ethical portfolios)
KELLY_LOOKBACKS = [60, 75, 90, 105, 120, 126, 135, 150, 165, 180, 200, 252,
                   280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 504]

# Backtest period - use 2016 onwards (Phase 3 validated period)
BACKTEST_START = "2016-01-01"
BACKTEST_END = HOLDOUT_END

print(f"\nTesting {len(KELLY_LOOKBACKS)} lookback values")
print(f"Period: {BACKTEST_START} to {BACKTEST_END}")

# ---------------------------------------------------------------------------
# Backtest engine with Kelly
# ---------------------------------------------------------------------------

class BacktestEngineWithKelly(BacktestEngine):
    """
    Phase 5 backtest engine using equal-weight portfolio returns with
    actual compounded returns for the Kelly calculation.
    """

    def __init__(self, kelly_lookback=126, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.prev_f_star = None
        self.kelly_f_stars = []
        self.kelly_sign_changes = 0
        self._force_rebalance = False
        self._debug_count = 0

    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using Kelly.

        f* = (portfolio_return - risk_free_return) / volatility^2
        """
        try:
            # 1. GARCH volatility
            models, _, _ = fit_garch_for_assets(returns)
            vols = get_latest_volatility(models, returns)
            avg_vol = get_average_volatility(vols)

            # 2. Get the lookback window
            if len(returns) >= self.kelly_lookback:
                kelly_returns = returns.iloc[-self.kelly_lookback:].copy()
            else:
                kelly_returns = returns.copy()

            days = len(kelly_returns)

            # 3. Calculate equal-weight portfolio returns
            portfolio_returns = kelly_returns.mean(axis=1)

            # 4. Calculate the actual portfolio return over the period (compounded)
            portfolio_return = (1 + portfolio_returns).prod() - 1

            # 5. Risk-free return over the same period
            risk_free_return = (1 + self.risk_free_rate) ** (days / 252) - 1

            # 6. Volatility over the period
            volatility = portfolio_returns.std()

            # 7. Excess return
            excess_return = portfolio_return - risk_free_return

            # 8. Kelly fraction
            if volatility > 0 and np.isfinite(volatility) and volatility < 1.0:
                f_star = excess_return / (volatility ** 2)
            else:
                f_star = 0.0

            # 9. Check for sign change (triggers rebalance)
            current_sign = 1 if f_star > 0 else -1
            if self.prev_f_star is not None:
                prev_sign = 1 if self.prev_f_star > 0 else -1
                if current_sign != prev_sign:
                    self.kelly_sign_changes += 1
                    self._force_rebalance = True

            self.prev_f_star = f_star

            # 10. Kelly cash cap
            if f_star > 0:
                cash_cap = 0.20  # Positive edge: 20% cash
            else:
                cash_cap = 1.00  # Negative edge: 100% cash

            # Store for debugging
            self.kelly_f_stars.append(f_star)

            # Debug output for the first 10 iterations
            if self._debug_count < 10:
                self._debug_count += 1
                print(f"\n    Kelly Debug #{self._debug_count} (lookback={self.kelly_lookback}):")
                print(f"      Days: {days}")
                print(f"      Portfolio Return: {portfolio_return:.6f} ({portfolio_return*100:.4f}%)")
                print(f"      Risk-Free Return: {risk_free_return:.6f} ({risk_free_return*100:.4f}%)")
                print(f"      Excess Return: {excess_return:.6f} ({excess_return*100:.4f}%)")
                print(f"      Volatility: {volatility:.6f}")
                print(f"      f*: {f_star:.4f}")
                print(f"      Sign: {'+' if f_star > 0 else '-'}")
                print(f"      Cash Cap: {cash_cap:.0%}")

        except Exception as e:
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = self.cash_max_allocation
            if self._debug_count < 10:
                print(f"    Kelly Error: {e}")
                self._debug_count += 1

        # 11. Cash allocation based on GARCH volatility
        if avg_vol <= self.cash_min_volatility:
            return 0.0
        elif avg_vol >= self.cash_max_volatility:
            return cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            return fraction * cash_cap

    def should_force_rebalance(self):
        """Check if a Kelly sign change should force a rebalance."""
        if self._force_rebalance:
            self._force_rebalance = False
            return True
        return False


# ---------------------------------------------------------------------------
# Update config
# ---------------------------------------------------------------------------

def update_config_kelly(universe_name, best_kelly):
    """Update KELLY_LOOKBACK in ethical_config.py."""
    config_path = os.path.join(script_dir, "ethical_config.py")

    with open(config_path, 'r') as f:
        content = f.read()

    if universe_name == "Optimal_Standard":
        param_name = "OPTIMAL_STANDARD_PARAMS"
    elif universe_name == "Optimal_Ethical":
        param_name = "OPTIMAL_ETHICAL_PARAMS"
    else:
        return

    best_kelly_int = int(best_kelly)
    pattern = rf"({param_name} = {{[^}}]*'kelly_lookback': )\d+"
    replacement = rf"\g<1>{best_kelly_int}"
    new_content = re.sub(pattern, replacement, content)

    with open(config_path, 'w') as f:
        f.write(new_content)

    print(f"   Updated {param_name} kelly_lookback to {best_kelly_int}")


# ---------------------------------------------------------------------------
# Run Kelly optimisation
# ---------------------------------------------------------------------------

def run_kelly_optimisation(tickers, params, universe_name):
    print(f"\n{'='*70}")
    print(f"RUNNING: {universe_name} ({len(tickers)} assets)")
    print(f"{'='*70}")

    all_results = []

    for kelly_lookback in KELLY_LOOKBACKS:
        print(f"\n  Testing: {kelly_lookback} days...", end="", flush=True)

        try:
            engine = BacktestEngineWithKelly(
                tickers=tickers,
                start_date=BACKTEST_START,
                end_date=BACKTEST_END,
                initial_capital=100.0,
                lookback_days=params['lookback_days'],
                rebalance_min_days=params['rebalance_min_days'],
                rebalance_max_days=params['rebalance_max_days'],
                drift_threshold=params['drift_threshold'],
                take_profit_pct=0.0,  # No Rules
                risk_free_rate=RISK_FREE_RATE,
                transaction_cost_pct=TRANSACTION_COST_PCT,
                cash_interest_rate=CASH_INTEREST_RATE,
                cash_min_volatility=CASH_MIN_VOLATILITY,
                cash_max_volatility=CASH_MAX_VOLATILITY,
                cash_max_allocation=params['cash_max_allocation'],
                kelly_lookback=kelly_lookback
            )

            results = engine.run()
            equity = results['equity_curve']['value']
            trades = len(results.get('trades', []))

            metrics = calculate_metrics(
                equity_curve=equity,
                initial_capital=100.0,
                risk_free_rate=RISK_FREE_RATE
            )

            # Calculate f* statistics
            f_stars = np.array(engine.kelly_f_stars) if engine.kelly_f_stars else np.array([0])

            result_dict = {
                'kelly_lookback': kelly_lookback,
                'total_return_pct': metrics['total_return'] * 100,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': trades,
                'f_star_mean': np.mean(f_stars),
                'f_star_positive': np.sum(f_stars > 0),
                'f_star_negative': np.sum(f_stars <= 0),
                'sign_changes': engine.kelly_sign_changes,
            }
            all_results.append(result_dict)

            print(f" Return: {result_dict['total_return_pct']:.1f}%, Sharpe: {result_dict['sharpe_ratio']:.3f}")
            print(f"    f* mean: {result_dict['f_star_mean']:.4f}")
            print(f"    f* > 0: {result_dict['f_star_positive']}, f* <= 0: {result_dict['f_star_negative']}")
            print(f"    Sign Changes: {result_dict['sign_changes']}")
            print(f"    Trades: {result_dict['num_trades']}")

        except Exception as e:
            print(f" ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) == 0:
        return None

    df_results = pd.DataFrame(all_results)

    print(f"\n  Results Summary:")
    print(df_results[['kelly_lookback', 'total_return_pct', 'sharpe_ratio', 'sign_changes', 'num_trades']].to_string(index=False))

    best_sharpe = df_results.loc[df_results['sharpe_ratio'].idxmax()]
    best_return = df_results.loc[df_results['total_return_pct'].idxmax()]

    print(f"\n  Best Sharpe: {best_sharpe['kelly_lookback']}d (Sharpe: {best_sharpe['sharpe_ratio']:.3f})")
    print(f"  Best Return: {best_return['kelly_lookback']}d ({best_return['total_return_pct']:.2f}%)")

    safe_name = universe_name.replace(" ", "_")
    df_results.to_csv(os.path.join(LOG_DIR, f"kelly_lookback_results_{safe_name}.csv"), index=False)

    return {
        'universe_name': universe_name,
        'df_results': df_results,
        'best_sharpe': best_sharpe,
        'best_return': best_return,
    }


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------

def plot_kelly_results(standard_results, ethical_results):
    """Plot results across the four comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'Optimal_Standard': '#ff7f0e',
        'Optimal_Ethical': '#2ca02c'
    }

    # Plot 1: return vs lookback - Optimal_Standard
    ax = axes[0, 0]
    if standard_results is not None:
        df = standard_results['df_results']
        ax.plot(df['kelly_lookback'], df['total_return_pct'], 'o-', color=colors['Optimal_Standard'], linewidth=2, markersize=6)
        ax.set_xlabel('Kelly Lookback (days)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Optimal_Standard - Return vs Lookback')
        ax.grid(True, alpha=0.3)

        best = df.loc[df['sharpe_ratio'].idxmax()]
        ax.axvline(x=best['kelly_lookback'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.annotate(f'Best: {best["kelly_lookback"]}d\nSharpe: {best["sharpe_ratio"]:.3f}',
                   xy=(best['kelly_lookback'], best['total_return_pct']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No results', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Optimal_Standard - Return vs Lookback')

    # Plot 2: return vs lookback - Optimal_Ethical
    ax = axes[0, 1]
    if ethical_results is not None:
        df = ethical_results['df_results']
        ax.plot(df['kelly_lookback'], df['total_return_pct'], 'o-', color=colors['Optimal_Ethical'], linewidth=2, markersize=6)
        ax.set_xlabel('Kelly Lookback (days)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Optimal_Ethical - Return vs Lookback')
        ax.grid(True, alpha=0.3)

        best = df.loc[df['sharpe_ratio'].idxmax()]
        ax.axvline(x=best['kelly_lookback'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.annotate(f'Best: {best["kelly_lookback"]}d\nSharpe: {best["sharpe_ratio"]:.3f}',
                   xy=(best['kelly_lookback'], best['total_return_pct']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'No results', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Optimal_Ethical - Return vs Lookback')

    # Plot 3: Sharpe ratio vs lookback (both universes)
    ax = axes[1, 0]
    if standard_results is not None:
        df = standard_results['df_results']
        ax.plot(df['kelly_lookback'], df['sharpe_ratio'], 'o-', color=colors['Optimal_Standard'], linewidth=2, markersize=6, label='Optimal_Standard')
    if ethical_results is not None:
        df = ethical_results['df_results']
        ax.plot(df['kelly_lookback'], df['sharpe_ratio'], 'o-', color=colors['Optimal_Ethical'], linewidth=2, markersize=6, label='Optimal_Ethical')

    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: sign changes vs lookback (both universes)
    ax = axes[1, 1]
    if standard_results is not None:
        df = standard_results['df_results']
        ax.plot(df['kelly_lookback'], df['sign_changes'], 'o-', color=colors['Optimal_Standard'], linewidth=2, markersize=6, label='Optimal_Standard')
    if ethical_results is not None:
        df = ethical_results['df_results']
        ax.plot(df['kelly_lookback'], df['sign_changes'], 'o-', color=colors['Optimal_Ethical'], linewidth=2, markersize=6, label='Optimal_Ethical')

    ax.set_xlabel('Kelly Lookback (days)')
    ax.set_ylabel('Number of Sign Changes')
    ax.set_title('Sign Changes vs Kelly Lookback')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Kelly Lookback Optimisation - Ethical Portfolios', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "kelly_lookback_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: kelly_lookback_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL KELLY LOOKBACKS")
    print("=" * 70)
    print("Kelly fraction is calculated from actual portfolio returns")
    print("f* = (portfolio_return - risk_free_return) / volatility^2")
    print("Sign changes trigger early rebalances")
    print("=" * 70)

    universes = {
        'Optimal_Standard': {
            'tickers': OPTIMAL_STANDARD_PORTFOLIO,
            'params': OPTIMAL_STANDARD_PARAMS,
        },
        'Optimal_Ethical': {
            'tickers': OPTIMAL_ETHICAL_PORTFOLIO,
            'params': OPTIMAL_ETHICAL_PARAMS,
        }
    }

    results_dict = {}

    for name, config in universes.items():
        results = run_kelly_optimisation(
            config['tickers'],
            config['params'],
            name
        )
        results_dict[name] = results

    print("\n" + "=" * 70)
    print("UPDATING ETHICAL_CONFIG.PY")
    print("=" * 70)

    for name, results in results_dict.items():
        if results is not None:
            best_kelly = results['best_sharpe']['kelly_lookback']
            update_config_kelly(name, best_kelly)

    plot_kelly_results(
        results_dict.get('Optimal_Standard'),
        results_dict.get('Optimal_Ethical')
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print("\n| Universe | Optimal Lookback | Return | Sharpe | Max DD | Trades | Sign Changes |")
    print("|----------|------------------|--------|--------|--------|--------|--------------|")

    for name, results in results_dict.items():
        if results is not None:
            best = results['best_sharpe']
            print(f"| {name:<14} | {best['kelly_lookback']:>16} | {best['total_return_pct']:>6.1f}% | {best['sharpe_ratio']:>6.3f} | {best['max_drawdown']*100:>6.2f}% | {best['num_trades']:>6} | {best['sign_changes']:>12} |")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()