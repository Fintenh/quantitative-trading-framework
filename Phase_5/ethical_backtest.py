"""
ethical_backtest.py
-------------------
Runs backtests for Original_18, Optimal_Standard, and Optimal_Ethical.
Uses the final portfolios and parameters (including Kelly lookback) from
ethical_config.py. Kelly uses the Maximum Sharpe portfolio with sign-change
detection for immediate rebalancing.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))

from backtest_engine import BacktestEngine
from performance_metrics import calculate_metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# ---------------------------------------------------------------------------
# Import from config (final portfolios & parameters)
# ---------------------------------------------------------------------------

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
    HOLDOUT_START,
    HOLDOUT_END,
    LOG_DIR,
    FIGURES_DIR,
    KELLY_BASE_CAP,
    KELLY_MAX_CAP,
)

from ethical_universe import ORIGINAL_UNIVERSE

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

UNIVERSE_COLORS = {
    "Original_18": {
        'no_rules': '#1f77b4', 'active': '#6baed6', 'total_wealth': '#9ecae1'
    },
    "Optimal_Standard": {
        'no_rules': '#ff7f0e', 'active': '#ffb347', 'total_wealth': '#ffd7a0'
    },
    "Optimal_Ethical": {
        'no_rules': '#2ca02c', 'active': '#66c266', 'total_wealth': '#a8dba8'
    },
}
SPY_COLOR = '#d62728'

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FULL_START = "2010-01-01"
FULL_END = datetime.now().strftime("%Y-%m-%d")
HOLDOUT_START = "2020-01-01"
HOLDOUT_END = datetime.now().strftime("%Y-%m-%d")

INITIAL_CAPITAL = 100.0
RISK_FREE_RATE = 0.045
TRANSACTION_COST_PCT = 0.004
CASH_INTEREST_RATE = 0.0525
CASH_MIN_VOLATILITY = 0.25
CASH_MAX_VOLATILITY = 0.50

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Working directory: {os.getcwd()}")
print(f"Logs directory: {LOG_DIR}")
print(f"Figures directory: {FIGURES_DIR}")

# ---------------------------------------------------------------------------
# Backtest engine with Kelly + sign-change detection + emergency rebalance
# ---------------------------------------------------------------------------

class BacktestEngineWithKelly(BacktestEngine):
    """
    BacktestEngine with Kelly + sign-change detection.
    Forces an immediate rebalance when f* changes sign.
    Kelly caps are read from config to match the live trading engine.
    """

    def __init__(self, kelly_lookback=126, kelly_base_cap=0.20, kelly_max_cap=1.00, **kwargs):
        super().__init__(**kwargs)
        self.kelly_lookback = kelly_lookback
        self.kelly_base_cap = kelly_base_cap      # From config
        self.kelly_max_cap = kelly_max_cap        # From config
        self.prev_f_star = None
        self.kelly_f_stars = []
        self.kelly_cash_caps = []
        self.kelly_cash_allocations = []
        self.kelly_sign_changes = 0
        self._force_rebalance = False

    def _calc_cash_allocation(self, returns):
        """
        Calculate cash allocation using Kelly + GARCH.
        Detects sign changes to trigger immediate rebalances.
        Kelly caps are taken from config, matching final_trading_engine.py.
        """
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            from portfolio_optimiser import optimise_portfolios
            from data_fetcher import calculate_annualised_stats

            # 1. Maximum Sharpe portfolio
            exp_ret, cov, _ = calculate_annualised_stats(returns)
            opt_results = optimise_portfolios(exp_ret, cov, self.risk_free_rate)
            weights = opt_results['msr_weights']
            mu = np.sum(exp_ret * weights)
            sigma = np.sqrt(weights.T @ cov @ weights)
            f_star = (mu - self.risk_free_rate) / (sigma ** 2) if sigma > 0 else 0.0

            # 2. Kelly cash cap, using the configured base/max values
            if f_star > 0:
                cash_cap = self.kelly_base_cap   # e.g. 0.20
            else:
                cash_cap = self.kelly_max_cap    # e.g. 1.00

            # 3. Sign change detection (triggers an immediate rebalance)
            current_sign = 1 if f_star > 0 else -1
            if self.prev_f_star is not None:
                prev_sign = 1 if self.prev_f_star > 0 else -1
                if current_sign != prev_sign:
                    self.kelly_sign_changes += 1
                    self._force_rebalance = True

            self.prev_f_star = f_star

            # Store for analysis
            self.kelly_f_stars.append(f_star)
            self.kelly_cash_caps.append(cash_cap)

            # 4. GARCH volatility
            models, _, _ = fit_garch_for_assets(returns)
            avg_vol = get_average_volatility(get_latest_volatility(models, returns))

        except Exception as e:
            avg_vol = returns.std().mean() * np.sqrt(252)
            cash_cap = self.cash_max_allocation  # Fallback

        # 5. Cash allocation based on GARCH volatility (bounded by the Kelly cap)
        if avg_vol <= self.cash_min_volatility:
            garch_cash = 0.0
        elif avg_vol >= self.cash_max_volatility:
            garch_cash = cash_cap
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            garch_cash = fraction * cash_cap

        self.kelly_cash_allocations.append(garch_cash)
        return garch_cash

    def should_force_rebalance(self):
        """Check if a Kelly sign change should force an immediate rebalance."""
        if self._force_rebalance:
            self._force_rebalance = False
            return True
        return False

    # ---- Override run() to incorporate emergency rebalance ----
    def run(self):
        """
        Run the backtest with Kelly emergency rebalance.
        On each day, check if a Kelly sign change has occurred.
        If so, immediately rebalance to the current target weights.
        """
        self._download_data()
        if self.price_data is None or len(self.price_data) == 0:
            return self._empty_results()

        dates = self.price_data.index
        start_idx = self.lookback_days

        if len(dates) <= start_idx:
            print(f"  Not enough data: {len(dates)} days, need {start_idx + 1}")
            return self._empty_results()

        active = self.initial_capital
        realised_profit = 0.0
        total_wealth = active + realised_profit

        self.weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)
        self.cash_allocation = 0.0

        last_rebalance = dates[start_idx]
        self.spy_start = self._get_spy_value(dates[start_idx])

        equity_curve = []
        wealth_curve = []
        trades = []
        rebalances = []

        print(f"\nRunning from {dates[start_idx]} to {dates[-1]}")
        print(f"   Initial capital: £{self.initial_capital:.2f}\n")

        for i in range(start_idx, len(dates)):
            try:
                current_date = dates[i]
                current_prices = self.price_data.loc[current_date]
                prev_prices = self.price_data.loc[dates[i-1]]

                spy_price = self._get_spy_value(current_date)
                spy_value = (spy_price / self.spy_start) * self.initial_capital

                # ---- Update active value (same as parent) ----
                if i > start_idx:
                    valid_mask = current_prices.notna() & prev_prices.notna()
                    if valid_mask.sum() > 0:
                        equity_return = np.sum(
                            self.weights[valid_mask] * (current_prices[valid_mask] / prev_prices[valid_mask] - 1)
                        )
                    else:
                        equity_return = 0.0

                    total_return = (equity_return * (1 - self.cash_allocation)) + (self.daily_cash_rate * self.cash_allocation)
                    active *= (1 + total_return)

                total_wealth = active + realised_profit

                # ---- Take-profit (inherited logic) ----
                if self.take_profit_pct > 0:
                    target = spy_value * (1 + self.take_profit_pct)
                    if active > target:
                        realised_profit, active, rebalances = self._trigger_take_profit(
                            current_date, active, spy_value, realised_profit, rebalances, i, dates
                        )
                        equity_curve.append({'date': current_date, 'value': active})
                        wealth_curve.append({'date': current_date, 'value': total_wealth})
                        continue

                # ---- Emergency rebalance due to Kelly sign change ----
                force_rebalance = self.should_force_rebalance()
                if force_rebalance:
                    # Build returns window using the current lookback
                    if i > self.lookback_days:
                        returns_window = self.price_data.iloc[i-self.lookback_days:i].pct_change().dropna()
                    else:
                        returns_window = self.price_data.iloc[:i].pct_change().dropna()

                    if len(returns_window) >= 1:
                        target_weights = self._optimise_weights(returns_window)
                        cash_allocation = self._calc_cash_allocation(returns_window)
                        adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
                        # Execute rebalance
                        print(f"EMERGENCY REBALANCE on {current_date.strftime('%Y-%m-%d')} (Kelly sign change)")
                        self.weights, active, new_trades = self._rebalance(current_date, adj_weights, active)
                        self.cash_allocation = cash_allocation
                        trades.extend(new_trades)
                        last_rebalance = current_date
                        rebalances.append({'date': current_date, 'action': 'EMERGENCY_REBALANCE'})
                        # Record state and continue to next day
                        equity_curve.append({'date': current_date, 'value': active})
                        wealth_curve.append({'date': current_date, 'value': total_wealth})
                        continue

                # ---- Normal rebalance check (same as parent) ----
                if not rebalances or rebalances[-1]['date'] != current_date:
                    days_since = (current_date - last_rebalance).days

                    if days_since >= self.rebalance_min_days:
                        window_data = self.price_data.iloc[i - self.lookback_days:i]
                        window_returns = window_data.pct_change().dropna()

                        if len(window_returns) >= self.lookback_days * 0.5:
                            target_weights = self._optimise_weights(window_returns)
                            cash_allocation = self._calc_cash_allocation(window_returns)
                            adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}

                            max_drift, drifting = self._calc_max_drift(adj_weights)

                            if max_drift > self.drift_threshold or days_since >= self.rebalance_max_days:
                                reason = f"Time-based: {days_since} days" if days_since >= self.rebalance_max_days else f"Drift-based: {drifting} drifted {max_drift*100:.2f}%"
                                print(f"REBALANCE on {current_date.strftime('%Y-%m-%d')}")
                                print(f"   Reason: {reason}")

                                self.weights, active, new_trades = self._rebalance(current_date, adj_weights, active)
                                self.cash_allocation = cash_allocation
                                trades.extend(new_trades)
                                last_rebalance = current_date
                                rebalances.append({'date': current_date, 'action': 'REBALANCE'})

                equity_curve.append({'date': current_date, 'value': active})
                wealth_curve.append({'date': current_date, 'value': total_wealth})

            except Exception as e:
                print(f"  Error on {dates[i]}: {type(e).__name__}: {e}")
                continue

        return self._summary(active, realised_profit, total_wealth, trades, rebalances, equity_curve, wealth_curve)


# ---------------------------------------------------------------------------
# Run backtest (uses the updated engine)
# ---------------------------------------------------------------------------

def run_backtest_for_universe(tickers, params, start_date, end_date, kelly_lookback, universe_name):
    """Run backtest for one universe with Kelly sign-change detection."""

    print(f"\nRunning {universe_name}...")
    print(f"   Kelly lookback: {kelly_lookback} days")
    print(f"   Kelly caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")

    # Active (with take-profit)
    engine = BacktestEngineWithKelly(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
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
        cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=kelly_lookback,
        kelly_base_cap=KELLY_BASE_CAP,
        kelly_max_cap=KELLY_MAX_CAP,
    )
    results = engine.run()
    active = results['equity_curve']['value']
    total_wealth = results['wealth_curve']['value']
    active_trades = len(results.get('trades', []))
    sign_changes = engine.kelly_sign_changes

    # No Rules (no take-profit)
    engine_nr = BacktestEngineWithKelly(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        lookback_days=params['lookback_days'],
        rebalance_min_days=params['rebalance_min_days'],
        rebalance_max_days=params['rebalance_max_days'],
        drift_threshold=params['drift_threshold'],
        take_profit_pct=0.0,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_interest_rate=CASH_INTEREST_RATE,
        cash_min_volatility=CASH_MIN_VOLATILITY,
        cash_max_volatility=CASH_MAX_VOLATILITY,
        cash_max_allocation=params['cash_max_allocation'],
        kelly_lookback=kelly_lookback,
        kelly_base_cap=KELLY_BASE_CAP,
        kelly_max_cap=KELLY_MAX_CAP,
    )
    results_nr = engine_nr.run()
    no_rules = results_nr['equity_curve']['value']
    no_rules_trades = len(results_nr.get('trades', []))
    no_rules_sign_changes = engine_nr.kelly_sign_changes

    # Print Kelly stats
    if len(engine.kelly_f_stars) > 0:
        f_stars = np.array(engine.kelly_f_stars)
        print(f"   Active: f* mean={np.mean(f_stars):.4f}, sign_changes={sign_changes}")
    if len(engine_nr.kelly_f_stars) > 0:
        f_stars_nr = np.array(engine_nr.kelly_f_stars)
        print(f"   No Rules: f* mean={np.mean(f_stars_nr):.4f}, sign_changes={no_rules_sign_changes}")

    return {
        'no_rules': no_rules,
        'active': active,
        'total_wealth': total_wealth,
        'trades': active_trades,
        'trades_no_rules': no_rules_trades,
        'sign_changes': sign_changes,
        'no_rules_sign_changes': no_rules_sign_changes,
    }


# ---------------------------------------------------------------------------
# Plot performance
# ---------------------------------------------------------------------------

def plot_performance(all_results, title, filename_linear, filename_log, period_label):
    """Plot performance in both linear and log scales."""

    fig1, ax1 = plt.subplots(figsize=(14, 10))

    for name, res in all_results.items():
        colors = UNIVERSE_COLORS[name]
        d = res[period_label]['common_dates']
        ax1.plot(d, res[period_label]['no_rules'].loc[d], color=colors['no_rules'], linestyle='-', lw=2,
                label=f'{name} - No Rules')
        ax1.plot(d, res[period_label]['active'].loc[d], color=colors['active'], linestyle='--', lw=2,
                label=f'{name} - Active')
        ax1.plot(d, res[period_label]['total_wealth'].loc[d], color=colors['total_wealth'], linestyle='-.', lw=2.5,
                label=f'{name} - Total Wealth')

    spy_d = all_results["Original_18"][period_label]['common_dates']
    ax1.plot(spy_d, all_results["Original_18"][period_label]['spy'].loc[spy_d],
            color=SPY_COLOR, linestyle='-', lw=3, label='SPY (Benchmark)')
    ax1.axhline(INITIAL_CAPITAL, color='gray', ls='-', alpha=0.3)
    ax1.set_title(title + '\n(Linear Scale)', fontsize=16, weight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Value (£)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename_linear), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {filename_linear}")

    fig2, ax2 = plt.subplots(figsize=(14, 10))

    for name, res in all_results.items():
        colors = UNIVERSE_COLORS[name]
        d = res[period_label]['common_dates']
        ax2.semilogy(d, res[period_label]['no_rules'].loc[d], color=colors['no_rules'], linestyle='-', lw=2,
                    label=f'{name} - No Rules')
        ax2.semilogy(d, res[period_label]['active'].loc[d], color=colors['active'], linestyle='--', lw=2,
                    label=f'{name} - Active')
        ax2.semilogy(d, res[period_label]['total_wealth'].loc[d], color=colors['total_wealth'], linestyle='-.', lw=2.5,
                    label=f'{name} - Total Wealth')

    spy_d = all_results["Original_18"][period_label]['common_dates']
    ax2.semilogy(spy_d, all_results["Original_18"][period_label]['spy'].loc[spy_d],
                color=SPY_COLOR, linestyle='-', lw=3, label='SPY (Benchmark)')
    ax2.axhline(INITIAL_CAPITAL, color='gray', ls='-', alpha=0.3)
    ax2.set_title(title + '\n(Log Scale)', fontsize=16, weight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Value (£) - Log Scale', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename_log), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Saved: {filename_log}")


# ---------------------------------------------------------------------------
# Helper: extract scalar from Series/DataFrame
# ---------------------------------------------------------------------------

def get_scalar_value(series_or_df):
    """Extract a scalar value from a Series or DataFrame."""
    if isinstance(series_or_df, pd.Series):
        return float(series_or_df.iloc[0])
    elif isinstance(series_or_df, pd.DataFrame):
        return float(series_or_df.iloc[0, 0])
    else:
        return float(series_or_df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ETHICAL PORTFOLIO BACKTEST (WITH SIGN CHANGE DETECTION)")
    print("=" * 70)
    print(f"Original_18: {len(ORIGINAL_UNIVERSE)} assets")
    print(f"Optimal_Standard: {len(OPTIMAL_STANDARD_PORTFOLIO)} assets")
    print(f"Optimal_Ethical: {len(OPTIMAL_ETHICAL_PORTFOLIO)} assets")
    print(f"Period: {FULL_START} to {FULL_END}")
    print(f"Holdout: {HOLDOUT_START} to {HOLDOUT_END}")
    print(f"Kelly: Maximum Sharpe portfolio with sign change detection (immediate rebalance)")
    print(f"Kelly caps: Base={KELLY_BASE_CAP:.0%}, Max={KELLY_MAX_CAP:.0%}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Define universes - reads Kelly lookback from config
    # -----------------------------------------------------------------

    universes = {
        "Original_18": {
            'tickers': ORIGINAL_UNIVERSE,
            'params': ORIGINAL_PARAMS,
            'kelly': ORIGINAL_PARAMS['kelly_lookback'],
        },
        "Optimal_Standard": {
            'tickers': OPTIMAL_STANDARD_PORTFOLIO,
            'params': OPTIMAL_STANDARD_PARAMS,
            'kelly': OPTIMAL_STANDARD_PARAMS['kelly_lookback'],
        },
        "Optimal_Ethical": {
            'tickers': OPTIMAL_ETHICAL_PORTFOLIO,
            'params': OPTIMAL_ETHICAL_PARAMS,
            'kelly': OPTIMAL_ETHICAL_PARAMS['kelly_lookback'],
        },
    }

    print("\nKelly Lookback Values (from ethical_config.py):")
    for name, config in universes.items():
        print(f"  {name}: {config['kelly']} days")
    print("=" * 70)

    all_results = {}

    for name, config in universes.items():
        print(f"\n{'='*60}")
        print(f"RUNNING: {name}")
        print(f"{'='*60}")

        # SPY data
        spy_data = yf.download("SPY", start=FULL_START, end=FULL_END, progress=False)["Close"]
        spy = (spy_data / spy_data.iloc[0]) * INITIAL_CAPITAL

        # Full period
        full = run_backtest_for_universe(
            config['tickers'],
            config['params'],
            FULL_START,
            FULL_END,
            config['kelly'],
            f"{name} - Full"
        )

        # Holdout period
        holdout = run_backtest_for_universe(
            config['tickers'],
            config['params'],
            HOLDOUT_START,
            HOLDOUT_END,
            config['kelly'],
            f"{name} - Holdout"
        )

        # Align dates
        common_full = full['total_wealth'].index.intersection(spy.index)
        common_full = common_full.intersection(full['no_rules'].index).intersection(full['active'].index)

        spy_holdout = yf.download("SPY", start=HOLDOUT_START, end=HOLDOUT_END, progress=False)["Close"]
        spy_holdout = (spy_holdout / spy_holdout.iloc[0]) * INITIAL_CAPITAL
        common_holdout = holdout['total_wealth'].index.intersection(spy_holdout.index)
        common_holdout = common_holdout.intersection(holdout['no_rules'].index).intersection(holdout['active'].index)

        all_results[name] = {
            'full': {
                'no_rules': full['no_rules'].loc[common_full],
                'active': full['active'].loc[common_full],
                'total_wealth': full['total_wealth'].loc[common_full],
                'spy': spy.loc[common_full],
                'common_dates': common_full,
                'trades': full['trades'],
                'trades_no_rules': full['trades_no_rules'],
                'sign_changes': full['sign_changes'],
                'no_rules_sign_changes': full['no_rules_sign_changes'],
            },
            'holdout': {
                'no_rules': holdout['no_rules'].loc[common_holdout],
                'active': holdout['active'].loc[common_holdout],
                'total_wealth': holdout['total_wealth'].loc[common_holdout],
                'spy': spy_holdout.loc[common_holdout],
                'common_dates': common_holdout,
                'trades': holdout['trades'],
                'trades_no_rules': holdout['trades_no_rules'],
                'sign_changes': holdout['sign_changes'],
                'no_rules_sign_changes': holdout['no_rules_sign_changes'],
            }
        }

    # -----------------------------------------------------------------
    # Print comparison tables
    # -----------------------------------------------------------------

    for period, label in [('full', 'FULL PERIOD (2010-2026)'), ('holdout', 'HOLDOUT PERIOD (2020-2026)')]:
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY - {label}")
        print(f"{'='*70}")

        print("| Universe | Assets | Kelly | No Rules | Active | Total Wealth | SPY | Sharpe (Active) | Max DD | Trades | Sign Chg |")
        print("|----------|--------|-------|----------|--------|--------------|-----|-----------------|--------|--------|----------|")

        for name, res in all_results.items():
            wealth = res[period]['total_wealth']
            returns = wealth.pct_change().dropna()

            # Extract scalar values using the helper function
            no_rules_val = get_scalar_value(res[period]['no_rules'].iloc[-1])
            active_val = get_scalar_value(res[period]['active'].iloc[-1])
            total_val = get_scalar_value(res[period]['total_wealth'].iloc[-1])
            spy_val = get_scalar_value(res[period]['spy'].iloc[-1])

            no_rules_return = float((no_rules_val / INITIAL_CAPITAL - 1) * 100)
            active_return = float((active_val / INITIAL_CAPITAL - 1) * 100)
            total_return = float((total_val / INITIAL_CAPITAL - 1) * 100)
            spy_return = float((spy_val / INITIAL_CAPITAL - 1) * 100)

            sharpe = float((returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252)))
            max_dd = float((wealth / wealth.cummax() - 1).min() * 100)
            sign_changes = res[period]['sign_changes']

            print(f"| {name:<14} | {len(universes[name]['tickers']):>6} | {universes[name]['kelly']:>5} | "
                  f"{no_rules_return:>8.1f}% | {active_return:>6.1f}% | {total_return:>12.1f}% | "
                  f"{spy_return:>3.1f}% | {sharpe:>15.3f} | {max_dd:>6.2f}% | {res[period]['trades']:>6} | {sign_changes:>8} |")

    # -----------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------

    print("\nGenerating plots...")

    plot_performance(
        all_results,
        'Optimal Standard vs Optimal Ethical vs Original Portfolio Performance\nFull Period (2010-2026)',
        'optimal_portfolio_comparison_full_linear.png',
        'optimal_portfolio_comparison_full_log.png',
        'full'
    )

    plot_performance(
        all_results,
        'Optimal Standard vs Optimal Ethical vs Original Portfolio Performance\nHoldout Period (2020-2026)',
        'optimal_portfolio_comparison_holdout_linear.png',
        'optimal_portfolio_comparison_holdout_log.png',
        'holdout'
    )

    # -----------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------

    print("\nSaving CSVs...")

    for period, label in [('full', 'full'), ('holdout', 'holdout')]:
        common = all_results["Original_18"][period]['common_dates']
        for name in all_results.keys():
            common = common.intersection(all_results[name][period]['common_dates'])

        df = pd.DataFrame({'date': common})
        for name in all_results.keys():
            prefix = name.lower().replace('_', '_')
            r = all_results[name][period]
            df[f'{prefix}_no_rules'] = r['no_rules'].loc[common].values.flatten()
            df[f'{prefix}_active'] = r['active'].loc[common].values.flatten()
            df[f'{prefix}_total_wealth'] = r['total_wealth'].loc[common].values.flatten()
            df[f'{prefix}_sign_changes'] = r['sign_changes']
        df['spy_benchmark'] = all_results["Original_18"][period]['spy'].loc[common].values.flatten()
        df.to_csv(os.path.join(LOG_DIR, f"ethical_backtest_results_{label}.csv"), index=False)
        print(f"ethical_backtest_results_{label}.csv")

    print("\n" + "=" * 70)
    print("ETHICAL BACKTEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()