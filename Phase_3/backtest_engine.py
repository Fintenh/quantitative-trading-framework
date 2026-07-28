"""
backtest_engine.py - Simplified backtesting engine with correct holiday handling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class BacktestEngine:
    def __init__(
        self,
        tickers,
        start_date,
        end_date,
        initial_capital=100.0,
        lookback_days=252,
        rebalance_min_days=30,
        rebalance_max_days=90,
        drift_threshold=0.03,
        take_profit_pct=0.30,
        risk_free_rate=0.02,
        transaction_cost_pct=0.004,
        cash_interest_rate=0.0525,
        cash_min_volatility=0.25,
        cash_max_volatility=0.50,
        cash_max_allocation=0.30
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.lookback_days = lookback_days
        self.rebalance_min_days = rebalance_min_days
        self.rebalance_max_days = rebalance_max_days
        self.drift_threshold = drift_threshold
        self.take_profit_pct = take_profit_pct
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_pct = transaction_cost_pct

        self.cash_interest_rate = cash_interest_rate
        self.daily_cash_rate = (1 + cash_interest_rate) ** (1 / 252) - 1 if cash_interest_rate > 0 else 0

        self.cash_min_volatility = cash_min_volatility
        self.cash_max_volatility = cash_max_volatility
        self.cash_max_allocation = cash_max_allocation

        self.price_data = None
        self.spy_data = None
        self.weights = None
        self.cash_allocation = 0.0

        self.results = {}

    def _empty_results(self):
        """Return empty results when the backtest can't run."""
        print("  Backtest failed, returning empty results")
        empty_curve = pd.Series([self.initial_capital], index=[pd.Timestamp.now()])
        return {
            'final_active': self.initial_capital,
            'realised_profit': 0.0,
            'total_wealth': self.initial_capital,
            'total_return': 0.0,
            'equity_curve': pd.DataFrame({'value': empty_curve}).set_index(empty_curve.index),
            'wealth_curve': pd.DataFrame({'value': empty_curve}).set_index(empty_curve.index),
            'trades': [],
            'rebalances': []
        }

    def _get_spy_value(self, date):
        """Get SPY value for a given date, handling holidays."""
        try:
            if date in self.spy_data.index:
                val = self.spy_data.loc[date]
                if isinstance(val, pd.Series):
                    return float(val.values[0])
                return float(val)

            mask = self.spy_data.index <= date
            if mask.any():
                last_date = self.spy_data.index[mask][-1]
                val = self.spy_data.loc[last_date]
                if isinstance(val, pd.Series):
                    return float(val.values[0])
                return float(val)

            return 100.0

        except Exception as e:
            return 100.0

    def run(self):
        """Run the backtest with correct holiday handling."""
        self._print_config()
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

                if i > start_idx:
                    # Use the existing weights as-is rather than re-normalising.
                    # Assets with a missing price (e.g. a local holiday) simply
                    # contribute 0% return that day instead of having their
                    # weight redistributed across the remaining assets - this
                    # avoids introducing artificial leverage on holidays.
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

                if self.take_profit_pct > 0:
                    target = spy_value * (1 + self.take_profit_pct)
                    if active > target:
                        realised_profit, active, rebalances = self._trigger_take_profit(
                            current_date, active, spy_value, realised_profit, rebalances, i, dates
                        )
                        continue

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

    def _print_config(self):
        """Print configuration."""
        print("Starting backtest...")
        print(f"   Lookback: {self.lookback_days} days")
        print(f"   Rebalance: {self.rebalance_min_days}-{self.rebalance_max_days} days")
        print(f"   Drift: {self.drift_threshold*100:.1f}%")
        print(f"   Take-Profit: {self.take_profit_pct*100:.0f}% vs SPY")
        print(f"   Cash Interest: {self.cash_interest_rate*100:.2f}% AER")
        print(f"   Cash Max Alloc: {self.cash_max_allocation*100:.0f}%")

    def _download_data(self):
        """Download price data with buffer."""
        buffer = 400
        start = pd.to_datetime(self.start_date) - timedelta(days=buffer)

        try:
            data = yf.download(self.tickers, start=start, end=self.end_date, progress=False)["Close"]

            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = self.tickers

            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            data = data.dropna(how='all', axis=1)

            self.price_data = data

            spy_start = pd.to_datetime(self.start_date) - timedelta(days=600)
            spy = yf.download("SPY", start=spy_start, end=self.end_date, progress=False)["Close"]
            self.spy_data = spy

            print(f"Downloaded {len(data)} days, {len(data.columns)} assets")

        except Exception as e:
            print(f"  Error downloading data: {e}")
            self.price_data = None
            self.spy_data = None

    def _optimise_weights(self, returns):
        """Max Sharpe ratio portfolio."""
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        n = len(exp_ret)

        def neg_sharpe(w):
            w = np.array(w)
            ret = np.sum(exp_ret * w)
            vol = np.sqrt(w.T @ cov @ w)
            return -(ret - self.risk_free_rate) / vol if vol > 1e-10 else 999

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=constraints)
        return pd.Series(result.x, index=returns.columns)

    def _calc_cash_allocation(self, returns):
        """Cash allocation from GARCH volatility."""
        try:
            from garch_forecaster import fit_garch_for_assets, get_latest_volatility, get_average_volatility
            models, _, _ = fit_garch_for_assets(returns)
            vols = get_latest_volatility(models, returns)
            avg_vol = get_average_volatility(vols)
        except:
            avg_vol = returns.std().mean() * np.sqrt(252)

        if avg_vol <= self.cash_min_volatility:
            return 0.0
        elif avg_vol >= self.cash_max_volatility:
            return self.cash_max_allocation
        else:
            fraction = (avg_vol - self.cash_min_volatility) / (self.cash_max_volatility - self.cash_min_volatility)
            return fraction * self.cash_max_allocation

    def _calc_max_drift(self, target_weights):
        """Calculate maximum drift between current and target weights."""
        max_drift = 0.0
        drifting = None
        for ticker in self.weights.index:
            drift = abs(self.weights[ticker] - target_weights.get(ticker, 0.0))
            if drift > max_drift:
                max_drift = drift
                drifting = ticker
        return max_drift, drifting

    def _rebalance(self, date, target_weights, active):
        """Execute rebalance with transaction costs."""
        trades = []
        for ticker in self.tickers:
            current = active * self.weights.get(ticker, 0)
            target = active * target_weights.get(ticker, 0)
            diff = target - current
            if abs(diff) > 0.01:
                cost = abs(diff) * self.transaction_cost_pct / 2
                active -= cost
                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY' if diff > 0 else 'SELL',
                    'amount': abs(diff),
                    'cost': cost
                })
        return pd.Series(target_weights), active, trades

    def _trigger_take_profit(self, date, active, spy_value, realised_profit, rebalances, i, dates):
        """Trigger take-profit and lock in gains."""
        profit = active - spy_value
        print(f"TAKE-PROFIT on {date.strftime('%Y-%m-%d')}")
        print(f"   Active: £{active:.2f}, SPY: £{spy_value:.2f}")
        print(f"   Profit locked: £{profit:.2f}")

        realised_profit += profit
        active = spy_value

        window_data = self.price_data.iloc[i - self.lookback_days:i]
        window_returns = window_data.pct_change().dropna()
        if len(window_returns) >= self.lookback_days * 0.5:
            target_weights = self._optimise_weights(window_returns)
            cash_allocation = self._calc_cash_allocation(window_returns)
            adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
            self.weights, active, new_trades = self._rebalance(date, adj_weights, active)
            self.cash_allocation = cash_allocation

        rebalances.append({'date': date, 'action': 'TAKE_PROFIT'})

        return realised_profit, active, rebalances

    def _summary(self, active, realised_profit, total_wealth, trades, rebalances, equity_curve, wealth_curve):
        """Generate summary."""
        print("\nBacktest complete")
        print(f"   Final active: £{active:.2f}")
        print(f"   Realised profit: £{realised_profit:.2f}")
        print(f"   Total wealth: £{total_wealth:.2f}")
        print(f"   Return: {(total_wealth / self.initial_capital - 1) * 100:.2f}%")
        print(f"   Trades: {len(trades)}")
        print(f"   Rebalances: {len(rebalances)}")

        self.results = {
            'final_active': active,
            'realised_profit': realised_profit,
            'total_wealth': total_wealth,
            'total_return': (total_wealth / self.initial_capital - 1) * 100,
            'equity_curve': pd.DataFrame(equity_curve).set_index('date'),
            'wealth_curve': pd.DataFrame(wealth_curve).set_index('date'),
            'trades': trades,
            'rebalances': rebalances
        }

        return self.results