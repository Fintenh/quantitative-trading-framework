"""
backtest_engine.py - Simplified backtesting engine with trailing stop-loss and catch-up reinvestment.
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
        stop_loss_pct=0.25,
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
        self.stop_loss_pct = stop_loss_pct
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
        self.in_cash = False
        self.stop_loss_triggered = False
        self.current_peak = None
        self.stop_loss_level = None
        
        # Catch-up tracking
        self.stop_loss_value = None
        self.stop_loss_spy = None
        self.stop_loss_date = None
        self.no_rules_value = None
        
        self.results = {}

    def run(self):
        """Run the backtest with trailing stop-loss and catch-up reinvestment."""
        self._print_config()
        self._download_data()
        
        dates = self.price_data.index
        start_idx = self.lookback_days
        
        active = self.initial_capital
        realised_profit = 0.0
        total_wealth = active + realised_profit
        
        self.weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)
        self.cash_allocation = 0.0
        
        # Trailing stop-loss
        self.current_peak = self.initial_capital
        self.stop_loss_level = self.current_peak * (1 - self.stop_loss_pct)
        print(f"   Stop-Loss Level: £{self.stop_loss_level:.2f}")
        
        last_rebalance = dates[start_idx]
        self.in_cash = False
        self.stop_loss_triggered = False
        
        # Reset catch-up tracking
        self.stop_loss_value = None
        self.stop_loss_spy = None
        self.stop_loss_date = None
        self.no_rules_value = None
        
        self.spy_start = float(self.spy_data.iloc[start_idx].values[0])
        
        equity_curve = []
        wealth_curve = []
        trades = []
        rebalances = []
        
        print(f"\n📈 Running from {dates[start_idx]} to {dates[-1]}")
        print(f"   Initial capital: £{self.initial_capital:.2f}\n")
        
        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            current_prices = self.price_data.loc[current_date]
            
            # SPY value
            spy_price = float(self.spy_data.loc[current_date].values[0])
            spy_value = (spy_price / self.spy_start) * self.initial_capital
            
            # ------------------------------------------------------------
            # UPDATE PORTFOLIO VALUE
            # ------------------------------------------------------------
            if i > start_idx and not self.in_cash:
                prev_prices = self.price_data.loc[dates[i-1]]
                equity_return = np.sum(self.weights * (current_prices / prev_prices - 1))
                total_return = (equity_return * (1 - self.cash_allocation)) + (self.daily_cash_rate * self.cash_allocation)
                active *= (1 + total_return)
            
            total_wealth = active + realised_profit
            
            # Update no-rules value
            if self.stop_loss_value is not None and self.stop_loss_spy is not None:
                self.no_rules_value = self.stop_loss_value * (spy_value / self.stop_loss_spy)
            else:
                self.no_rules_value = active
            
            # ------------------------------------------------------------
            # UPDATE PEAK FOR TRAILING STOP-LOSS
            # ------------------------------------------------------------
            if not self.in_cash and active > self.current_peak:
                self.current_peak = active
                self.stop_loss_level = self.current_peak * (1 - self.stop_loss_pct)
            
            # ------------------------------------------------------------
            # STOP-LOSS CHECK
            # ------------------------------------------------------------
            if self.stop_loss_pct > 0 and not self.in_cash and not self.stop_loss_triggered:
                if active < self.stop_loss_level:
                    self._trigger_stop_loss(current_date, active, spy_value)
                    rebalances.append({'date': current_date, 'action': 'STOP_LOSS'})
                    continue
            
            # ------------------------------------------------------------
            # TAKE-PROFIT CHECK
            # ------------------------------------------------------------
            if self.take_profit_pct > 0 and not self.in_cash:
                target = spy_value * (1 + self.take_profit_pct)
                if active > target:
                    realised_profit, active, rebalances = self._trigger_take_profit(
                        current_date, active, spy_value, realised_profit, rebalances, i, dates
                    )
                    continue
            
            # ------------------------------------------------------------
            # REINVEST FROM CASH (Catch-up logic)
            # ------------------------------------------------------------
            if self.in_cash and self.stop_loss_value is not None:
                days_since_stop = (current_date - self.stop_loss_date).days
                should_reinvest = self._check_reinvest_condition(
                    active, self.no_rules_value, spy_value, self.stop_loss_spy, days_since_stop
                )
                
                if should_reinvest:
                    active, trades, rebalances = self._reinvest_from_cash(
                        current_date, i, active, rebalances
                    )
                    last_rebalance = current_date
            
            # ------------------------------------------------------------
            # REBALANCE (drift or time)
            # ------------------------------------------------------------
            if not self.in_cash:
                days_since = (current_date - last_rebalance).days
                
                if days_since >= self.rebalance_min_days:
                    window_returns = self.price_data.iloc[i - self.lookback_days:i].pct_change().dropna()
                    
                    if len(window_returns) >= self.lookback_days * 0.5:
                        target_weights = self._optimise_weights(window_returns)
                        cash_allocation = self._calc_cash_allocation(window_returns)
                        adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
                        
                        max_drift, drifting = self._calc_max_drift(adj_weights)
                        
                        if max_drift > self.drift_threshold or days_since >= self.rebalance_max_days:
                            reason = f"Time-based: {days_since} days" if days_since >= self.rebalance_max_days else f"Drift-based: {drifting} drifted {max_drift*100:.2f}%"
                            
                            print(f"🔄 REBALANCE on {current_date.strftime('%Y-%m-%d')}")
                            print(f"   Reason: {reason}")
                            
                            self.weights, active, new_trades = self._rebalance(current_date, adj_weights, active)
                            self.cash_allocation = cash_allocation
                            trades.extend(new_trades)
                            last_rebalance = current_date
                            rebalances.append({'date': current_date, 'action': 'REBALANCE'})
            
            # Record
            equity_curve.append({'date': current_date, 'value': active})
            wealth_curve.append({'date': current_date, 'value': total_wealth})
        
        # ------------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------------
        return self._summary(active, realised_profit, total_wealth, trades, rebalances, equity_curve, wealth_curve)

    # ============================================================================
    # PRIVATE METHODS
    # ============================================================================

    def _print_config(self):
        """Print configuration."""
        print("🚀 Starting backtest...")
        print(f"   Lookback: {self.lookback_days} days")
        print(f"   Rebalance: {self.rebalance_min_days}-{self.rebalance_max_days} days")
        print(f"   Drift: {self.drift_threshold*100:.1f}%")
        print(f"   Stop-Loss: {self.stop_loss_pct*100:.0f}% (trailing with catch-up)")
        print(f"   Take-Profit: {self.take_profit_pct*100:.0f}% vs SPY")
        print(f"   Cash Interest: {self.cash_interest_rate*100:.2f}% AER")
        print(f"   Cash Max Alloc: {self.cash_max_allocation*100:.0f}%")

    def _download_data(self):
        """Download price data."""
        buffer = 400
        start = pd.to_datetime(self.start_date) - timedelta(days=buffer)
        
        data = yf.download(self.tickers, start=start, end=self.end_date, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = self.tickers
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        data = data.dropna(how='all')
        self.price_data = data
        
        spy = yf.download("SPY", start=start, end=self.end_date, progress=False)["Close"]
        self.spy_data = spy
        print(f"✅ Downloaded {len(data)} days")

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
        """Cash allocation from volatility."""
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
        """Execute rebalance."""
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

    def _trigger_stop_loss(self, date, active, spy_value):
        """Trigger stop-loss and move to cash."""
        print(f"🔴 STOP-LOSS on {date.strftime('%Y-%m-%d')}")
        print(f"   Peak: £{self.current_peak:.2f}")
        print(f"   Stop-Loss Level: £{self.stop_loss_level:.2f}")
        print(f"   Current Value: £{active:.2f}")
        print(f"   Drop: {(1 - active/self.current_peak)*100:.1f}%")
        
        self.stop_loss_value = active
        self.stop_loss_spy = spy_value
        self.stop_loss_date = date
        self.no_rules_value = active
        
        self.weights = pd.Series(0, index=self.tickers)
        self.cash_allocation = 1.0
        self.in_cash = True
        self.stop_loss_triggered = True

    def _trigger_take_profit(self, date, active, spy_value, realised_profit, rebalances, i, dates):
        """Trigger take-profit and lock in gains."""
        profit = active - spy_value
        print(f"💰 TAKE-PROFIT on {date.strftime('%Y-%m-%d')}")
        print(f"   Active: £{active:.2f}, SPY: £{spy_value:.2f}")
        print(f"   Profit locked: £{profit:.2f}")
        
        realised_profit += profit
        active = spy_value
        
        # Reset peak to SPY level
        self.current_peak = active
        self.stop_loss_level = self.current_peak * (1 - self.stop_loss_pct)
        
        # Reset catch-up tracking
        self.stop_loss_value = None
        self.stop_loss_spy = None
        self.stop_loss_date = None
        self.no_rules_value = None
        
        # Re-optimise
        window_data = self.price_data.iloc[i - self.lookback_days:i]
        window_returns = window_data.pct_change().dropna()
        if len(window_returns) >= self.lookback_days * 0.5:
            target_weights = self._optimise_weights(window_returns)
            cash_allocation = self._calc_cash_allocation(window_returns)
            adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
            self.weights, active, new_trades = self._rebalance(date, adj_weights, active)
            self.cash_allocation = cash_allocation
        
        self.in_cash = False
        self.stop_loss_triggered = False
        rebalances.append({'date': date, 'action': 'TAKE_PROFIT'})
        
        return realised_profit, active, rebalances

    def _check_reinvest_condition(self, active, no_rules_value, spy_value, stop_loss_spy, days_since_stop):
        """Check if we should reinvest from cash."""
        # Catch-up: reinvest when portfolio catches up to no-rules
        if active >= no_rules_value * 0.98:
            print(f"🔄 CATCH-UP REINVEST (active caught up)")
            return True
        
        # Within 5% of no-rules
        if active >= no_rules_value * 0.95:
            print(f"🔄 CATCH-UP REINVEST (within 5%)")
            return True
        
        # Time-based: wait 30 days, then check market recovery
        if days_since_stop >= 30:
            market_recovery = (spy_value / stop_loss_spy) - 1
            if market_recovery >= 0.10:
                print(f"🔄 MARKET RECOVERY REINVEST ({market_recovery*100:.1f}%)")
                return True
        
        return False

    def _reinvest_from_cash(self, date, i, active, rebalances):
        """Reinvest from cash back into the market."""
        print(f"🔄 REINVEST on {date.strftime('%Y-%m-%d')}")
        print(f"   Active: £{active:.2f}, No Rules: £{self.no_rules_value:.2f}")
        
        window_data = self.price_data.iloc[i - self.lookback_days:i]
        window_returns = window_data.pct_change().dropna()
        
        if len(window_returns) >= self.lookback_days * 0.5:
            target_weights = self._optimise_weights(window_returns)
        else:
            target_weights = pd.Series(1.0 / len(self.tickers), index=self.tickers)
        
        cash_allocation = self._calc_cash_allocation(window_returns)
        adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
        
        self.weights, active, new_trades = self._rebalance(date, adj_weights, active)
        self.cash_allocation = cash_allocation
        
        self.in_cash = False
        self.stop_loss_triggered = False
        self.current_peak = active
        self.stop_loss_level = self.current_peak * (1 - self.stop_loss_pct)
        
        # Reset catch-up tracking
        self.stop_loss_value = None
        self.stop_loss_spy = None
        self.stop_loss_date = None
        self.no_rules_value = None
        
        print(f"   New stop-loss level: £{self.stop_loss_level:.2f}")
        rebalances.append({'date': date, 'action': 'REINVEST'})
        
        return active, new_trades, rebalances

    def _summary(self, active, realised_profit, total_wealth, trades, rebalances, equity_curve, wealth_curve):
        """Generate summary."""
        print("\n✅ Backtest complete!")
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