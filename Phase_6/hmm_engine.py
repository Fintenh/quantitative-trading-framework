"""
hmm_engine.py - Unified HMM engine with Kelly cash allocation.
Uses heuristic regime probabilities (based on SPY returns and volatility).
GARCH output is always silenced.
"""

import sys
import numpy as np
import pandas as pd
import pickle
from io import StringIO
from ethical_backtest import BacktestEngineWithKelly

# Import config for paths (we may not use HMM but still load for compatibility)
from hmm_config import HMM_MODEL_PATH, HMM_STATE_MAP_PATH


# ---------------------------------------------------------------------------
# Heuristic regime probabilities (softmax on SPY returns and volatility)
# ---------------------------------------------------------------------------

def heuristic_regime_probabilities(date, spy_data):
    """
    Compute smooth regime probabilities using softmax on SPY returns and volatility.
    This is the original rule that worked well in testing.
    """
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()
    if not isinstance(spy_data, pd.Series):
        raise TypeError("spy_data must be a pandas Series")

    date_ts = pd.Timestamp(date)
    spy_prices = spy_data[spy_data.index <= date_ts]
    if len(spy_prices) < 20:
        return {'bull': 0.7, 'bear': 0.2, 'crash': 0.1}

    spy_returns = spy_prices.pct_change().dropna()
    if len(spy_returns) < 2:
        return {'bull': 0.7, 'bear': 0.2, 'crash': 0.1}

    ret_val = spy_returns.iloc[-1]
    if isinstance(ret_val, pd.Series):
        ret_val = ret_val.iloc[0]
    ret = float(ret_val)

    vol_series = spy_returns.rolling(20).std()
    vol_val = vol_series.iloc[-1]
    if isinstance(vol_val, pd.Series):
        vol_val = vol_val.iloc[0]
    vol = float(vol_val) if not np.isnan(vol_val) else 0.015
    vol = max(vol, 0.005)

    ret_norm = np.clip(ret / 0.02, -1, 1)
    vol_norm = np.clip((vol - 0.01) / 0.03, 0, 1)

    bull_score = 1.0 + ret_norm - 0.3 * vol_norm
    bear_score = 1.0 - ret_norm - 0.3 * vol_norm
    crash_score = 1.0 - ret_norm + 0.8 * vol_norm

    scores = np.array([bull_score, bear_score, crash_score])
    exp_scores = np.exp(scores * 2.0)
    probs = exp_scores / exp_scores.sum()

    return {
        'bull': probs[0],
        'bear': probs[1],
        'crash': probs[2]
    }


# ---------------------------------------------------------------------------
# Unified HMM engine (uses heuristic probabilities)
# ---------------------------------------------------------------------------

class HMMEngineWithKelly(BacktestEngineWithKelly):
    """
    Regime-weighted engine with Kelly cash allocation.
    Blends rebalancing parameters using heuristic regime probabilities from SPY.
    GARCH output is always silenced.
    Set verbose=True to print rebalance details.
    Emergency rebalance on Kelly sign change is included.
    """
    def __init__(self, spy_data, regime_params, verbose=False, debug_probs=False, **kwargs):
        super().__init__(**kwargs)
        if isinstance(spy_data, pd.DataFrame):
            spy_data = spy_data.squeeze()
        self.spy_data = spy_data
        self.regime_params = regime_params
        self.verbose = verbose
        self.debug_probs = debug_probs
        self._last_blended = {}

    def _calc_cash_allocation(self, returns):
        """Override to always silence GARCH output."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = super()._calc_cash_allocation(returns)
        finally:
            sys.stdout = old_stdout
        return result

    def _get_blended_params(self, date):
        """
        Compute blended parameters using heuristic regime probabilities.
        """
        # Get heuristic regime probabilities
        probs = heuristic_regime_probabilities(date, self.spy_data)
        self._last_blended = probs

        # Debug: print probabilities at certain dates if debug_probs is True
        if self.debug_probs:
            # Print for the first of each quarter
            if date.day == 1 and date.month in [1, 4, 7, 10]:
                print(f"\n{date.strftime('%Y-%m-%d')} probs: bull={probs['bull']:.3f}, bear={probs['bear']:.3f}, crash={probs['crash']:.3f}")

        blended = {}
        for param in ['lookback_days', 'drift_threshold', 'rebalance_min_days', 'rebalance_max_days']:
            blended[param] = (
                probs['bull'] * self.regime_params['bull'][param] +
                probs['bear'] * self.regime_params['bear'][param] +
                probs['crash'] * self.regime_params['crash'][param]
            )
        # Cap drift to avoid extreme values
        if blended['drift_threshold'] > 0.10:
            blended['drift_threshold'] = 0.10

        # Round integer parameters
        blended['lookback_days'] = int(round(blended['lookback_days']))
        blended['rebalance_min_days'] = int(round(blended['rebalance_min_days']))
        blended['rebalance_max_days'] = int(round(blended['rebalance_max_days']))

        return blended

    def run(self):
        """
        Override run() to use blended parameters and emergency rebalance.
        All other logic (data download, Kelly cash allocation, take-profit)
        is inherited from the parent.
        """
        self._download_data()
        if self.price_data is None or len(self.price_data) == 0:
            return self._empty_results()

        prices = self.price_data
        tickers = self.tickers
        dates = prices.index
        start_idx = self.lookback_days
        if len(dates) <= start_idx:
            return self._empty_results()

        active = self.initial_capital
        realised_profit = 0.0
        total_wealth = active + realised_profit

        self.weights = pd.Series(1.0 / len(tickers), index=tickers)
        self.cash_allocation = 0.0
        last_rebalance = dates[start_idx]
        self.spy_start = self._get_spy_value(dates[start_idx])

        equity_curve = []
        wealth_curve = []
        trades = []
        rebalances = []
        self._rebalance_log = []

        for i in range(start_idx, len(dates)):
            current_date = dates[i]
            current_prices = prices.loc[current_date]
            prev_prices = prices.loc[dates[i-1]]

            # Update active value
            if i > start_idx:
                daily_cash_rate = (1 + self.cash_interest_rate) ** (1/252) - 1 if self.cash_interest_rate > 0 else 0

                # Only include tickers that have valid prices on both days -
                # this avoids treating a missing price as a 0% or undefined
                # return and keeps the equity return calculation well-defined.
                valid_tickers = []
                for t in tickers:
                    if t in current_prices.index and t in prev_prices.index:
                        if pd.notna(current_prices[t]) and pd.notna(prev_prices[t]) and prev_prices[t] > 0:
                            valid_tickers.append(t)

                if len(valid_tickers) > 0:
                    # Calculate equity return using only valid tickers
                    equity_return = 0.0
                    for t in valid_tickers:
                        weight = self.weights.get(t, 0.0)
                        if weight > 0:
                            equity_return += weight * (current_prices[t] / prev_prices[t] - 1)
                else:
                    equity_return = 0.0

                total_return = (equity_return * (1 - self.cash_allocation)) + (daily_cash_rate * self.cash_allocation)
                active *= (1 + total_return)

            total_wealth = active + realised_profit

            # Take-profit (if enabled)
            if self.take_profit_pct > 0:
                spy_value = self._get_spy_value(current_date) / self.spy_start * self.initial_capital
                target = spy_value * (1 + self.take_profit_pct)
                if active > target:
                    realised_profit, active, rebalances = self._trigger_take_profit(
                        current_date, active, spy_value, realised_profit, rebalances, i, dates
                    )
                    equity_curve.append({'date': current_date, 'value': active})
                    wealth_curve.append({'date': current_date, 'value': total_wealth})
                    continue

            # Emergency rebalance due to Kelly sign change
            force_rebalance = self.should_force_rebalance()
            if force_rebalance:
                blended = self._get_blended_params(current_date)
                lookback = blended['lookback_days']
                drift_threshold = blended['drift_threshold']
                min_days = blended['rebalance_min_days']
                max_days = blended['rebalance_max_days']

                if i > lookback:
                    returns_window = prices.iloc[i-lookback:i].pct_change().dropna()
                else:
                    returns_window = prices.iloc[:i].pct_change().dropna()

                if len(returns_window) >= 1:
                    target_weights = self._optimise_weights(returns_window)
                    cash_allocation = self._calc_cash_allocation(returns_window)  # GARCH silenced
                    adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
                    if self.verbose:
                        print(f"\nEMERGENCY REBALANCE on {current_date.strftime('%Y-%m-%d')} (Kelly sign change)")
                        print(f"   Blended params: lookback={lookback}, drift={drift_threshold:.4f}, min={min_days}, max={max_days}")
                    self.weights, active, new_trades = self._rebalance(current_date, adj_weights, active)
                    self.cash_allocation = cash_allocation
                    trades.extend(new_trades)
                    last_rebalance = current_date
                    rebalances.append({'date': current_date, 'action': 'EMERGENCY_REBALANCE'})
                    equity_curve.append({'date': current_date, 'value': active})
                    wealth_curve.append({'date': current_date, 'value': total_wealth})
                    continue

            # Normal rebalance check
            if not rebalances or rebalances[-1]['date'] != current_date:
                blended = self._get_blended_params(current_date)
                lookback = blended['lookback_days']
                drift_threshold = blended['drift_threshold']
                min_days = blended['rebalance_min_days']
                max_days = blended['rebalance_max_days']

                days_since = (current_date - last_rebalance).days

                if days_since >= min_days:
                    if i > lookback:
                        returns_window = prices.iloc[i-lookback:i].pct_change().dropna()
                    else:
                        returns_window = prices.iloc[:i].pct_change().dropna()

                    if len(returns_window) >= 1:
                        target_weights = self._optimise_weights(returns_window)
                        cash_allocation = self._calc_cash_allocation(returns_window)  # GARCH silenced
                        adj_weights = {t: w * (1 - cash_allocation) for t, w in target_weights.items()}
                        max_drift, drifting = self._calc_max_drift(adj_weights)

                        if days_since >= max_days or max_drift > drift_threshold:
                            if self.verbose:
                                reason = f"Time-based: {days_since} days" if days_since >= max_days else f"Drift-based: {drifting} drifted {max_drift*100:.2f}%"
                                print(f"\nREBALANCE on {current_date.strftime('%Y-%m-%d')}")
                                print(f"   Reason: {reason}")
                                print(f"   Blended params: lookback={lookback}, drift={drift_threshold:.4f}, min={min_days}, max={max_days}")
                            self.weights, active, new_trades = self._rebalance(current_date, adj_weights, active)
                            self.cash_allocation = cash_allocation
                            trades.extend(new_trades)
                            last_rebalance = current_date
                            rebalances.append({'date': current_date, 'action': 'REBALANCE'})
                    else:
                        pass

            equity_curve.append({'date': current_date, 'value': active})
            wealth_curve.append({'date': current_date, 'value': total_wealth})

        return self._summary(active, realised_profit, total_wealth, trades, rebalances, equity_curve, wealth_curve)