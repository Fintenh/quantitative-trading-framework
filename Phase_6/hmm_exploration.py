"""
hmm_exploration.py - Train HMM, map states, and provide probability function.
Also plots the regime detection for visualisation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn import hmm
import pickle
import warnings
warnings.filterwarnings('ignore')

from hmm_config import *


def train_hmm(start_date=TRAIN_START, end_date=TRAIN_END):
    """
    Train a 3‑state Gaussian HMM on SPY returns + 20‑day volatility.
    Returns: (model, state_to_regime) where state_to_regime maps state index to regime label.
    """
    print(f"Training HMM on SPY data from {start_date} to {end_date}...")
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    if len(spy) == 0:
        raise ValueError("No SPY data downloaded.")

    prices = spy['Close']
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    returns = prices.pct_change().dropna()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]

    vol = returns.rolling(20).std()
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0]

    feature_data = pd.DataFrame({
        'return': returns,
        'volatility': vol
    }).dropna()

    X = feature_data.values
    print(f"Training on {len(feature_data)} daily observations.")

    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type=COVARIANCE_TYPE,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        tol=1e-4,
        verbose=False
    )
    model.fit(X)
    states = model.predict(X).ravel()

    # Map states to regimes by sorting by mean return
    state_means = []
    state_stds = []
    for i in range(N_STATES):
        mask = (states == i)
        if np.sum(mask) > 0:
            state_returns = feature_data.loc[mask, 'return']
            state_means.append(state_returns.mean())
            state_stds.append(state_returns.std())
        else:
            state_means.append(-np.inf)
            state_stds.append(np.inf)

    state_info = [(i, state_means[i], state_stds[i]) for i in range(N_STATES)]
    state_info.sort(key=lambda x: (-x[1], x[2]))   # highest mean first → bull
    regime_order = ['bull', 'bear', 'crash']
    state_to_regime = {}
    for idx, (state, mean, std) in enumerate(state_info):
        if idx < len(regime_order):
            state_to_regime[state] = regime_order[idx]

    print("\nState mapping:")
    for state, regime in state_to_regime.items():
        for s, m, std in state_info:
            if s == state:
                print(f"  State {state} -> {regime}: mean={m*100:.4f}%, std={std*100:.2f}%")

    return model, state_to_regime


def get_blended_probabilities(date, model, state_to_regime, alpha, spy_data):
    """
    Compute blended probabilities for a given date.
    Returns: dict {'bull': p_bull, 'bear': p_bear, 'crash': p_crash}
    """
    # Ensure spy_data is a Series
    if isinstance(spy_data, pd.DataFrame):
        spy_data = spy_data.squeeze()

    # Get SPY data up to date
    spy_prices = spy_data[spy_data.index <= pd.Timestamp(date)]
    if len(spy_prices) < 20:
        return {'bull': 1.0, 'bear': 0.0, 'crash': 0.0}

    spy_returns = spy_prices.pct_change().dropna()
    if len(spy_returns) < 2:
        return {'bull': 1.0, 'bear': 0.0, 'crash': 0.0}

    # ---- FIX: extract scalar from Series ----
    ret_val = spy_returns.iloc[-1]
    if isinstance(ret_val, pd.Series):
        ret_val = ret_val.iloc[0]
    ret = float(ret_val)

    vol_series = spy_returns.rolling(20).std()
    vol_val = vol_series.iloc[-1]
    if isinstance(vol_val, pd.Series):
        vol_val = vol_val.iloc[0]
    vol = float(vol_val) if not np.isnan(vol_val) else 0.01
    vol = max(vol, 0.01)   # avoid zero

    X_new = np.array([[ret, vol]])
    posterior = model.predict_proba(X_new)[-1]

    current_probs = {}
    for i, regime in state_to_regime.items():
        current_probs[regime] = posterior[i]

    # Forecast next day probabilities using transition matrix
    transmat = model.transmat_
    state_list = list(state_to_regime.keys())
    n = len(state_list)
    regime_trans = np.zeros((n, n))
    for i, from_state in enumerate(state_list):
        for j, to_state in enumerate(state_list):
            regime_trans[i, j] = transmat[from_state, to_state]

    prob_vector = np.array([current_probs.get(state_to_regime[s], 0) for s in state_list])
    forecast_vector = prob_vector @ regime_trans
    forecast_probs = {}
    for i, state in enumerate(state_list):
        forecast_probs[state_to_regime[state]] = forecast_vector[i]

    # Blend
    blended = {}
    for regime in ['bull', 'bear', 'crash']:
        blended[regime] = alpha * current_probs[regime] + (1 - alpha) * forecast_probs[regime]

    return blended


def plot_regimes():
    """
    Visualise the regimes detected by HMM on the full dataset (for exploration).
    """
    # Download full period
    spy = yf.download("SPY", start="1990-01-01", end=TEST_END, progress=False)
    prices = spy['Close']
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    returns = prices.pct_change().dropna()

    # Train on full period (in‑sample)
    model, state_to_regime = train_hmm(start_date="1990-01-01", end_date=TEST_END)

    # Predict states for the full period
    vol = returns.rolling(20).std()
    feature_data = pd.DataFrame({'return': returns, 'volatility': vol}).dropna()
    X = feature_data.values
    states = model.predict(X).ravel()
    regime_labels = [state_to_regime[s] for s in states]

    dates = feature_data.index
    regime_df = pd.DataFrame({
        'date': dates,
        'price': prices.loc[dates],
        'return': returns.loc[dates],
        'regime': regime_labels
    })

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    colors = {'bull': 'green', 'bear': 'orange', 'crash': 'red'}

    ax = axes[0]
    ax.plot(regime_df['date'], regime_df['price'], color='black', linewidth=1)
    for regime in ['bull', 'bear', 'crash']:
        mask = regime_df['regime'] == regime
        ax.fill_between(regime_df['date'], regime_df['price'].min(), regime_df['price'].max(),
                        where=mask, color=colors[regime], alpha=0.15, label=regime.capitalize())
    ax.set_title('HMM Regimes (1990–2026)')
    ax.set_ylabel('SPY Price')
    ax.legend()

    ax = axes[1]
    ax.bar(regime_df['date'], regime_df['return'] * 100, width=1, color='gray', alpha=0.5)
    for regime in ['bull', 'bear', 'crash']:
        mask = regime_df['regime'] == regime
        ax.fill_between(regime_df['date'], -5, 5, where=mask, color=colors[regime], alpha=0.1)
    ax.set_ylabel('Daily Return (%)')
    ax.set_title('Returns with Regime Shading')

    ax = axes[2]
    # Posterior probabilities
    posterior = model.predict_proba(X)
    posterior_df = pd.DataFrame(posterior, index=dates, columns=[f'State_{i}' for i in range(N_STATES)])
    regime_cols = {f'State_{i}': state_to_regime[i] for i in range(N_STATES)}
    posterior_df = posterior_df.rename(columns=regime_cols)
    for regime in ['bull', 'bear', 'crash']:
        ax.plot(posterior_df.index, posterior_df[regime], label=f'P({regime.capitalize()})', color=colors[regime], linewidth=1)
    ax.set_ylabel('Probability')
    ax.set_title('Posterior Probabilities')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hmm_exploration_plot.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {FIGURES_DIR}/hmm_exploration_plot.png")


if __name__ == "__main__":
    plot_regimes()