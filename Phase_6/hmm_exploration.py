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
    Train a 3-state Gaussian HMM on SPY returns + 20-day volatility.
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
    state_info.sort(key=lambda x: (-x[1], x[2]))   # highest mean first -> bull
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


def train_and_save_hmm():
    """Train HMM on the configured training period and save to disk."""
    print("\n" + "=" * 70)
    print("TRAINING AND SAVING HMM MODEL")
    print("=" * 70)
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Saving to: {HMM_MODEL_PATH} and {HMM_STATE_MAP_PATH}")
    print("=" * 70)

    model, state_to_regime = train_hmm(TRAIN_START, TRAIN_END)

    # Save model and state mapping
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(HMM_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(HMM_STATE_MAP_PATH, 'wb') as f:
        pickle.dump(state_to_regime, f)

    print(f"\nModel saved to {HMM_MODEL_PATH}")
    print(f"State mapping saved to {HMM_STATE_MAP_PATH}")
    print("=" * 70)
    return model, state_to_regime


def plot_regimes():
    """
    Visualise the regimes detected by HMM on the full dataset (for exploration).
    This function trains a temporary model and does NOT overwrite the saved model files.
    """
    print("\nPlotting regimes on full period (training a temporary model)...")
    # Download full period
    spy = yf.download("SPY", start="1990-01-01", end=TEST_END, progress=False)
    prices = spy['Close']
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    returns = prices.pct_change().dropna()

    # Train on full period (in-sample) - temporary, not saved
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

    # ---- Plot with smaller size ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {'bull': 'green', 'bear': 'orange', 'crash': 'red'}

    ax = axes[0]
    ax.plot(regime_df['date'], regime_df['price'], color='black', linewidth=0.8)
    for regime in ['bull', 'bear', 'crash']:
        mask = regime_df['regime'] == regime
        ax.fill_between(regime_df['date'], regime_df['price'].min(), regime_df['price'].max(),
                        where=mask, color=colors[regime], alpha=0.12, label=regime.capitalize())
    ax.set_title('HMM Regimes (1990-2026)', fontsize=10)
    ax.set_ylabel('SPY Price', fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    ax = axes[1]
    ax.bar(regime_df['date'], regime_df['return'] * 100, width=1, color='gray', alpha=0.5)
    for regime in ['bull', 'bear', 'crash']:
        mask = regime_df['regime'] == regime
        ax.fill_between(regime_df['date'], -5, 5, where=mask, color=colors[regime], alpha=0.08)
    ax.set_ylabel('Daily Return (%)', fontsize=9)
    ax.set_title('Returns with Regime Shading', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylim(-8, 8)

    ax = axes[2]
    # Posterior probabilities
    posterior = model.predict_proba(X)
    posterior_df = pd.DataFrame(posterior, index=dates, columns=[f'State_{i}' for i in range(N_STATES)])
    regime_cols = {f'State_{i}': state_to_regime[i] for i in range(N_STATES)}
    posterior_df = posterior_df.rename(columns=regime_cols)
    for regime in ['bull', 'bear', 'crash']:
        ax.plot(posterior_df.index, posterior_df[regime], label=f'P({regime.capitalize()})', color=colors[regime], linewidth=0.8)
    ax.set_ylabel('Probability', fontsize=9)
    ax.set_title('Posterior Probabilities', fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'hmm_exploration_plot.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved plot to {FIGURES_DIR}/hmm_exploration_plot.png")


if __name__ == "__main__":
    # Train and save the model on 1990-2015
    train_and_save_hmm()
    # Generate the plot (uses a temporary model on full period)
    plot_regimes()