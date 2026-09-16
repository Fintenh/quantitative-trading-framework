"""
final_ml_research.py
--------------------
ML-based portfolio selection for research purposes.
Uses ML predictions as expected returns with your existing portfolio optimiser.

Key features:
- Random Forest with 10 enhanced factors
- Correctly scales 120-day predictions to annualised decimal returns
- Uses your Phase 5 portfolio optimiser
- Allows 0% weights (like your live strategy)
- Variable number of assets (not forced to 15)

This is a RESEARCH tool. Do not use for live trading.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import linregress
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(parent_dir, "Phase_2"))
sys.path.insert(0, os.path.join(parent_dir, "Phase_3"))

from final_universe import FINAL_POOL, SECTOR_MAPPING
from portfolio_optimiser import optimise_portfolios

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logs_dir = os.path.join(script_dir, "logs")
models_dir = os.path.join(script_dir, "models")
figures_dir = os.path.join(script_dir, "figures")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

MODEL_PATH = os.path.join(models_dir, "rf_selector_120d.pkl")
SCALER_PATH = os.path.join(models_dir, "target_scaler_rf.pkl")
MODEL_INFO_PATH = os.path.join(models_dir, "rf_model_info.txt")
WEIGHTS_PATH = os.path.join(logs_dir, "ml_portfolio_weights.csv")

print("=" * 70)
print("FINAL ML RESEARCH - 120-DAY PORTFOLIO SELECTOR")
print("=" * 70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ---------------------------------------------------------------------------
# Parameters (aligned with Phase 5)
# ---------------------------------------------------------------------------

START_DATE = "2010-01-01"
END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

RISK_FREE = 0.045
HORIZON = 120   # Prediction horizon in days

# Matches Phase 5 LOOKBACK_DAYS = 405
TRAIN_LOOKBACK = 405

# Random Forest parameters (fast, parallel training)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'n_jobs': -1           # Use all CPU cores
}

# ---------------------------------------------------------------------------
# Data loading (with cutoff date for tracker and manual selection)
# ---------------------------------------------------------------------------

def load_data_up_to(cutoff_date, tickers=FINAL_POOL):
    """Load data up to a specific cutoff date."""
    start_date = cutoff_date - timedelta(days=TRAIN_LOOKBACK * 2)
    end_date = cutoff_date
    print(f"\nLoading data for {len(tickers)} assets...")
    print(f"   From: {start_date.date()} to {end_date.date()}")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    valid_tickers = []
    for ticker in tickers:
        if ticker in data.columns:
            available_pct = data[ticker].notna().sum() / len(data)
            if available_pct >= 0.8:
                valid_tickers.append(ticker)
    data = data[valid_tickers]
    returns = data.pct_change().dropna()
    print(f"   {len(valid_tickers)} assets with sufficient data")
    print(f"   Returns: {returns.index[0].date()} to {returns.index[-1].date()}")
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy_returns = spy.pct_change().dropna()
    return data, returns, valid_tickers, spy_returns

# ---------------------------------------------------------------------------
# Original load_data (for full training, kept for compatibility)
# ---------------------------------------------------------------------------

def load_data(tickers, start_date, end_date):
    print(f"\nLoading data for {len(tickers)} assets...")
    print(f"   From: {start_date} to {end_date}")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    valid_tickers = []
    for ticker in tickers:
        if ticker in data.columns:
            available_pct = data[ticker].notna().sum() / len(data)
            if available_pct >= 0.8:
                valid_tickers.append(ticker)
    data = data[valid_tickers]
    returns = data.pct_change().dropna()
    print(f"   {len(valid_tickers)} assets with sufficient data")
    print(f"   Returns: {returns.index[0].date()} to {returns.index[-1].date()}")
    return data, returns, valid_tickers

def get_spy_returns(start_date, end_date):
    print("   Loading SPY for benchmark...")
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy_returns = spy.pct_change().dropna()
    return spy_returns

def get_sector(ticker):
    return SECTOR_MAPPING.get(ticker, "Other")

# ---------------------------------------------------------------------------
# Feature computation (10 enhanced factors)
# ---------------------------------------------------------------------------

def compute_factors(returns, spy_returns):
    """
    Computes 10 features for each asset:
    1. Momentum (weighted 21/63/126/252 day)
    2. Growth (acceleration)
    3. Quality (Sharpe + consistency)
    4. Stability (low volatility)
    5. Diversification (sector penalty)
    6. Volatility (recent 63-day)
    7. Skewness (3-month)
    8. Relative Strength vs SPY (63-day)
    9. Max Drawdown (63-day)
    10. Momentum Slope (linear regression of log prices)
    """
    if len(returns) < 63:
        return pd.DataFrame(index=returns.columns)

    # 1. Momentum
    periods = [21, 63, 126, 252]
    period_weights = [0.4, 0.3, 0.2, 0.1]
    momentum_scores = []
    for period, weight in zip(periods, period_weights):
        if len(returns) >= period:
            period_returns = returns.iloc[-period:].mean() * 252
            period_vol = returns.iloc[-period:].std() * np.sqrt(252)
            denom = period_vol + 0.01
            momentum_scores.append(weight * period_returns / denom)
        else:
            momentum_scores.append(0)
    momentum = pd.Series(sum(momentum_scores), index=returns.columns)

    # 2. Growth
    if len(returns) >= 252:
        recent = returns.iloc[-126:].mean() * 252
        long_term = returns.iloc[-252:].mean() * 252
        growth = (recent - long_term) / (np.abs(long_term) + 0.01)
        growth = growth.clip(-1, 1)
    else:
        growth = pd.Series(0, index=returns.columns)

    # 3. Quality
    exp_ret = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = (exp_ret - RISK_FREE) / (vol + 0.01)
    sharpe = sharpe.clip(-5, 5)
    if len(returns) >= 12:
        monthly_returns = returns.resample('ME').mean()
        positive_months = (monthly_returns > 0).sum()
        pos_ratio = positive_months / len(monthly_returns)
    else:
        pos_ratio = 0.5
    quality = 0.6 * sharpe.clip(-1, 1) + 0.4 * pos_ratio

    # 4. Stability
    rolling_vol = returns.rolling(63).std() * np.sqrt(252)
    avg_vol = rolling_vol.mean()
    stability = 1 / (1 + avg_vol)
    stability = (stability - stability.min()) / (stability.max() - stability.min() + 0.01)

    # 5. Diversification
    sector_counts = {}
    for ticker in returns.columns:
        sector = get_sector(ticker)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    diversification = pd.Series(index=returns.columns)
    for ticker in returns.columns:
        sector = get_sector(ticker)
        count = sector_counts.get(sector, 1)
        diversification[ticker] = 1 / count
    diversification = diversification / diversification.max()

    # 6. Volatility
    recent_vol = returns.tail(63).std() * np.sqrt(252)
    recent_vol = recent_vol.clip(0, 1)

    # 7. Skewness
    if len(returns) >= 63:
        skew = returns.tail(63).skew()
        skew = skew.clip(-2, 2)
    else:
        skew = pd.Series(0, index=returns.columns)

    # 8. Relative strength vs SPY
    common_dates = returns.index.intersection(spy_returns.index)
    if len(common_dates) >= 63:
        spy_recent = spy_returns.loc[common_dates].tail(63)
        asset_63d_ret = (1 + returns.loc[common_dates].tail(63)).prod() - 1
        spy_63d_ret = (1 + spy_recent).prod() - 1
        rel_strength = asset_63d_ret - spy_63d_ret
    else:
        rel_strength = pd.Series(0, index=returns.columns)

    # 9. Max drawdown
    if len(returns) >= 63:
        cumprod = (1 + returns.tail(63)).cumprod()
        running_max = cumprod.cummax()
        drawdown = (cumprod / running_max) - 1
        max_dd = -drawdown.min()
    else:
        max_dd = pd.Series(0, index=returns.columns)

    # 10. Momentum slope
    if len(returns) >= 63:
        prices = (1 + returns.tail(63)).cumprod()
        log_prices = np.log(prices)
        slope = pd.Series(index=returns.columns)
        for ticker in returns.columns:
            x = np.arange(len(log_prices))
            y = log_prices[ticker].values
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                slope[ticker] = 0
            else:
                slope[ticker] = linregress(x, y)[0]
    else:
        slope = pd.Series(0, index=returns.columns)

    def normalise(x):
        if x.max() == x.min():
            return pd.Series(0.5, index=x.index)
        return (x - x.min()) / (x.max() - x.min() + 0.01)

    return pd.DataFrame({
        'momentum': normalise(momentum),
        'growth': normalise(growth),
        'quality': normalise(quality),
        'stability': normalise(stability),
        'diversification': normalise(diversification),
        'volatility': normalise(recent_vol),
        'skewness': normalise(skew),
        'rel_strength': normalise(rel_strength),
        'max_drawdown': normalise(max_dd),
        'momentum_slope': normalise(slope)
    })

# ---------------------------------------------------------------------------
# Train model (original full training)
# ---------------------------------------------------------------------------

def train_model(returns, spy_returns, horizon=HORIZON, lookback=TRAIN_LOOKBACK):
    print(f"\nTraining Random Forest model (lookback={lookback} days)...")

    tickers = returns.columns
    n = len(returns)

    X_rows, y_rows = [], []
    for asset in tickers:
        asset_returns = returns[asset]
        for i in range(lookback, n - horizon):
            # Get window of returns and compute features
            window = returns.iloc[i-lookback:i]
            spy_window = spy_returns.loc[window.index] if len(spy_returns) > 0 else None
            if spy_window is None or len(spy_window) < 63:
                continue
            factors = compute_factors(window, spy_window)
            if factors.empty or asset not in factors.index:
                continue
            features = factors.loc[asset].values

            # Compute the cumulative log return over the next 'horizon' days.
            # Compound the daily returns first, then take the log, rather than
            # summing log returns directly - this keeps the target consistent
            # with how the prediction is later converted back to a return.
            forward_returns = asset_returns.iloc[i+1 : i+horizon+1]
            if len(forward_returns) < horizon:
                continue
            cum_return = (1 + forward_returns).prod() - 1
            if cum_return <= -1:  # log undefined if cum_return <= -1
                continue
            forward_log_return = np.log(1 + cum_return)
            # Guard against NaN/Inf
            if not np.isfinite(forward_log_return):
                continue
            # Scale to percentage for the model (optional but keeps scale)
            forward_log_return_pct = forward_log_return * 100
            # Cap extreme values to avoid outliers
            forward_log_return_pct = np.clip(forward_log_return_pct, -200, 200)

            X_rows.append(features)
            y_rows.append(forward_log_return_pct)

    if len(X_rows) < 100:
        raise ValueError(f"Not enough training samples: {len(X_rows)}")

    X = np.array(X_rows)
    y = np.array(y_rows)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    print(f"   Training samples: {len(X)}")

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y_scaled[:split], y_scaled[split:]

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    print(f"   Validation R2: {val_r2:.3f}")

    importance = model.feature_importances_
    feature_names = ['momentum', 'growth', 'quality', 'stability',
                     'diversification', 'volatility', 'skewness',
                     'rel_strength', 'max_drawdown', 'momentum_slope']
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False)
    print("\n   Feature importance:")
    for _, row in imp_df.iterrows():
        print(f"      {row['feature']}: {row['importance']:.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(MODEL_INFO_PATH, 'w') as f:
        f.write(f"Model: Random Forest (120-day selector)\n")
        f.write(f"Trained on: {returns.index[0].date()} to {returns.index[-1].date()}\n")
        f.write(f"Lookback: {lookback} days\n")
        f.write(f"Horizon: {horizon} days\n")
        f.write(f"Training samples: {len(X)}\n")
        f.write(f"Validation R2: {val_r2:.3f}\n")
        f.write("\nFeature importance:\n")
        for _, row in imp_df.iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.3f}\n")

    print(f"\nModel saved to {MODEL_PATH}")
    return model, scaler, imp_df

# ---------------------------------------------------------------------------
# Predict returns (full output for manual run)
# ---------------------------------------------------------------------------

def predict_future_returns(model, scaler, returns, spy_returns):
    print("\n" + "=" * 70)
    print("PREDICTING 120-DAY RETURNS")
    print("=" * 70)

    if len(returns) < TRAIN_LOOKBACK:
        window = returns
    else:
        window = returns.iloc[-TRAIN_LOOKBACK:]

    print(f"   Using {len(window)} days of data (from {window.index[0].date()})")

    spy_window = spy_returns.loc[window.index] if len(spy_returns) > 0 else None
    if spy_window is None or len(spy_window) < 63:
        raise ValueError("Not enough SPY data for prediction")

    factors = compute_factors(window, spy_window)
    if factors.empty:
        raise ValueError("Could not compute factors for prediction")

    X = factors.values
    tickers = factors.index
    pred_scaled = model.predict(X)
    pred_log_returns_pct = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    results = pd.DataFrame({
        'ticker': tickers,
        'sector': [get_sector(t) for t in tickers],
        'predicted_log_return_pct': pred_log_returns_pct
    })
    results = results.sort_values('predicted_log_return_pct', ascending=False)
    results['rank'] = range(1, len(results) + 1)
    return results

# ---------------------------------------------------------------------------
# Select portfolio (full output for manual run)
# ---------------------------------------------------------------------------

def select_optimal_portfolio(predictions, returns):
    """
    Use ML predictions as expected returns in your existing portfolio optimiser.
    This is exactly the same logic as your live trading engine.

    The only difference from Phase 5 is that expected returns come from ML
    instead of historical averages. Everything else (covariance, constraints,
    optimisation method) is identical.

    Note: ML predictions are 120-day log-return percentages, so they must be
    converted to annualised decimal returns before being fed to the optimiser.
    """
    print("\n" + "=" * 70)
    print("OPTIMAL PORTFOLIO (ML Predictions + Phase 5 Optimiser)")
    print("=" * 70)

    tickers = returns.columns
    asset_returns = returns.loc[returns.index[-TRAIN_LOOKBACK:]]

    # Get ML-predicted returns for all assets (aligned to tickers)
    # predictions['predicted_log_return_pct'] is in % over 120 days
    ml_returns = predictions.set_index('ticker')['predicted_log_return_pct'] / 100   # decimal log return over 120d
    ml_returns = ml_returns.reindex(tickers).fillna(0)

    # Calculate covariance matrix (same as your live engine)
    cov = asset_returns.cov() * 252   # annualised variance

    # Convert the 120-day log-return decimal to an annualised decimal log-return
    annual_factor = 252 / HORIZON   # 252/120 = 2.1
    exp_ret_ml = pd.Series(
        ml_returns.values * annual_factor,
        index=tickers
    )

    print(f"   Assets in universe: {len(tickers)}")
    print(f"   Using ML predictions as expected returns")
    print(f"   Covariance matrix from last {TRAIN_LOOKBACK} days")
    print(f"   Risk-free rate: {RISK_FREE*100:.1f}%")
    print(f"   Annualisation factor: {annual_factor:.2f}")

    # Show a sample conversion for verification
    sample_ticker = predictions.iloc[0]['ticker']
    sample_pred = predictions.iloc[0]['predicted_log_return_pct']
    sample_annual = sample_pred / 100 * annual_factor
    print(f"   Sample: {sample_ticker} 120-day return = {sample_pred:.2f}% -> annualised = {sample_annual*100:.2f}%")

    try:
        # Run your existing portfolio optimiser
        opt = optimise_portfolios(exp_ret_ml, cov, risk_free=RISK_FREE)

        # Extract weights
        weights = {ticker: opt['msr_weights'][i] for i, ticker in enumerate(tickers)}

        # Calculate portfolio stats
        weight_array = np.array([weights[t] for t in tickers])
        exp_ret_array = np.array([exp_ret_ml[t] for t in tickers])
        portfolio_return = np.sum(exp_ret_array * weight_array)
        portfolio_vol = np.sqrt(weight_array.T @ cov.values @ weight_array)
        sharpe = (portfolio_return - RISK_FREE) / portfolio_vol if portfolio_vol > 0 else 0

        # Filter assets with positive weight
        selected = [t for t, w in weights.items() if w > 0.001]
        selected = sorted(selected, key=lambda t: weights[t], reverse=True)

        print(f"\n   Assets with positive weight: {len(selected)} / {len(tickers)}")
        print(f"   Portfolio Return: {portfolio_return*100:.2f}% (annualised)")
        print(f"   Portfolio Volatility: {portfolio_vol*100:.2f}%")
        print(f"   Sharpe Ratio: {sharpe:.3f}")

        print("\n   Top weights:")
        for t in selected[:10]:
            print(f"      {t}: {weights[t]*100:.1f}%")

        return selected, weights, sharpe

    except Exception as e:
        print(f"\n   Optimisation failed: {e}")
        print("   Using equal weights as fallback (not recommended)")

        equal_weight = 1.0 / len(tickers)
        weights = {t: equal_weight for t in tickers}
        selected = tickers[:15]
        return selected, weights, 0.0

# ---------------------------------------------------------------------------
# Save weights
# ---------------------------------------------------------------------------

def save_weights(selected_tickers, weight_dict, predictions):
    # Filter to only assets with positive weight
    positive_weights = {t: w for t, w in weight_dict.items() if w > 0.001}

    if not positive_weights:
        print("   WARNING: No positive weights found. Saving empty file.")
        pd.DataFrame({'ticker': [], 'weight': []}).to_csv(WEIGHTS_PATH, index=False)
        return

    df = pd.DataFrame({
        'ticker': list(positive_weights.keys()),
        'weight': list(positive_weights.values()),
        'weight_pct': [w*100 for w in positive_weights.values()],
        'predicted_return_pct': [
            predictions[predictions['ticker'] == t]['predicted_log_return_pct'].values[0]
            if t in predictions['ticker'].values else 0
            for t in positive_weights.keys()
        ]
    })
    df = df.sort_values('weight', ascending=False)
    df['date'] = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(WEIGHTS_PATH, index=False)
    print(f"\nPortfolio weights saved to: {WEIGHTS_PATH}")
    return df

# ---------------------------------------------------------------------------
# Generate ML weights for a specific date (used by the performance tracker)
# ---------------------------------------------------------------------------

def generate_ml_weights_for_date(cutoff_date, tickers=FINAL_POOL):
    """
    Generate ML portfolio weights using data up to cutoff_date.
    cutoff_date can be a date or datetime object.
    Returns: dict {ticker: weight} (weights sum to 1)
    """
    # Convert to datetime if needed
    if isinstance(cutoff_date, datetime):
        cutoff_dt = cutoff_date
    else:
        cutoff_dt = datetime.combine(cutoff_date, datetime.min.time())

    print(f"\nGenerating ML weights for cutoff date: {cutoff_dt.strftime('%Y-%m-%d')}")
    data, returns, valid_tickers, spy_returns = load_data_up_to(cutoff_dt, tickers)
    if len(valid_tickers) < 15:
        raise ValueError("Not enough valid tickers.")

    # Load or train model
    model = None
    scaler = None
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("\nLoading pre-trained model...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        model, scaler, _ = train_model(returns, spy_returns)

    predictions = predict_returns(model, scaler, returns, spy_returns)
    selected, weights, sharpe = select_portfolio(predictions, returns)
    # Filter to positive weights
    result = {t: weights[t] for t in selected if weights[t] > 0.001}
    # Normalise to sum to 1 (they should already, but just in case)
    total = sum(result.values())
    if total > 0:
        result = {t: w / total for t, w in result.items()}
    return result

# ---------------------------------------------------------------------------
# Predict returns (simplified for tracker, without print overhead)
# ---------------------------------------------------------------------------

def predict_returns(model, scaler, returns, spy_returns, lookback=TRAIN_LOOKBACK):
    if len(returns) < lookback:
        window = returns
    else:
        window = returns.iloc[-lookback:]

    spy_window = spy_returns.loc[window.index] if len(spy_returns) > 0 else None
    if spy_window is None or len(spy_window) < 63:
        raise ValueError("Not enough SPY data for prediction")

    factors = compute_factors(window, spy_window)
    if factors.empty:
        raise ValueError("Could not compute factors for prediction")

    X = factors.values
    tickers = factors.index
    pred_scaled = model.predict(X)
    pred_log_returns_pct = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    results = pd.DataFrame({
        'ticker': tickers,
        'sector': [get_sector(t) for t in tickers],
        'predicted_log_return_pct': pred_log_returns_pct
    })
    results = results.sort_values('predicted_log_return_pct', ascending=False)
    results['rank'] = range(1, len(results) + 1)
    return results

# ---------------------------------------------------------------------------
# Select portfolio (simplified for tracker, without print overhead)
# ---------------------------------------------------------------------------

def select_portfolio(predictions, returns, risk_free=RISK_FREE, lookback=TRAIN_LOOKBACK, horizon=HORIZON):
    tickers = returns.columns
    asset_returns = returns.loc[returns.index[-lookback:]]

    ml_returns = predictions.set_index('ticker')['predicted_log_return_pct'] / 100
    ml_returns = ml_returns.reindex(tickers).fillna(0)

    cov = asset_returns.cov() * 252
    annual_factor = 252 / horizon
    exp_ret_ml = pd.Series(ml_returns.values * annual_factor, index=tickers)

    try:
        opt = optimise_portfolios(exp_ret_ml, cov, risk_free=risk_free)
        weights = {ticker: opt['msr_weights'][i] for i, ticker in enumerate(tickers)}
        weight_array = np.array([weights[t] for t in tickers])
        exp_ret_array = np.array([exp_ret_ml[t] for t in tickers])
        portfolio_return = np.sum(exp_ret_array * weight_array)
        portfolio_vol = np.sqrt(weight_array.T @ cov.values @ weight_array)
        sharpe = (portfolio_return - risk_free) / portfolio_vol if portfolio_vol > 0 else 0

        selected = [t for t, w in weights.items() if w > 0.001]
        selected = sorted(selected, key=lambda t: weights[t], reverse=True)

        return selected, weights, sharpe

    except Exception as e:
        print(f"   Optimisation failed: {e}")
        equal_weight = 1.0 / len(tickers)
        weights = {t: equal_weight for t in tickers}
        selected = tickers[:15]
        return selected, weights, 0.0

# ---------------------------------------------------------------------------
# Main (manual runner - full detailed output, uses the same universe as tracker)
# ---------------------------------------------------------------------------

def main():
    print(f"\nGenerating portfolio weights using ML predictions + Phase 5 optimiser.")
    print(f"   Training data: {START_DATE} to {END_DATE}")
    print(f"   Lookback: {TRAIN_LOOKBACK} days (matched to Phase 5)")
    print("=" * 70)

    # Use yesterday as cutoff to avoid look-ahead bias
    cutoff_date = datetime.now() - timedelta(days=1)
    print(f"Using cutoff date: {cutoff_date.strftime('%Y-%m-%d')} (yesterday)")

    # Load data using the tracker's method (short history) to get all 55 assets
    data, returns, valid_tickers, spy_returns = load_data_up_to(cutoff_date, FINAL_POOL)
    if len(valid_tickers) < 15:
        print(f"Only {len(valid_tickers)} assets available. Need at least 15.")
        return

    # Load the pre-trained model (must exist)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Pre-trained model not found. Please run training first.")
        return

    print("\nLoading pre-trained model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH, 'r') as f:
            print(f.read())

    predictions = predict_future_returns(model, scaler, returns, spy_returns)

    print("\n" + "-" * 70)
    print("TOP 20 PREDICTED 120-DAY LOG RETURNS")
    print("-" * 70)
    print(f"{'Rank':<6} {'Ticker':<10} {'Sector':<20} {'Predicted Return (%)':<25}")
    print("-" * 70)
    top20 = predictions.head(20)
    for _, row in top20.iterrows():
        print(f"{row['rank']:<6} {row['ticker']:<10} {row['sector'][:20]:<20} {row['predicted_log_return_pct']:>10.2f}%")

    # Select portfolio using your existing optimiser
    selected, weights, sharpe = select_optimal_portfolio(predictions, returns)

    print("\n" + "=" * 70)
    print("FINAL PORTFOLIO WEIGHTS (From ML Predictions + Phase 5 Optimiser)")
    print("=" * 70)
    print(f"Total assets with positive weight: {len(selected)} / {len(weights)}")
    if sharpe > 0:
        print(f"Sharpe Ratio: {sharpe:.3f}")
    else:
        print("Sharpe Ratio: N/A")

    if len(selected) > 0 and sharpe > 0:
        print("-" * 70)
        print(f"{'Ticker':<10} {'Weight':<12} {'Predicted Return'}")
        print("-" * 70)

        for t in selected[:15]:
            w = weights[t]*100
            ret = predictions[predictions['ticker'] == t]['predicted_log_return_pct'].values[0] \
                  if t in predictions['ticker'].values else 0
            print(f"{t:<10} {w:>6.1f}%        {ret:>10.2f}%")
    else:
        print("\n   No positive weights found. Check the optimisation.")

    # Save weights
    save_weights(selected, weights, predictions)

    print("\n" + "=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)
    print("\n** DISCLAIMER: This is RESEARCH output. Do NOT use for live trading. **")
    print("   The ML portfolio weights are for comparison with the proven")
    print("   Phase 5 strategy. Use final_trading_engine.py for live trading.")
    print("\n   Key difference: Expected returns come from ML predictions,")
    print("   not historical averages. Everything else (covariance, constraints,")
    print("   optimisation) is identical to your Phase 5 strategy.")

if __name__ == "__main__":
    main()