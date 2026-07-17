"""
robust_optimiser.py - Bayesian Optimisation + Polynomial Fitting + Robustness Testing.
"""

import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

from walk_forward_cv import WalkForwardCV

try:
    from config_optimised import RISK_FREE_RATE, TRANSACTION_COST_PCT
except ImportError:
    RISK_FREE_RATE = 0.02
    TRANSACTION_COST_PCT = 0.004


class RobustOptimiser:
    def __init__(
        self,
        backtest_engine,
        param_space,
        n_bayesian_calls=80,
        n_initial_points=15,
        polynomial_degree=4,
        smoothing_window=11,
        random_state=42,
        cash_interest_rate=0.0525,
        risk_free_rate=RISK_FREE_RATE,
        transaction_cost_pct=TRANSACTION_COST_PCT,
        cash_min_volatility=0.25,
        cash_max_volatility=0.50,
        cash_max_allocation=0.30
    ):
        self.backtest_engine = backtest_engine
        self.param_space = param_space
        self.n_bayesian_calls = n_bayesian_calls
        self.n_initial_points = n_initial_points
        self.polynomial_degree = polynomial_degree
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.random_state = random_state
        
        self.cash_interest_rate = cash_interest_rate
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_pct = transaction_cost_pct
        self.cash_min_volatility = cash_min_volatility
        self.cash_max_volatility = cash_max_volatility
        self.cash_max_allocation = cash_max_allocation
        
        self.results = []
        self.fold_results = []
        self.best_params = None
        self.robust_params = None
        self.tickers = None

    def run(self, data, tickers, n_splits=5):
        """Run the complete robust optimisation pipeline."""
        self.tickers = tickers
        self._print_config(n_splits)
        
        cv = WalkForwardCV(train_size_years=5, test_size_years=1, step_size_years=1)
        folds = cv.split(data)
        
        if len(folds) < n_splits:
            print(f"\n⚠️ Only {len(folds)} folds available. Using all.")
            n_splits = len(folds)
        
        all_fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds[:n_splits]):
            fold_info = cv.get_fold_info(data, fold_idx, train_idx, test_idx)
            fold_data = cv.get_fold_data(data, train_idx, test_idx)
            
            # ================================================================
            # FIX: Skip if training data is too small
            # ================================================================
            if len(fold_data['train']) < 100:
                print(f"\n⚠️ Skipping fold {fold_idx + 1}: only {len(fold_data['train'])} training days")
                continue
            
            print(f"\n{'='*70}")
            print(f"FOLD {fold_idx + 1}/{n_splits}")
            print(f"{'='*70}")
            print(f"  Train: {fold_info['train_start'].strftime('%Y-%m-%d')} → {fold_info['train_end'].strftime('%Y-%m-%d')}")
            print(f"  Test:  {fold_info['test_start'].strftime('%Y-%m-%d')} → {fold_info['test_end'].strftime('%Y-%m-%d')}")
            
            print(f"\n📊 Bayesian Optimisation on Training Data...")
            bayesian_results = self._run_bayesian(fold_data['train'])
            
            # ================================================================
            # FIX: Skip if no Bayesian results
            # ================================================================
            if bayesian_results is None or len(bayesian_results) == 0:
                print(f"  ⚠️ No Bayesian results for fold {fold_idx + 1}, skipping")
                continue
            
            print(f"\n📈 Polynomial Fitting...")
            polynomial_results = self._fit_polynomial(bayesian_results)
            
            fold_optimal = self._find_polynomial_optimum(polynomial_results)
            
            print(f"\n✅ Validating on Test Data...")
            test_performance = self._run_backtest(fold_data['test'], fold_optimal)
            
            fold_result = {
                'fold': fold_idx + 1,
                'optimal_params': fold_optimal,
                'test_sharpe': test_performance['sharpe_ratio'],
                'test_return': test_performance['total_return'],
                'test_drawdown': test_performance['max_drawdown'],
                'test_volatility': test_performance['annualised_volatility'],
                'num_trades': test_performance['num_trades'],
                'train_start': fold_info['train_start'],
                'train_end': fold_info['train_end'],
                'test_start': fold_info['test_start'],
                'test_end': fold_info['test_end'],
                'all_bayesian_results': bayesian_results,
                'polynomial_fits': polynomial_results
            }
            
            all_fold_results.append(fold_result)
            
            print(f"\n  📊 Fold {fold_idx + 1} Results:")
            print(f"     Sharpe: {test_performance['sharpe_ratio']:.3f}")
            print(f"     Return: {test_performance['total_return']*100:.2f}%")
            print(f"     Drawdown: {test_performance['max_drawdown']*100:.2f}%")
            print(f"     Trades: {test_performance['num_trades']}")
            print(f"     Parameters: {fold_optimal}")
        
        # ================================================================
        # FIX: Handle case where all folds were skipped
        # ================================================================
        if len(all_fold_results) == 0:
            print("\n❌ No valid folds! Using default parameters.")
            # Use the first parameter from the space as default
            default_params = {}
            for key, bounds in self.param_space.items():
                if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                    default_params[key] = int((bounds[0] + bounds[1]) / 2)
                else:
                    default_params[key] = round((bounds[0] + bounds[1]) / 2, 3)
            self.robust_params = default_params
            
            print("\n✅ DEFAULT PARAMETERS:")
            for key, val in default_params.items():
                if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                    print(f"  {key}: {val} days")
                elif key == 'drift_threshold':
                    print(f"  {key}: {val*100:.2f}%")
                else:
                    print(f"  {key}: {val*100:.1f}%")
            
            return {
                'robust_params': default_params,
                'fold_results': [],
                'robustness_score': 0.5,
                'final_performance': {'sharpe_ratio': 0, 'total_return': 0, 'max_drawdown': 0, 'annualised_volatility': 0, 'num_trades': 0}
            }
        
        self.fold_results = all_fold_results
        
        print(f"\n{'='*70}")
        print("FINAL STEP: Robust Parameter Selection")
        print(f"{'='*70}")
        
        robust_params = self._select_robust_parameters(all_fold_results)
        self.robust_params = robust_params
        
        print(f"\n✅ ROBUST PARAMETERS:")
        for key, val in robust_params.items():
            if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                print(f"  {key}: {val} days")
            elif key == 'drift_threshold':
                print(f"  {key}: {val*100:.2f}%")
            else:
                print(f"  {key}: {val*100:.1f}%")
        
        print(f"\n{'='*70}")
        print("FINAL ROBUSTNESS TEST")
        print(f"{'='*70}")
        robustness_score = self._test_robustness(data, robust_params)
        
        print(f"\n{'='*70}")
        print("FINAL BACKTEST WITH ROBUST PARAMETERS")
        print(f"{'='*70}")
        final_performance = self._run_backtest(data, robust_params)
        
        print(f"\n📊 Final Performance:")
        print(f"  Sharpe: {final_performance['sharpe_ratio']:.3f}")
        print(f"  Return: {final_performance['total_return']*100:.2f}%")
        print(f"  Drawdown: {final_performance['max_drawdown']*100:.2f}%")
        print(f"  Volatility: {final_performance['annualised_volatility']*100:.2f}%")
        print(f"  Trades: {final_performance['num_trades']}")
        
        return {
            'robust_params': robust_params,
            'fold_results': all_fold_results,
            'robustness_score': robustness_score,
            'final_performance': final_performance
        }

    def _print_config(self, n_splits):
        """Print configuration."""
        print("=" * 70)
        print("ROBUST OPTIMISATION PIPELINE")
        print("=" * 70)
        print(f"Bayesian calls per fold: {self.n_bayesian_calls}")
        print(f"Walk-forward splits: {n_splits}")
        print(f"Polynomial degree: {self.polynomial_degree}")
        print(f"Smoothing window: {self.smoothing_window}")
        print(f"Cash Interest Rate: {self.cash_interest_rate*100:.2f}% AER")
        print(f"Risk-Free Rate: {self.risk_free_rate*100:.1f}%")
        print(f"Transaction Cost: {self.transaction_cost_pct*100:.1f}%")
        print("=" * 70)

    def _run_bayesian(self, data):
        """Run Bayesian optimisation on training data."""
        space = []
        for name, bounds in self.param_space.items():
            if name in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                space.append(Integer(bounds[0], bounds[1], name=name))
            else:
                space.append(Real(bounds[0], bounds[1], name=name))
        
        all_params = []
        all_scores = []
        
        def objective(params):
            lookback = int(params[0])
            rebalance_min = int(params[1])
            rebalance_max = int(params[2])
            drift = float(params[3])
            take_profit = float(params[4])
            
            param_dict = {
                'lookback_days': lookback,
                'rebalance_min_days': rebalance_min,
                'rebalance_max_days': rebalance_max,
                'drift_threshold': drift,
                'take_profit_pct': take_profit
            }
            
            try:
                results = self._run_backtest(data, param_dict)
                score = self._multi_objective_score(results)
                all_params.append(param_dict)
                all_scores.append(score)
                return -score
            except Exception as e:
                # Only print errors occasionally to avoid spam
                if len(all_params) % 10 == 0:
                    print(f"  ⚠️ Error: {e}")
                return 10.0
        
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_bayesian_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=False
        )
        
        # ================================================================
        # FIX: Handle empty results
        # ================================================================
        if len(all_scores) == 0:
            print("  ⚠️ No Bayesian results generated")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(all_params)
        results_df['score'] = all_scores
        
        best_idx = np.argmax(all_scores)
        best_params = all_params[best_idx]
        best_score = all_scores[best_idx]
        
        print(f"  Best Bayesian Score: {best_score:.3f}")
        print(f"  Best Bayesian Params: {best_params}")
        
        return results_df

    def _fit_polynomial(self, results_df):
        """Fit polynomial to Bayesian results for each parameter."""
        param_names = list(self.param_space.keys())
        polynomial_results = {}
        
        # ================================================================
        # FIX: Handle empty DataFrame
        # ================================================================
        if results_df is None or len(results_df) == 0:
            print("  ⚠️ No results to fit polynomial to, using defaults")
            for param in param_names:
                bounds = self.param_space[param]
                if param in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                    default_val = int((bounds[0] + bounds[1]) / 2)
                else:
                    default_val = round((bounds[0] + bounds[1]) / 2, 3)
                polynomial_results[param] = {
                    'optimal': default_val,
                    'max_score': 0,
                    'x_vals': np.array([default_val]),
                    'y_vals': np.array([0]),
                    'x_range': None,
                    'y_pred': None
                }
                print(f"  {param}: {default_val} (default)")
            return polynomial_results
        
        for param in param_names:
            grouped = results_df.groupby(param)['score'].mean().reset_index()
            grouped = grouped.sort_values(param)
            
            x_vals = grouped[param].values
            y_vals = grouped['score'].values
            
            if len(x_vals) > self.polynomial_degree:
                try:
                    X = x_vals.reshape(-1, 1)
                    model = make_pipeline(
                        PolynomialFeatures(degree=self.polynomial_degree),
                        Ridge(alpha=0.1)
                    )
                    model.fit(X, y_vals)
                    
                    x_range = np.linspace(x_vals.min(), x_vals.max(), 1000)
                    X_range = x_range.reshape(-1, 1)
                    y_pred = model.predict(X_range)
                    
                    if len(y_pred) > self.smoothing_window:
                        y_pred = savgol_filter(y_pred, self.smoothing_window, 3)
                    
                    peak_idx = np.argmax(y_pred)
                    optimal_x = x_range[peak_idx]
                    optimal_y = y_pred[peak_idx]
                    
                    if param in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                        optimal_x = int(round(optimal_x))
                    else:
                        optimal_x = round(optimal_x, 3)
                    
                    polynomial_results[param] = {
                        'optimal': optimal_x,
                        'max_score': optimal_y,
                        'x_vals': x_vals,
                        'y_vals': y_vals,
                        'x_range': x_range,
                        'y_pred': y_pred
                    }
                    print(f"  {param}: {optimal_x} (score: {optimal_y:.3f})")
                except Exception as e:
                    max_idx = np.argmax(y_vals)
                    polynomial_results[param] = {
                        'optimal': x_vals[max_idx],
                        'max_score': y_vals[max_idx],
                        'x_vals': x_vals,
                        'y_vals': y_vals,
                        'x_range': None,
                        'y_pred': None
                    }
                    print(f"  {param}: {x_vals[max_idx]} (score: {y_vals[max_idx]:.3f}) - using max")
            else:
                max_idx = np.argmax(y_vals)
                polynomial_results[param] = {
                    'optimal': x_vals[max_idx],
                    'max_score': y_vals[max_idx],
                    'x_vals': x_vals,
                    'y_vals': y_vals,
                    'x_range': None,
                    'y_pred': None
                }
                print(f"  {param}: {x_vals[max_idx]} (score: {y_vals[max_idx]:.3f}) - using max")
        
        return polynomial_results

    def _find_polynomial_optimum(self, polynomial_results):
        """Extract optimal parameters from polynomial fits."""
        return {param: result['optimal'] for param, result in polynomial_results.items()}

    def _run_backtest(self, data, params):
        """Run a single backtest with given parameters."""
        start = data.index[0].strftime('%Y-%m-%d')
        end = data.index[-1].strftime('%Y-%m-%d')
        
        engine = self.backtest_engine(
            tickers=self.tickers,
            start_date=start,
            end_date=end,
            initial_capital=100.0,
            lookback_days=params['lookback_days'],
            rebalance_min_days=params['rebalance_min_days'],
            rebalance_max_days=params['rebalance_max_days'],
            drift_threshold=params['drift_threshold'],
            take_profit_pct=params['take_profit_pct'],
            risk_free_rate=self.risk_free_rate,
            transaction_cost_pct=self.transaction_cost_pct,
            cash_interest_rate=self.cash_interest_rate,
            cash_min_volatility=self.cash_min_volatility,
            cash_max_volatility=self.cash_max_volatility,
            cash_max_allocation=self.cash_max_allocation
        )
        
        results = engine.run()
        
        from performance_metrics import calculate_metrics
        metrics = calculate_metrics(
            equity_curve=results['wealth_curve']['value'],
            initial_capital=100.0,
            risk_free_rate=self.risk_free_rate
        )
        
        metrics['num_trades'] = len(results.get('trades', []))
        return metrics

    def _multi_objective_score(self, metrics):
        """Calculate multi-objective score for optimisation."""
        sharpe = min(max(metrics['sharpe_ratio'], -2), 3)
        drawdown = min(abs(metrics['max_drawdown']), 0.5)
        num_trades = min(metrics['num_trades'], 500)
        volatility = min(metrics['annualised_volatility'], 0.5)
        total_return = min(max(metrics['total_return'], -1), 2)
        
        sharpe_score = (sharpe + 2) / 5
        drawdown_score = 1 - (drawdown / 0.5)
        trade_score = 1 - (num_trades / 500)
        volatility_score = 1 - (volatility / 0.5)
        return_score = (total_return + 1) / 3
        
        return (
            0.35 * sharpe_score +
            0.25 * drawdown_score +
            0.20 * trade_score +
            0.10 * volatility_score +
            0.10 * return_score
        )

    def _select_robust_parameters(self, fold_results):
        """Select robust parameters weighted by Sharpe ratio."""
        params_list = [r['optimal_params'] for r in fold_results]
        sharpe_list = [r['test_sharpe'] for r in fold_results]
        
        weights = np.maximum(sharpe_list, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(sharpe_list)) / len(sharpe_list)
        
        robust_params = {}
        for key in params_list[0].keys():
            weighted_sum = sum(w * p[key] for w, p in zip(weights, params_list))
            if key in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                robust_params[key] = int(round(weighted_sum / 5) * 5)
            else:
                robust_params[key] = round(weighted_sum, 3)
        
        return robust_params

    def _test_robustness(self, data, params):
        """Test parameter sensitivity."""
        print("\n📊 Parameter Sensitivity Test...")
        
        sensitivity_results = []
        
        for param, value in params.items():
            if param in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                step = max(5, int(value * 0.05))
            else:
                step = value * 0.05
            
            variations = [-2*step, -step, step, 2*step]
            
            for var in variations:
                test_params = params.copy()
                if param in ['lookback_days', 'rebalance_min_days', 'rebalance_max_days']:
                    test_params[param] = int(round(value + var))
                    if test_params[param] < 10:
                        continue
                else:
                    test_params[param] = round(value + var, 3)
                    if test_params[param] < 0:
                        continue
                
                test_metrics = self._run_backtest(data, test_params)
                sensitivity_results.append({
                    'param': param,
                    'variation': var,
                    'value': test_params[param],
                    'sharpe': test_metrics['sharpe_ratio']
                })
        
        if len(sensitivity_results) == 0:
            print("  ⚠️ No sensitivity results")
            return 0.5
        
        sharpe_values = [r['sharpe'] for r in sensitivity_results]
        sharpe_std = np.std(sharpe_values)
        sharpe_mean = np.mean(sharpe_values)
        
        print(f"  Sharpe std: {sharpe_std:.3f}")
        print(f"  Average Sharpe: {sharpe_mean:.3f}")
        
        worst = min(sensitivity_results, key=lambda x: x['sharpe'])
        best = max(sensitivity_results, key=lambda x: x['sharpe'])
        print(f"  Worst: {worst['param']} {worst['variation']:+.1%} → Sharpe={worst['sharpe']:.3f}")
        print(f"  Best: {best['param']} {best['variation']:+.1%} → Sharpe={best['sharpe']:.3f}")
        
        if sharpe_std < 0.2:
            print("  ✅ ROBUST (low sensitivity)")
            return 1.0
        elif sharpe_std < 0.4:
            print("  ⚠️ MODERATELY ROBUST")
            return 0.6
        else:
            print("  ❌ SENSITIVE (high variance)")
            return 0.2

    def get_results_dataframe(self):
        """Get fold results as DataFrame."""
        if not self.fold_results:
            return pd.DataFrame()
        
        rows = []
        for r in self.fold_results:
            rows.append({
                'fold': r['fold'],
                'sharpe': r['test_sharpe'],
                'return': r['test_return'],
                'drawdown': r['test_drawdown'],
                'volatility': r.get('test_volatility', 0),
                'num_trades': r.get('num_trades', 0),
                'lookback': r['optimal_params']['lookback_days'],
                'rebalance_min': r['optimal_params']['rebalance_min_days'],
                'rebalance_max': r['optimal_params']['rebalance_max_days'],
                'drift': r['optimal_params']['drift_threshold'],
                'take_profit': r['optimal_params']['take_profit_pct'],
                'train_start': r['train_start'],
                'train_end': r['train_end'],
                'test_start': r['test_start'],
                'test_end': r['test_end']
            })
        
        return pd.DataFrame(rows)