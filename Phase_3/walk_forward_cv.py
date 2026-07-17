"""
walk_forward_cv.py - Walk-forward cross-validation for time-series data.
Prevents look-ahead bias and overfitting.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class WalkForwardCV:
    def __init__(self, train_size_years=5, test_size_years=1, step_size_years=1):
        """
        Walk-forward cross-validation splitter.
        
        Args:
            train_size_years: Years of data for training (optimisation)
            test_size_years: Years of data for testing (validation)
            step_size_years: Years to step forward each fold
        """
        self.train_size = train_size_years * 252
        self.test_size = test_size_years * 252
        self.step_size = step_size_years * 252
        
        print(f"📊 Walk-Forward CV:")
        print(f"   Train: {train_size_years} yrs ({self.train_size} days)")
        print(f"   Test:  {test_size_years} yrs ({self.test_size} days)")
        print(f"   Step:  {step_size_years} yrs ({self.step_size} days)")
    
    def split(self, data):
        """Generate walk-forward splits."""
        n = len(data)
        folds = []
        
        for start in range(0, n - self.train_size - self.test_size, self.step_size):
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            folds.append((range(start, train_end), range(train_end, test_end)))
        
        return folds
    
    def get_fold_info(self, data, fold_idx, train_idx, test_idx):
        """Get human-readable fold information."""
        return {
            'fold': fold_idx + 1,
            'train_start': data.index[train_idx[0]],
            'train_end': data.index[train_idx[-1]],
            'test_start': data.index[test_idx[0]],
            'test_end': data.index[test_idx[-1]],
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        }
    
    def get_fold_data(self, data, train_idx, test_idx):
        """Get the actual data for a fold."""
        return {
            'train': data.iloc[train_idx],
            'test': data.iloc[test_idx]
        }
    
    def print_summary(self, data, folds):
        """Print a summary of all folds."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD CV SUMMARY")
        print("=" * 70)
        
        for idx, (train_idx, test_idx) in enumerate(folds):
            info = self.get_fold_info(data, idx, train_idx, test_idx)
            print(f"\nFold {idx + 1}:")
            print(f"  Train: {info['train_start'].strftime('%Y-%m-%d')} → {info['train_end'].strftime('%Y-%m-%d')} ({info['train_size']} days)")
            print(f"  Test:  {info['test_start'].strftime('%Y-%m-%d')} → {info['test_end'].strftime('%Y-%m-%d')} ({info['test_size']} days)")