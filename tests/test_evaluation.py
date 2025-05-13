"""Unit tests for the evaluation module."""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.evaluation import (
    calculate_metrics,
    compare_models,
    rolling_prediction_analysis,
    trading_strategy_evaluation,
    plot_trading_strategy
)

class TestEvaluation(unittest.TestCase):
    """Tests for evaluation functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create perfect prediction data
        self.y_true_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Create imperfect prediction data
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.2, 1.8, 3.2, 3.8, 5.1])
        
        # Create predictions with wrong directional movement
        self.y_true_direction = np.array([1.0, 2.0, 1.5, 2.5, 2.0])
        self.y_pred_direction = np.array([1.1, 1.9, 1.7, 2.3, 2.6])  # Last value has wrong direction
        
        # Multiple model predictions
        self.model_preds = {
            'Model A': self.y_pred,
            'Model B': self.y_pred_perfect
        }
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        # Test perfect prediction
        metrics_perfect = calculate_metrics(self.y_true_perfect, self.y_pred_perfect)
        
        # Check metrics for perfect prediction
        self.assertEqual(metrics_perfect['mse'], 0.0)
        self.assertEqual(metrics_perfect['rmse'], 0.0)
        self.assertEqual(metrics_perfect['mae'], 0.0)
        self.assertEqual(metrics_perfect['r2'], 1.0)
        self.assertEqual(metrics_perfect['directional_accuracy'], 1.0)
        
        # Test imperfect prediction
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        # Check metrics for imperfect prediction
        self.assertGreater(metrics['mse'], 0.0)
        self.assertGreater(metrics['rmse'], 0.0)
        self.assertGreater(metrics['mae'], 0.0)
        self.assertLessEqual(metrics['r2'], 1.0)
        
        # Test directional accuracy with wrong direction
        metrics_direction = calculate_metrics(self.y_true_direction, self.y_pred_direction)
        expected_dir_acc = 0.75  # 3 out of 4 directions correct
        self.assertAlmostEqual(metrics_direction['directional_accuracy'], expected_dir_acc)
        
        # Test with 2D arrays
        metrics_2d_true = calculate_metrics(self.y_true.reshape(-1, 1), self.y_pred)
        metrics_2d_pred = calculate_metrics(self.y_true, self.y_pred.reshape(-1, 1))
        
        # Results should be the same regardless of shape
        self.assertEqual(metrics['mse'], metrics_2d_true['mse'])
        self.assertEqual(metrics['rmse'], metrics_2d_pred['rmse'])
    
    def test_compare_models(self):
        """Test model comparison."""
        # Test with default metrics
        comparison = compare_models(self.y_true, self.model_preds)
        
        # Check that result is a DataFrame with expected structure
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(comparison.shape, (2, 5))  # 2 models, 5 default metrics
        
        # Check model names
        self.assertIn('Model A', comparison.index)
        self.assertIn('Model B', comparison.index)
        
        # Check metrics
        self.assertIn('rmse', comparison.columns)
        self.assertIn('mae', comparison.columns)
        self.assertIn('r2', comparison.columns)
        self.assertIn('directional_accuracy', comparison.columns)
        
        # Check values
        self.assertEqual(comparison.loc['Model B', 'rmse'], 0.0)  # Perfect model
        self.assertGreater(comparison.loc['Model A', 'rmse'], 0.0)  # Imperfect model
        
        # Test with custom metrics
        custom_metrics = ['mse', 'r2']
        custom_comparison = compare_models(self.y_true, self.model_preds, metrics=custom_metrics)
        
        # Check that only specified metrics are included
        self.assertEqual(custom_comparison.shape, (2, 2))
        self.assertIn('mse', custom_comparison.columns)
        self.assertIn('r2', custom_comparison.columns)
        self.assertNotIn('mae', custom_comparison.columns)
    
    def test_rolling_prediction_analysis(self):
        """Test rolling prediction analysis."""
        # Create longer series for meaningful rolling analysis
        n = 30
        x = np.linspace(0, 10, n)
        true_vals = np.sin(x) * 10 + 50
        pred_vals = true_vals + np.random.normal(0, 1, n)
        
        window_size = 5
        
        # Calculate rolling metrics
        rmse, mae, r2 = rolling_prediction_analysis(
            true_vals,
            pred_vals,
            window_size=window_size
        )
        
        # Check shapes
        expected_length = n - window_size + 1
        self.assertEqual(len(rmse), expected_length)
        self.assertEqual(len(mae), expected_length)
        self.assertEqual(len(r2), expected_length)
        
        # Values should be reasonable
        self.assertTrue(np.all(rmse >= 0))
        self.assertTrue(np.all(mae >= 0))
        self.assertTrue(np.all(r2 <= 1.0))
    
    def test_trading_strategy_evaluation(self):
        """Test trading strategy evaluation."""
        # Create price series with clear up and down trends
        prices = np.array([100, 102, 104, 103, 101, 105, 110, 108])
        
        # Perfect prediction: same as actual prices shifted one step earlier
        perfect_pred = np.array([102, 104, 103, 101, 105, 110, 108, 107])
        
        # Evaluate strategy
        final_return, buy_hold_return, returns_over_time, buy_hold_over_time = trading_strategy_evaluation(
            prices,
            perfect_pred
        )
        
        # Check shapes
        self.assertEqual(len(returns_over_time), len(prices))
        self.assertEqual(len(buy_hold_over_time), len(prices))
        
        # Initial returns should be 1.0
        self.assertEqual(returns_over_time[0], 1.0)
        self.assertEqual(buy_hold_over_time[0], 1.0)
        
        # Final buy and hold return
        expected_buyhold = prices[-1] / prices[0] - 1
        self.assertAlmostEqual(buy_hold_return, expected_buyhold)
        
        # Perfect prediction should outperform buy and hold
        self.assertGreater(final_return, buy_hold_return)
        
        # Test with wrong predictions (opposite of actual movements)
        opposite_pred = 2 * prices[0] - perfect_pred  # Flipped around the first price
        
        bad_final_return, _, bad_returns_over_time, _ = trading_strategy_evaluation(
            prices,
            opposite_pred
        )
        
        # Bad strategy should underperform
        self.assertLess(bad_final_return, buy_hold_return)
    
    def test_plot_trading_strategy(self):
        """Test plotting trading strategy returns."""
        # Create returns data
        returns = np.array([1.0, 1.02, 1.05, 1.03, 1.06, 1.1])
        buy_hold = np.array([1.0, 1.01, 1.03, 1.02, 1.04, 1.05])
        
        # Test basic plotting
        fig = plot_trading_strategy(returns, buy_hold)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with dates
        dates = pd.date_range('2020-01-01', periods=len(returns))
        fig = plot_trading_strategy(returns, buy_hold, dates=dates)
        self.assertIsInstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)


if __name__ == '__main__':
    unittest.main() 