"""Unit tests for the visualization module."""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.plots import (
    plot_stock_prices,
    plot_predictions,
    plot_model_comparison,
    plot_correlation_matrix,
    plot_feature_importance
)

class TestVisualization(unittest.TestCase):
    """Tests for visualization functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample stock data
        self.sample_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range('2020-01-01', periods=5))
        
        # Create prediction data
        self.actual = np.array([100, 101, 102, 103, 104])
        self.predicted = np.array([99, 101.5, 102.5, 102, 105])
        self.dates = pd.date_range('2020-01-01', periods=5)
    
    def test_plot_stock_prices(self):
        """Test plotting stock prices."""
        # Test basic plotting
        fig = plot_stock_prices(self.sample_data)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with ticker and title
        fig = plot_stock_prices(
            self.sample_data, 
            ticker="AAPL", 
            title="Apple Stock Price",
            show_volume=True
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with different axes
        ax1 = fig.axes[0]  # Price plot
        ax2 = fig.axes[1]  # Volume plot
        self.assertEqual(len(fig.axes), 2)
        
        # Clean up
        plt.close(fig)
        
        # Test with data that doesn't have DatetimeIndex
        data_no_index = self.sample_data.reset_index()
        fig = plot_stock_prices(data_no_index, date_col='index')
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_predictions(self):
        """Test plotting predictions vs actual values."""
        # Test basic plotting
        fig = plot_predictions(self.actual, self.predicted)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with dates and ticker
        fig = plot_predictions(
            self.actual,
            self.predicted,
            dates=self.dates,
            ticker="AAPL"
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Check subplots
        self.assertEqual(len(fig.axes), 2)  # Price and error subplots
        
        # Clean up
        plt.close(fig)
    
    def test_plot_model_comparison(self):
        """Test plotting model comparison."""
        # Create comparison data
        models = {
            'Model A': 5.21,
            'Model B': 4.85,
            'Model C': 4.32
        }
        
        # Test basic plotting
        fig = plot_model_comparison(models)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with different metric
        fig = plot_model_comparison(models, metric='mae')
        self.assertIsInstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    def test_plot_correlation_matrix(self):
        """Test plotting correlation matrix."""
        # Create correlated data
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [1, 3, 5, 7, 9]
        })
        
        # Test basic plotting
        fig = plot_correlation_matrix(data)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with title
        fig = plot_correlation_matrix(
            data,
            title="Test Correlation Matrix"
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    def test_plot_feature_importance(self):
        """Test plotting feature importance."""
        # Create feature importance data
        features = ['Feature A', 'Feature B', 'Feature C', 'Feature D']
        importances = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Test basic plotting
        fig = plot_feature_importance(features, importances)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with title
        fig = plot_feature_importance(
            features,
            importances,
            title="Test Feature Importance"
        )
        self.assertIsInstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)


if __name__ == '__main__':
    unittest.main() 