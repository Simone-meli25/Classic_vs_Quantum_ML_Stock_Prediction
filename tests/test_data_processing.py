"""Unit tests for the data processing module."""

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.data_processing import (
    clean_stock_data,
    extract_features,
    prepare_data_for_lstm,
    inverse_transform_predictions
)

class TestDataProcessing(unittest.TestCase):
    """Tests for data processing functions."""
    
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
        
        # Create sample data with NaN values
        self.data_with_nan = self.sample_data.copy()
        self.data_with_nan.iloc[1, 2] = np.nan  # Set Low value to NaN
        
        # Create a larger dataset for sequence preparation
        dates = pd.date_range('2020-01-01', periods=100)
        values = np.sin(np.linspace(0, 10, 100)) * 10 + 100  # Sine wave around 100
        self.larger_data = pd.DataFrame({
            'Open': values - 1,
            'High': values + 2,
            'Low': values - 2,
            'Close': values,
            'Volume': np.random.randint(1000, 2000, 100)
        }, index=dates)
    
    def test_clean_stock_data(self):
        """Test cleaning stock data."""
        # Test with data that has DatetimeIndex
        result = clean_stock_data(self.sample_data)
        
        # Check if result has Date column
        self.assertIn('Date', result.columns)
        
        # Check if values are preserved
        np.testing.assert_array_equal(result['Close'].values, self.sample_data['Close'].values)
        
        # Test with data that has NaN values
        result_nan = clean_stock_data(self.data_with_nan)
        
        # Check if NaN values were filled
        self.assertFalse(result_nan.isnull().any().any())
    
    def test_extract_features(self):
        """Test feature extraction."""
        # Use the larger dataset for more meaningful features
        result = extract_features(self.larger_data)
        
        # Check if expected features are created
        expected_features = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                           'Day', 'Month', 'Year', 'DayOfWeek',
                           'MA5', 'MA20', 'MA50', 'Volatility5', 'Volatility20',
                           'ROC5', 'ROC20']
        
        for feature in expected_features:
            self.assertIn(feature, result.columns)
        
        # Check that moving averages are calculated correctly for the first few points
        # where enough data is available
        expected_ma5 = self.larger_data['Close'].rolling(window=5).mean()
        pd.testing.assert_series_equal(result['MA5'][4:], expected_ma5[4:])
        
        # Check rate of change calculation
        expected_roc5 = self.larger_data['Close'].pct_change(periods=5) * 100
        pd.testing.assert_series_equal(result['ROC5'][5:], expected_roc5[5:], check_exact=False)
    
    def test_prepare_data_for_lstm(self):
        """Test preparation of data for LSTM."""
        window_size = 10
        test_size = 0.2
        
        # Use the larger dataset
        X_train, y_train, X_test, y_test, scaler = prepare_data_for_lstm(
            self.larger_data,
            target_col='Close',
            window_size=window_size,
            test_size=test_size
        )
        
        # Check shapes
        expected_total_samples = len(self.larger_data) - window_size
        expected_train_samples = int(expected_total_samples * (1 - test_size))
        expected_test_samples = expected_total_samples - expected_train_samples + window_size - 1
        
        self.assertEqual(X_train.shape[0], expected_train_samples)
        self.assertEqual(X_train.shape[1], window_size)
        self.assertEqual(X_train.shape[2], 1)  # Only using Close column
        
        self.assertEqual(y_train.shape[0], expected_train_samples)
        
        self.assertLessEqual(X_test.shape[0], expected_test_samples)
        self.assertEqual(X_test.shape[1], window_size)
        self.assertEqual(X_test.shape[2], 1)
        
        # Check if scaler is returned correctly
        self.assertIsInstance(scaler, MinMaxScaler)
        
        # Test with multiple feature columns
        X_train_multi, y_train_multi, X_test_multi, y_test_multi, scaler_multi = prepare_data_for_lstm(
            self.larger_data,
            target_col='Close',
            window_size=window_size,
            test_size=test_size,
            feature_cols=['Close', 'Volume']
        )
        
        # Check that the feature dimension has increased
        self.assertEqual(X_train_multi.shape[2], 2)  # Two features
    
    def test_inverse_transform_predictions(self):
        """Test inverse transformation of predictions."""
        # Create a scaler and fit it to some data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = np.array([[100], [110], [120], [130], [140]])
        scaler.fit(data)
        
        # Scale the data
        scaled_data = scaler.transform(data)
        
        # Test inverse transform
        orig_values = inverse_transform_predictions(scaled_data[:, 0], scaler)
        
        # Check if the values are close to the original data
        np.testing.assert_allclose(orig_values, data[:, 0], rtol=1e-5)
        
        # Test with a multi-feature scaler
        multi_scaler = MinMaxScaler(feature_range=(0, 1))
        multi_data = np.array([[100, 1000], [110, 1100], [120, 1200], [130, 1300], [140, 1400]])
        multi_scaler.fit(multi_data)
        
        # Scale the data
        scaled_multi = multi_scaler.transform(multi_data)
        
        # Test inverse transform for first feature
        orig_first = inverse_transform_predictions(scaled_multi[:, 0], multi_scaler, target_idx=0)
        np.testing.assert_allclose(orig_first, multi_data[:, 0], rtol=1e-5)
        
        # Test inverse transform for second feature
        orig_second = inverse_transform_predictions(scaled_multi[:, 1], multi_scaler, target_idx=1)
        np.testing.assert_allclose(orig_second, multi_data[:, 1], rtol=1e-5)


if __name__ == '__main__':
    unittest.main() 