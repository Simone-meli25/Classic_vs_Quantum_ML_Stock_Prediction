"""Unit tests for the data collection module."""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.data_collection import fetch_stock_data, get_ticker_info, fetch_multiple_stocks

class TestDataCollection(unittest.TestCase):
    """Tests for data collection functions."""
    
    @patch('src.data.data_collection.yf.download')
    def test_fetch_stock_data_with_period(self, mock_download):
        """Test fetching stock data with a period parameter."""
        # Create mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        # Configure the mock to return the mock data
        mock_download.return_value = mock_data
        
        # Call the function
        result = fetch_stock_data('AAPL', period='1mo')
        
        # Assert the mock was called with expected arguments
        mock_download.assert_called_once_with('AAPL', period='1mo', interval='1d', progress=False)
        
        # Assert the result is correct
        pd.testing.assert_frame_equal(result, mock_data)
    
    @patch('src.data.data_collection.yf.download')
    def test_fetch_stock_data_with_date_range(self, mock_download):
        """Test fetching stock data with date range parameters."""
        # Create mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        # Configure the mock to return the mock data
        mock_download.return_value = mock_data
        
        # Call the function
        result = fetch_stock_data('AAPL', start_date='2020-01-01', end_date='2020-01-31')
        
        # Assert the mock was called with expected arguments
        mock_download.assert_called_once_with(
            'AAPL', 
            start='2020-01-01', 
            end='2020-01-31', 
            interval='1d',
            progress=False
        )
        
        # Assert the result is correct
        pd.testing.assert_frame_equal(result, mock_data)
    
    @patch('src.data.data_collection.yf.download')
    def test_fetch_stock_data_empty_result(self, mock_download):
        """Test handling of empty data results."""
        # Configure the mock to return empty DataFrame
        mock_download.return_value = pd.DataFrame()
        
        # Call the function
        result = fetch_stock_data('INVALID', period='1mo')
        
        # Assert the result is an empty DataFrame
        self.assertTrue(result.empty)
    
    @patch('src.data.data_collection.yf.download')
    def test_fetch_stock_data_missing_params(self, mock_download):
        """Test error handling when required parameters are missing."""
        # Call the function without required parameters
        with self.assertRaises(ValueError):
            fetch_stock_data('AAPL')
    
    @patch('src.data.data_collection.yf.Ticker')
    def test_get_ticker_info(self, mock_ticker_class):
        """Test getting ticker information."""
        # Create mock ticker object
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'shortName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics'
        }
        mock_ticker_class.return_value = mock_ticker
        
        # Call the function
        result = get_ticker_info('AAPL')
        
        # Assert the mock was called correctly
        mock_ticker_class.assert_called_once_with('AAPL')
        
        # Assert the result is correct
        self.assertEqual(result, mock_ticker.info)
    
    @patch('src.data.data_collection.fetch_stock_data')
    def test_fetch_multiple_stocks(self, mock_fetch_stock_data):
        """Test fetching data for multiple stocks."""
        # Create mock data for each stock
        aapl_data = pd.DataFrame({
            'Close': [150, 151, 152]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        msft_data = pd.DataFrame({
            'Close': [250, 251, 252]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        # Configure the mock to return different data for each call
        mock_fetch_stock_data.side_effect = [aapl_data, msft_data]
        
        # Call the function
        result = fetch_multiple_stocks(['AAPL', 'MSFT'], period='1mo')
        
        # Assert fetch_stock_data was called twice with expected arguments
        self.assertEqual(mock_fetch_stock_data.call_count, 2)
        mock_fetch_stock_data.assert_any_call('AAPL', period='1mo', start_date=None, end_date=None, interval='1d')
        mock_fetch_stock_data.assert_any_call('MSFT', period='1mo', start_date=None, end_date=None, interval='1d')
        
        # Assert the result has expected structure
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        pd.testing.assert_frame_equal(result['AAPL'], aapl_data)
        pd.testing.assert_frame_equal(result['MSFT'], msft_data)
    
    @patch('src.data.data_collection.fetch_stock_data')
    def test_fetch_multiple_stocks_with_error(self, mock_fetch_stock_data):
        """Test handling errors when fetching multiple stocks."""
        # Configure the mock to raise an exception for the second call
        aapl_data = pd.DataFrame({
            'Close': [150, 151, 152]
        }, index=pd.date_range('2020-01-01', periods=3))
        mock_fetch_stock_data.side_effect = [aapl_data, Exception("Connection error")]
        
        # Call the function
        result = fetch_multiple_stocks(['AAPL', 'ERROR'], period='1mo')
        
        # Assert the result contains data for the successful call and empty DataFrame for the error
        self.assertIn('AAPL', result)
        self.assertIn('ERROR', result)
        pd.testing.assert_frame_equal(result['AAPL'], aapl_data)
        self.assertTrue(result['ERROR'].empty)


if __name__ == '__main__':
    unittest.main() 