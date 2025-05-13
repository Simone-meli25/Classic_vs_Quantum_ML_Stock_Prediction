"""Data collection module for stock prediction.

This module provides functions to fetch historical stock data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_stock_data(
    ticker: str,
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL' for Apple)
        period: Time period to download (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        start_date: Start date in 'YYYY-MM-DD' format (if period is None)
        end_date: End date in 'YYYY-MM-DD' format (if period is None)
        interval: Data interval (e.g., '1d', '1wk', '1mo')
    
    Returns:
        DataFrame containing historical stock data
    
    Examples:
        >>> # Get 5 years of Apple data
        >>> data = fetch_stock_data('AAPL', period='5y')
        >>> 
        >>> # Get data between specific dates
        >>> data = fetch_stock_data('MSFT', start_date='2020-01-01', end_date='2021-01-01')
    """
    try:
        logger.info(f"Fetching data for {ticker}")
        
        if period is not None:
            # Download data for specified period
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            logger.info(f"Downloaded {len(data)} records for {ticker} (period: {period})")
        elif start_date is not None and end_date is not None:
            # Download data for date range
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            logger.info(f"Downloaded {len(data)} records for {ticker} (from {start_date} to {end_date})")
        else:
            raise ValueError("Either 'period' or both 'start_date' and 'end_date' must be provided")
        
        # Check if data is empty
        if len(data) == 0:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        raise

def get_ticker_info(ticker: str) -> dict:
    """Get information about a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary containing ticker information
    """
    try:
        logger.info(f"Getting info for {ticker}")
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.info
    
    except Exception as e:
        logger.error(f"Error getting info for {ticker}: {str(e)}")
        raise

def fetch_multiple_stocks(
    tickers: list[str],
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for multiple stocks.
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period to download
        start_date: Start date in 'YYYY-MM-DD' format (if period is None)
        end_date: End date in 'YYYY-MM-DD' format (if period is None)
        interval: Data interval
    
    Returns:
        Dictionary with tickers as keys and DataFrames as values
    """
    result = {}
    
    for ticker in tickers:
        try:
            data = fetch_stock_data(
                ticker, 
                period=period, 
                start_date=start_date, 
                end_date=end_date, 
                interval=interval
            )
            result[ticker] = data
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            result[ticker] = pd.DataFrame()  # Empty DataFrame for failed fetches
    
    return result

if __name__ == "__main__":
    # Example usage
    apple_data = fetch_stock_data("AAPL", period="1y")
    print(apple_data.head()) 