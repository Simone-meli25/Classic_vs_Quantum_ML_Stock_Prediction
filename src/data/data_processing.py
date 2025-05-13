"""Data processing module for stock prediction.

This module provides functions to preprocess stock data for machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare stock data.
    
    Args:
        df: DataFrame containing stock data
    
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Reset index to make Date a column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # Check for missing values
    if df.isnull().any().any():
        logger.warning(f"Found {df.isnull().sum().sum()} missing values. Filling with forward fill.")
        df = df.fillna(method='ffill')
        # If there are still NaN values (at the beginning), fill with backward fill
        df = df.fillna(method='bfill')
    
    return df

def extract_features(df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
    """Extract features from stock data for prediction.
    
    Args:
        df: DataFrame containing stock data
        price_column: Column name for the price to use for feature extraction
    
    Returns:
        DataFrame with extracted features
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Ensure date is in datetime format
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract basic time features
    if 'Date' in df.columns:
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Calculate rolling statistics
    if price_column in df.columns:
        # Moving averages
        df['MA5'] = df[price_column].rolling(window=5).mean()
        df['MA20'] = df[price_column].rolling(window=20).mean()
        df['MA50'] = df[price_column].rolling(window=50).mean()
        
        # Volatility (standard deviation)
        df['Volatility5'] = df[price_column].rolling(window=5).std()
        df['Volatility20'] = df[price_column].rolling(window=20).std()
        
        # Momentum indicators
        df['ROC5'] = df[price_column].pct_change(periods=5) * 100  # Rate of Change
        df['ROC20'] = df[price_column].pct_change(periods=20) * 100
        
    # Drop NaN values created by rolling windows
    df = df.dropna()
    
    return df

def prepare_data_for_lstm(
    df: pd.DataFrame, 
    target_col: str = 'Close',
    window_size: int = 60,
    test_size: float = 0.2,
    feature_cols: Optional[list] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare data for LSTM model training.
    
    Args:
        df: DataFrame containing stock data
        target_col: Column name for the target variable (price to predict)
        window_size: Number of time steps to use for each prediction
        test_size: Proportion of data to use for testing
        feature_cols: List of column names to use as features (default: only target_col)
    
    Returns:
        Tuple containing X_train, y_train, X_test, y_test, and the scaler used
    """
    df = df.copy()
    
    # If no feature columns provided, use only the target column
    if feature_cols is None:
        feature_cols = [target_col]
    
    # Extract the feature data
    data = df[feature_cols].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Determine training/testing split
    train_size = int(len(scaled_data) * (1 - test_size))
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - window_size:]  # Include some training data for initial window
    
    logger.info(f"Training samples: {train_size}, Testing samples: {len(scaled_data) - train_size}")
    
    # Create sequences for the training data
    X_train, y_train = [], []
    
    for i in range(window_size, len(train_data)):
        # For multivariate input, use all feature columns
        X_train.append(train_data[i-window_size:i])
        # For the target, use only the target column
        target_idx = feature_cols.index(target_col) if len(feature_cols) > 1 else 0
        y_train.append(train_data[i, target_idx])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Create sequences for the test data
    X_test, y_test = [], []
    
    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i])
        target_idx = feature_cols.index(target_col) if len(feature_cols) > 1 else 0
        y_test.append(test_data[i, target_idx])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(feature_cols))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(feature_cols))
    
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler

def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler, target_idx: int = 0) -> np.ndarray:
    """Convert scaled predictions back to their original scale.
    
    Args:
        predictions: Predicted values in scaled form
        scaler: The scaler used to scale the original data
        target_idx: Index of the target variable in the scaled data
    
    Returns:
        Predictions in the original scale
    """
    # Reshape predictions to 2D array (required by the scaler)
    pred_reshaped = predictions.reshape(-1, 1)
    
    # Create a dummy array to perform the inverse transform
    dummy = np.zeros((len(pred_reshaped), scaler.scale_.shape[0]))
    dummy[:, target_idx] = pred_reshaped.flatten()
    
    # Inverse transform
    inverted = scaler.inverse_transform(dummy)
    
    # Extract the target column
    return inverted[:, target_idx]

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get some sample data
    data = yf.download("AAPL", period="1y")
    
    # Clean and prepare data
    cleaned_data = clean_stock_data(data)
    
    # Prepare for LSTM
    X_train, y_train, X_test, y_test, scaler = prepare_data_for_lstm(cleaned_data)
    
    print(f"Sample X_train shape: {X_train.shape}")
    print(f"Sample y_train shape: {y_train.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}") 