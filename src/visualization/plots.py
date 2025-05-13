"""Visualization module for stock prediction project.

This module provides functions for visualizing stock data and model results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_stock_prices(
    data: pd.DataFrame,
    price_col: str = 'Close',
    date_col: Optional[str] = None,
    title: Optional[str] = None,
    ticker: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_volume: bool = False
) -> plt.Figure:
    """Plot stock price data.
    
    Args:
        data: DataFrame containing stock data
        price_col: Column name for price data
        date_col: Column name for date data (if None, assumes DatetimeIndex)
        title: Plot title
        ticker: Stock ticker symbol for title
        figsize: Figure size (width, height)
        show_volume: Whether to include volume subplot
    
    Returns:
        Matplotlib figure
    """
    # Copy data to avoid modifying the original
    df = data.copy()
    
    # Handle date index
    if date_col is None and isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
    elif date_col is not None and date_col in df.columns:
        dates = df[date_col]
    else:
        # Create a dummy date index
        dates = pd.date_range(start='2020-01-01', periods=len(df))
    
    # Create figure
    if show_volume and 'Volume' in df.columns:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot price data
    ax1.plot(dates, df[price_col], label=price_col, color='blue')
    
    # Plot moving averages if available
    if 'MA5' in df.columns:
        ax1.plot(dates, df['MA5'], label='5-day MA', color='red', linestyle='--', alpha=0.7)
    if 'MA20' in df.columns:
        ax1.plot(dates, df['MA20'], label='20-day MA', color='green', linestyle='--', alpha=0.7)
    if 'MA50' in df.columns:
        ax1.plot(dates, df['MA50'], label='50-day MA', color='purple', linestyle='--', alpha=0.7)
    
    # Configure price plot
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title(f"Stock Price: {ticker}" if ticker else "Stock Price")
    
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Format date axis
    if len(dates) > 0:
        if (dates.max() - dates.min()).days > 365:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Plot volume if requested
    if show_volume and 'Volume' in df.columns:
        ax2.bar(dates, df['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    dates: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None,
    ticker: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot actual vs predicted stock prices.
    
    Args:
        actual: Array of actual prices
        predicted: Array of predicted prices
        dates: Array of dates corresponding to the prices
        ticker: Stock ticker symbol for title
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib figure
    """
    # Create dummy dates if none provided
    if dates is None:
        dates = pd.date_range(start='2020-01-01', periods=len(actual))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot actual vs predicted
    ax1.plot(dates, actual, label='Actual', color='blue')
    ax1.plot(dates, predicted, label='Predicted', color='red', linestyle='--')
    
    # Calculate metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Configure plot
    title = f"{ticker} - Actual vs Predicted" if ticker else "Actual vs Predicted"
    ax1.set_title(f"{title}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}")
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot errors
    errors = predicted - actual
    ax2.bar(dates, errors, color='gray', alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Error')
    ax2.grid(True, alpha=0.3)
    
    # Format date axis
    if len(dates) > 0:
        if isinstance(dates, pd.DatetimeIndex) or isinstance(dates[0], (pd.Timestamp, np.datetime64)):
            if (dates.max() - dates.min()).days > 365:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            else:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    
    return fig

def plot_model_comparison(
    models_results: dict,
    metric: str = 'rmse',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot comparison of multiple models.
    
    Args:
        models_results: Dictionary with model names as keys and metric values as values
        metric: Metric name (e.g., 'rmse', 'mae')
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    models = list(models_results.keys())
    values = [models_results[model] for model in models]
    
    # Create bar plot
    bars = ax.bar(models, values, color=sns.color_palette('muted'))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02 * max(values),
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )
    
    # Configure plot
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'Model Comparison ({metric.upper()})')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig

def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = 'Feature Correlation Matrix',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot correlation matrix of features.
    
    Args:
        data: DataFrame containing features
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib figure
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=True,
        fmt=".2f",
        ax=ax
    )
    
    # Configure plot
    ax.set_title(title)
    
    plt.tight_layout()
    
    return fig

def plot_feature_importance(
    features: List[str],
    importances: np.ndarray,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot feature importance.
    
    Args:
        features: List of feature names
        importances: Array of feature importance values
        title: Plot title
        figsize: Figure size (width, height)
    
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    indices = np.argsort(importances)
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    ax.barh(range(len(sorted_features)), sorted_importances, color='skyblue')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    
    # Configure plot
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Get sample data
    data = yf.download("AAPL", period="1y")
    
    # Plot stock price
    fig = plot_stock_prices(data, ticker="AAPL", show_volume=True)
    plt.show()
    
    # Example model comparison
    models = {
        'LSTM': 5.21,
        'Quantum': 4.85,
        'Ensemble': 4.32
    }
    fig = plot_model_comparison(models)
    plt.show() 