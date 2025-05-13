"""Evaluation utilities for model performance.

This module provides functions for evaluating and comparing model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate regression metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with calculated metrics
    """
    # Ensure input arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE with handling for zeros
    # Replace zeros with a small value to avoid division by zero
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    mape = mean_absolute_percentage_error(y_true_safe, y_pred)
    
    # Calculate directional accuracy (percentage of correct up/down predictions)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    
    # Calculate if the direction (up/down) is the same
    correct_direction = np.sign(y_true_diff) == np.sign(y_pred_diff)
    directional_accuracy = np.mean(correct_direction)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

def compare_models(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compare multiple models using various metrics.
    
    Args:
        y_true: True values
        model_predictions: Dictionary with model names as keys and predictions as values
        metrics: List of metrics to calculate (default: all)
    
    Returns:
        DataFrame with model comparison results
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2', 'mape', 'directional_accuracy']
    
    # Initialize results dictionary
    results = {}
    
    # Calculate metrics for each model
    for model_name, y_pred in model_predictions.items():
        model_metrics = calculate_metrics(y_true, y_pred)
        results[model_name] = {metric: model_metrics[metric] for metric in metrics}
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    return results_df

def rolling_prediction_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_size: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate rolling metrics to analyze prediction performance over time.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        window_size: Size of rolling window
    
    Returns:
        Tuple of (rolling_rmse, rolling_mae, rolling_r2)
    """
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    n = len(y_true)
    rolling_rmse = np.zeros(n - window_size + 1)
    rolling_mae = np.zeros(n - window_size + 1)
    rolling_r2 = np.zeros(n - window_size + 1)
    
    for i in range(n - window_size + 1):
        # Extract window
        true_window = y_true[i:i+window_size]
        pred_window = y_pred[i:i+window_size]
        
        # Calculate metrics
        rolling_rmse[i] = np.sqrt(mean_squared_error(true_window, pred_window))
        rolling_mae[i] = mean_absolute_error(true_window, pred_window)
        rolling_r2[i] = r2_score(true_window, pred_window)
    
    return rolling_rmse, rolling_mae, rolling_r2

def trading_strategy_evaluation(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate a simple trading strategy based on predictions.
    
    The strategy is to buy when the model predicts a price increase and 
    sell when it predicts a decrease.
    
    Args:
        actual_prices: Array of actual stock prices
        predicted_prices: Array of predicted stock prices
        initial_capital: Initial investment amount
        transaction_cost: Transaction cost as a fraction of trade value
    
    Returns:
        Tuple of (final_return, buy_hold_return, returns_over_time, buy_hold_over_time)
    """
    # Flatten arrays
    actual_prices = actual_prices.flatten()
    predicted_prices = predicted_prices.flatten()
    
    # Calculate price changes predicted by the model
    predicted_changes = np.diff(predicted_prices)
    
    # Initial portfolio value
    portfolio_value = initial_capital
    shares = 0
    cash = initial_capital
    
    # Track returns over time
    returns_over_time = np.zeros(len(actual_prices))
    returns_over_time[0] = 1.0  # Initial return = 1 (no change)
    
    # Buy and hold returns for comparison
    initial_shares = initial_capital / actual_prices[0]
    buy_hold_over_time = np.zeros(len(actual_prices))
    buy_hold_over_time[0] = 1.0  # Initial return = 1 (no change)
    
    # Implement the trading strategy
    for i in range(1, len(actual_prices)):
        # Buy and hold strategy
        buy_hold_value = initial_shares * actual_prices[i]
        buy_hold_over_time[i] = buy_hold_value / initial_capital
        
        # Trading strategy based on predictions
        if i < len(predicted_changes) + 1:
            # If predict price increase and don't own shares, buy
            if predicted_changes[i-1] > 0 and shares == 0:
                shares = cash * (1 - transaction_cost) / actual_prices[i-1]
                cash = 0
            
            # If predict price decrease and own shares, sell
            elif predicted_changes[i-1] < 0 and shares > 0:
                cash = shares * actual_prices[i-1] * (1 - transaction_cost)
                shares = 0
        
        # Calculate current portfolio value
        current_value = cash + (shares * actual_prices[i])
        returns_over_time[i] = current_value / initial_capital
    
    # Calculate final returns
    final_return = returns_over_time[-1] - 1.0  # Subtract 1 to get percentage gain/loss
    buy_hold_return = buy_hold_over_time[-1] - 1.0
    
    return final_return, buy_hold_return, returns_over_time, buy_hold_over_time

def plot_trading_strategy(
    returns_over_time: np.ndarray,
    buy_hold_over_time: np.ndarray,
    dates: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """Plot the returns of the trading strategy vs buy and hold.
    
    Args:
        returns_over_time: Array of strategy returns over time
        buy_hold_over_time: Array of buy and hold returns over time
        dates: Optional array of dates
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Create x-axis values
    if dates is None:
        x_values = np.arange(len(returns_over_time))
    else:
        x_values = dates
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot returns
    ax.plot(x_values, returns_over_time, label='Trading Strategy', color='blue')
    ax.plot(x_values, buy_hold_over_time, label='Buy & Hold', color='red', linestyle='--')
    
    # Calculate final returns
    final_return = returns_over_time[-1] - 1.0
    buy_hold_return = buy_hold_over_time[-1] - 1.0
    
    # Add horizontal line at y=1 (initial investment)
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
    
    # Configure plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Return (1.0 = initial investment)')
    ax.set_title(f'Trading Strategy Returns\nStrategy: {final_return:.2%}, Buy & Hold: {buy_hold_return:.2%}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    n = 100
    time = np.linspace(0, 10, n)
    actual = np.sin(time) * 10 + 50
    model1_pred = actual + np.random.normal(0, 1, n)
    model2_pred = actual + np.random.normal(0, 2, n)
    
    # Compare models
    model_preds = {
        'Model 1': model1_pred,
        'Model 2': model2_pred
    }
    
    comparison = compare_models(actual, model_preds)
    print("Model Comparison:")
    print(comparison)
    
    # Evaluate trading strategy
    final_return, buy_hold_return, returns, buy_hold = trading_strategy_evaluation(
        actual, model1_pred
    )
    
    print(f"Trading strategy return: {final_return:.2%}")
    print(f"Buy & hold return: {buy_hold_return:.2%}") 