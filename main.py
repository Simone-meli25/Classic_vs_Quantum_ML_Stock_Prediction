#!/usr/bin/env python
"""Main script to run stock prediction models.

This script demonstrates the full workflow of stock price prediction:
1. Data collection from Yahoo Finance
2. Data preprocessing
3. Training classical LSTM model
4. Training quantum model with PennyLane
5. Model comparison and evaluation
6. Visualization of results
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import project modules
from src.data.data_collection import fetch_stock_data
from src.data.data_processing import (
    clean_stock_data,
    extract_features,
    prepare_data_for_lstm,
    inverse_transform_predictions
)
from src.models.classical import LSTMModel
from src.models.quantum import QuantumModel, QuantumKernelModel
from src.visualization.plots import (
    plot_stock_prices,
    plot_predictions,
    plot_model_comparison
)
from src.utils.evaluation import (
    calculate_metrics,
    compare_models,
    trading_strategy_evaluation,
    plot_trading_strategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Stock Price Prediction")
    
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="Time period to download (default: 5y)"
    )
    
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Window size for sequence data (default: 60)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    
    parser.add_argument(
        "--quantum-model",
        type=str,
        choices=["variational", "kernel", "both", "none"],
        default="variational",
        help="Quantum model type to use (default: variational)"
    )
    
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits for quantum models (default: 4)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    
    return parser.parse_args()

def main():
    """Run the main workflow."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Step 1: Data Collection
    logger.info(f"Fetching data for {args.ticker} over {args.period}")
    stock_data = fetch_stock_data(args.ticker, period=args.period)
    
    if stock_data.empty:
        logger.error(f"Failed to fetch data for {args.ticker}")
        return
    
    # Step 2: Data Cleaning and Feature Extraction
    logger.info("Cleaning and preparing data")
    cleaned_data = clean_stock_data(stock_data)
    
    # Plot and save raw stock data
    fig = plot_stock_prices(stock_data, ticker=args.ticker, show_volume=True)
    fig.savefig(os.path.join(args.output, f"{args.ticker}_prices.png"))
    plt.close(fig)
    
    # Extract features
    enriched_data = extract_features(cleaned_data)
    
    # Step 3: Prepare Data for Models
    logger.info(f"Preparing sequences with window size {args.window}")
    
    # Prepare data for LSTM
    X_train, y_train, X_test, y_test, scaler = prepare_data_for_lstm(
        enriched_data,
        window_size=args.window,
        test_size=0.2
    )
    
    # Save training and testing data shapes
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")
    
    # Dictionary to store model predictions
    model_predictions = {}
    
    # Step 4: Train Classical LSTM Model
    logger.info("Training classical LSTM model")
    lstm_model = LSTMModel(lstm_units=[100, 50])
    lstm_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train LSTM model
    lstm_history = lstm_model.train(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=32,
        save_path=os.path.join(args.output, "lstm_model.h5")
    )
    
    # Plot and save training history
    fig = lstm_model.plot_history()
    fig.savefig(os.path.join(args.output, "lstm_training_history.png"))
    plt.close(fig)
    
    # Make LSTM predictions
    logger.info("Making LSTM predictions")
    lstm_predictions = lstm_model.predict(X_test)
    lstm_pred_orig = inverse_transform_predictions(lstm_predictions, scaler)
    model_predictions['LSTM'] = lstm_pred_orig
    
    # Step 5: Train Quantum Models if requested
    if args.quantum_model in ["variational", "both"]:
        logger.info("Training variational quantum model")
        # Use a smaller subset of data for quantum model to speed up training
        train_subset_size = min(200, len(X_train))
        X_train_subset = X_train[:train_subset_size]
        y_train_subset = y_train[:train_subset_size]
        
        quantum_model = QuantumModel(
            n_qubits=args.n_qubits,
            n_layers=2,
            learning_rate=0.01
        )
        quantum_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Train with a smaller number of epochs for demonstration
        quantum_epochs = min(args.epochs, 10)
        quantum_model.train(
            X_train_subset, 
            y_train_subset,
            epochs=quantum_epochs,
            batch_size=16,
            save_path=os.path.join(args.output, "quantum_model.h5")
        )
        
        # Plot and save quantum training history
        fig = quantum_model.plot_history()
        fig.savefig(os.path.join(args.output, "quantum_training_history.png"))
        plt.close(fig)
        
        # Make predictions with variational quantum model
        logger.info("Making variational quantum predictions")
        quantum_predictions = quantum_model.predict(X_test)
        quantum_pred_orig = inverse_transform_predictions(quantum_predictions, scaler)
        model_predictions['Quantum Variational'] = quantum_pred_orig
    
    if args.quantum_model in ["kernel", "both"]:
        logger.info("Training quantum kernel model")
        # Use an even smaller subset for the kernel model (it's computationally expensive)
        kernel_train_size = min(100, len(X_train))
        X_train_kernel = X_train[:kernel_train_size]
        y_train_kernel = y_train[:kernel_train_size]
        
        kernel_model = QuantumKernelModel(
            n_qubits=args.n_qubits,
            feature_map="ZZFeatureMap",
            C=1.0,
            epsilon=0.1
        )
        
        kernel_model.train(X_train_kernel, y_train_kernel)
        
        # Save the kernel model
        kernel_model.save(os.path.join(args.output, "quantum_kernel_model.pkl"))
        
        # Make predictions with quantum kernel model
        logger.info("Making quantum kernel predictions")
        # Use a small subset of test data for kernel predictions (faster)
        kernel_test_size = min(50, len(X_test))
        X_test_kernel = X_test[:kernel_test_size]
        y_test_kernel = y_test[:kernel_test_size]
        
        kernel_predictions = kernel_model.predict(X_test_kernel)
        kernel_pred_orig = inverse_transform_predictions(kernel_predictions, scaler)
        
        # Only use this subset for evaluation
        if kernel_test_size < len(X_test):
            y_test_kernel_orig = inverse_transform_predictions(y_test[:kernel_test_size], scaler)
            model_predictions['Quantum Kernel'] = kernel_pred_orig
            
            # Separate evaluation for kernel model
            kernel_metrics = calculate_metrics(y_test_kernel_orig, kernel_pred_orig)
            logger.info(f"Quantum Kernel Model RMSE: {kernel_metrics['rmse']:.4f}")
            
            # Plot kernel predictions separately
            fig = plot_predictions(
                y_test_kernel_orig,
                kernel_pred_orig,
                ticker=f"{args.ticker} - Quantum Kernel"
            )
            fig.savefig(os.path.join(args.output, f"{args.ticker}_quantum_kernel_predictions.png"))
            plt.close(fig)
    
    # Get actual values in original scale
    y_test_dates = enriched_data.index[-len(y_test):]
    y_test_orig = inverse_transform_predictions(y_test, scaler)
    
    # Step 7: Evaluate and compare models
    logger.info("Evaluating models")
    
    # Calculate comparison metrics
    comparison = compare_models(y_test_orig, model_predictions)
    logger.info("\nModel Comparison:")
    logger.info(comparison)
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(args.output, "model_comparison.csv"))
    
    # Create comparison chart
    fig = plot_model_comparison(
        {model: comp['rmse'] for model, comp in comparison.to_dict('index').items()},
        metric='rmse'
    )
    fig.savefig(os.path.join(args.output, "model_comparison_rmse.png"))
    plt.close(fig)
    
    # Create comparison chart for directional accuracy
    fig = plot_model_comparison(
        {model: comp['directional_accuracy'] for model, comp in comparison.to_dict('index').items()},
        metric='directional_accuracy'
    )
    fig.savefig(os.path.join(args.output, "model_comparison_direction.png"))
    plt.close(fig)
    
    # Step 8: Create and save prediction plots
    logger.info("Creating visualization plots")
    
    # LSTM predictions
    fig = plot_predictions(
        y_test_orig,
        lstm_pred_orig,
        dates=y_test_dates,
        ticker=f"{args.ticker} - LSTM"
    )
    fig.savefig(os.path.join(args.output, f"{args.ticker}_lstm_predictions.png"))
    plt.close(fig)
    
    # Quantum variational predictions if available
    if 'Quantum Variational' in model_predictions:
        fig = plot_predictions(
            y_test_orig,
            model_predictions['Quantum Variational'],
            dates=y_test_dates,
            ticker=f"{args.ticker} - Quantum Variational"
        )
        fig.savefig(os.path.join(args.output, f"{args.ticker}_quantum_variational_predictions.png"))
        plt.close(fig)
    
    # Step 9: Evaluate trading strategies
    logger.info("Evaluating trading strategies")
    
    # LSTM trading strategy
    lstm_return, lstm_buyhold, lstm_returns, buyhold_returns = trading_strategy_evaluation(
        y_test_orig,
        lstm_pred_orig
    )
    
    logger.info(f"LSTM Strategy Return: {lstm_return:.2%}")
    logger.info(f"Buy & Hold Return: {lstm_buyhold:.2%}")
    
    # Plot trading strategy results
    fig = plot_trading_strategy(lstm_returns, buyhold_returns, dates=y_test_dates)
    fig.savefig(os.path.join(args.output, f"{args.ticker}_lstm_trading.png"))
    plt.close(fig)
    
    # Quantum trading strategy if available
    if 'Quantum Variational' in model_predictions:
        quantum_return, quantum_buyhold, quantum_returns, _ = trading_strategy_evaluation(
            y_test_orig,
            model_predictions['Quantum Variational']
        )
        
        logger.info(f"Quantum Variational Strategy Return: {quantum_return:.2%}")
        
        # Plot quantum trading strategy results
        fig = plot_trading_strategy(quantum_returns, buyhold_returns, dates=y_test_dates)
        fig.savefig(os.path.join(args.output, f"{args.ticker}_quantum_trading.png"))
        plt.close(fig)
    
    logger.info("Workflow completed successfully")
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 