"""Classical machine learning models for stock prediction.

This module implements classical machine learning models for stock price prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import logging
from typing import Tuple, Optional, Union, List
from datetime import datetime
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for stock price prediction."""
    
    def __init__(
        self, 
        input_shape: Optional[Tuple[int, int]] = None,
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        loss: str = 'mean_squared_error',
        optimizer: str = 'adam'
    ):
        """Initialize LSTM model.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate to prevent overfitting
            loss: Loss function
            optimizer: Optimizer
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.optimizer = optimizer
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: Optional[Tuple[int, int]] = None) -> None:
        """Build the LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
        """
        if input_shape is not None:
            self.input_shape = input_shape
        
        if self.input_shape is None:
            raise ValueError("input_shape must be provided either during initialization or when calling build_model")
        
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1  # True for all layers except the last one
            
            if i == 0:
                # First layer needs input_shape
                model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=self.input_shape))
            else:
                model.add(LSTM(units=units, return_sequences=return_sequences))
            
            # Add dropout after each LSTM layer
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
        self.model = model
        
        logger.info(f"Model built with input shape {self.input_shape}")
        logger.info(f"LSTM layers: {self.lstm_units}, Dropout rate: {self.dropout_rate}")
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        validation_split: float = 0.1,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        verbose: int = 1,
        save_path: Optional[str] = None
    ) -> tf.keras.callbacks.History:
        """Train the LSTM model.
        
        Args:
            X_train: Training data
            y_train: Target values
            validation_split: Fraction of training data to use for validation
            batch_size: Batch size for training
            epochs: Number of epochs for training
            patience: Patience for early stopping
            verbose: Verbosity level
            save_path: Path to save the best model
        
        Returns:
            Training history
        """
        # If model doesn't exist, build it
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Define callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint if save_path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = ModelCheckpoint(
                save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            callbacks.append(checkpoint)
        
        # Train the model
        logger.info(f"Training LSTM model with {len(X_train)} samples, {epochs} epochs, {batch_size} batch size")
        history = self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        
        logger.info(f"Model training completed. Final loss: {history.history['loss'][-1]:.6f}, "
                   f"Val loss: {history.history['val_loss'][-1]:.6f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Input data for prediction
        
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions on {len(X)} samples")
        predictions = self.model.predict(X)
        
        return predictions
    
    def save(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load a pre-trained model.
        
        Args:
            path: Path to the saved model
        """
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")
    
    def plot_history(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot training history.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.history.history['loss'], label='Training Loss')
        ax.plot(self.history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        return fig
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: True values
        
        Returns:
            Loss value
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test loss: {loss:.6f}")
        
        return loss

def create_multi_step_lstm(
    input_shape: Tuple[int, int],
    output_steps: int = 5,
    lstm_units: List[int] = [64, 64],
    dropout_rate: float = 0.2
) -> tf.keras.Model:
    """Create a multi-step LSTM model that predicts multiple future time steps.
    
    Args:
        input_shape: Shape of input data (time_steps, features)
        output_steps: Number of future steps to predict
        lstm_units: List of units for each LSTM layer
        dropout_rate: Dropout rate to prevent overfitting
    
    Returns:
        Multi-step LSTM model
    """
    model = Sequential()
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1  # True for all layers except the last one
        
        if i == 0:
            # First layer needs input_shape
            model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, return_sequences=return_sequences))
        
        # Add dropout after each LSTM layer
        model.add(Dropout(dropout_rate))
    
    # Output layer with output_steps units for multi-step prediction
    model.add(Dense(units=output_steps))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    logger.info(f"Multi-step LSTM model created with {output_steps} output steps")
    
    return model

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy data
    X_train = np.random.random((100, 60, 1))
    y_train = np.random.random((100,))
    
    # Create and train model
    model = LSTMModel()
    model.build_model(input_shape=(60, 1))
    model.train(X_train, y_train, epochs=5)
    
    # Make predictions
    predictions = model.predict(X_train[:5])
    print(f"Predictions shape: {predictions.shape}") 