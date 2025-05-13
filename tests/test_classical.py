"""Unit tests for the classical models module."""

import unittest
import numpy as np
import tensorflow as tf
import os
import tempfile
from src.models.classical import LSTMModel, create_multi_step_lstm

class TestLSTMModel(unittest.TestCase):
    """Tests for the LSTM model class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.window_size = 10
        self.features = 1
        self.X_train = np.random.random((50, self.window_size, self.features))
        self.y_train = np.random.random((50,))
        self.X_test = np.random.random((10, self.window_size, self.features))
        self.y_test = np.random.random((10,))
    
    def test_init(self):
        """Test model initialization."""
        model = LSTMModel(lstm_units=[32, 16])
        
        self.assertEqual(model.lstm_units, [32, 16])
        self.assertEqual(model.dropout_rate, 0.2)  # Default value
        self.assertIsNone(model.model)
        self.assertIsNone(model.history)
    
    def test_build_model(self):
        """Test building the model architecture."""
        model = LSTMModel()
        
        # Test building with input shape provided at initialization
        model.input_shape = (self.window_size, self.features)
        built_model = model.build_model()
        
        self.assertIsNotNone(model.model)
        self.assertIsInstance(built_model, tf.keras.Sequential)
        
        # Check model structure
        layers = model.model.layers
        lstm_layers = [layer for layer in layers if isinstance(layer, tf.keras.layers.LSTM)]
        dropout_layers = [layer for layer in layers if isinstance(layer, tf.keras.layers.Dropout)]
        dense_layers = [layer for layer in layers if isinstance(layer, tf.keras.layers.Dense)]
        
        self.assertEqual(len(lstm_layers), len(model.lstm_units))
        self.assertEqual(len(dropout_layers), len(model.lstm_units))
        self.assertEqual(len(dense_layers), 1)  # One output layer
        
        # Test building with input shape provided as parameter
        model2 = LSTMModel()
        built_model2 = model2.build_model(input_shape=(self.window_size, self.features))
        
        self.assertIsNotNone(model2.model)
        
        # Test validation error when no input shape is provided
        model3 = LSTMModel()
        with self.assertRaises(ValueError):
            model3.build_model()
    
    def test_train(self):
        """Test model training."""
        # Create a model and train it
        model = LSTMModel(lstm_units=[10])
        history = model.train(
            self.X_train, 
            self.y_train, 
            validation_split=0.2,
            batch_size=16,
            epochs=2  # Use a small number of epochs for testing
        )
        
        # Check if training history is returned
        self.assertIsInstance(history, tf.keras.callbacks.History)
        self.assertEqual(model.history, history)
        
        # Check if history contains expected metrics
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)
        
        # Check if number of epochs matches
        self.assertEqual(len(history.history['loss']), 2)
    
    def test_predict(self):
        """Test model prediction."""
        # Create and train a model
        model = LSTMModel(lstm_units=[10])
        model.train(
            self.X_train, 
            self.y_train, 
            epochs=1
        )
        
        # Test prediction
        predictions = model.predict(self.X_test)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (len(self.X_test), 1))
        
        # Test error when model is not trained
        model_untrained = LSTMModel()
        with self.assertRaises(ValueError):
            model_untrained.predict(self.X_test)
    
    def test_save_load(self):
        """Test saving and loading the model."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.h5")
            
            # Create and train a model
            model = LSTMModel(lstm_units=[10])
            model.train(
                self.X_train, 
                self.y_train, 
                epochs=1
            )
            
            # Save the model
            model.save(model_path)
            
            # Check if the file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Load the model into a new instance
            model2 = LSTMModel()
            model2.load(model_path)
            
            # Make predictions with both models
            preds1 = model.predict(self.X_test)
            preds2 = model2.predict(self.X_test)
            
            # Check if predictions are the same
            np.testing.assert_allclose(preds1, preds2, rtol=1e-5)
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Create and train a model
        model = LSTMModel(lstm_units=[10])
        model.train(
            self.X_train, 
            self.y_train, 
            epochs=1
        )
        
        # Evaluate the model
        loss = model.evaluate(self.X_test, self.y_test)
        
        # Check if loss is a float
        self.assertIsInstance(loss, float)
    
    def test_multi_step_lstm(self):
        """Test multi-step LSTM creation function."""
        input_shape = (self.window_size, self.features)
        output_steps = 5
        
        # Create multi-step LSTM model
        model = create_multi_step_lstm(
            input_shape=input_shape,
            output_steps=output_steps
        )
        
        # Check if correct model type is returned
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input and output shapes
        self.assertEqual(model.input_shape[1:], input_shape)
        self.assertEqual(model.output_shape[1:], (output_steps,))
        
        # Test with custom parameters
        custom_model = create_multi_step_lstm(
            input_shape=input_shape,
            output_steps=3,
            lstm_units=[20, 10],
            dropout_rate=0.3
        )
        
        self.assertEqual(custom_model.output_shape[1:], (3,))
        
        # Check layer structure
        layers = custom_model.layers
        lstm_layers = [layer for layer in layers if isinstance(layer, tf.keras.layers.LSTM)]
        self.assertEqual(len(lstm_layers), 2)  # Two LSTM layers


if __name__ == '__main__':
    unittest.main() 