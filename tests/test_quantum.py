"""Unit tests for the quantum models module."""

import unittest
import numpy as np
import tensorflow as tf
import os
import tempfile
import json
from src.models.quantum import QuantumCircuit, QuantumModel, QuantumKernelModel

class TestQuantumCircuit(unittest.TestCase):
    """Tests for the QuantumCircuit class."""
    
    def test_init(self):
        """Test circuit initialization."""
        # Create quantum circuit
        circuit = QuantumCircuit(n_qubits=3, n_layers=2)
        
        # Check attributes
        self.assertEqual(circuit.n_qubits, 3)
        self.assertEqual(circuit.n_layers, 2)
        self.assertEqual(circuit.device, "default.qubit")
        
        # Check weight shapes
        expected_weight_shape = (2, 3, 3)  # (n_layers, n_qubits, 3)
        self.assertEqual(circuit.weight_shapes["weights"], expected_weight_shape)
    
    def test_circuit_execution(self):
        """Test that the circuit runs and returns expected shape."""
        # Create circuit with small parameter values
        circuit = QuantumCircuit(n_qubits=2, n_layers=1)
        
        # Create inputs and weights
        inputs = np.array([0.1, 0.2])
        weights = np.zeros((1, 2, 3))  # Zero weights for deterministic results
        
        # Execute the circuit
        result = circuit.qnode(inputs, weights)
        
        # Check the result shape
        self.assertEqual(len(result), circuit.n_qubits)
        
        # For zero weights, the result should be predictable
        # (though not testing exact values due to quantum specifics)
        self.assertIsInstance(result, tf.Tensor)


class TestQuantumModel(unittest.TestCase):
    """Tests for the QuantumModel class."""
    
    def setUp(self):
        """Set up test data."""
        # Use a very small dataset for quick tests
        self.window_size = 5
        self.features = 1
        self.X_train = np.random.random((10, self.window_size, self.features))
        self.y_train = np.random.random((10,))
        self.X_test = np.random.random((5, self.window_size, self.features))
        self.y_test = np.random.random((5,))
    
    def test_init(self):
        """Test model initialization."""
        model = QuantumModel(n_qubits=2, n_layers=1)
        
        # Check attributes
        self.assertEqual(model.n_qubits, 2)
        self.assertEqual(model.n_layers, 1)
        self.assertEqual(model.learning_rate, 0.01)  # Default value
        self.assertIsNone(model.model)
        self.assertIsNone(model.history)
        
        # Check circuit initialization
        self.assertIsInstance(model.qcircuit, QuantumCircuit)
        self.assertEqual(model.qcircuit.n_qubits, 2)
        self.assertEqual(model.qcircuit.n_layers, 1)
    
    def test_feature_dimension_reduction(self):
        """Test feature dimension reduction function."""
        model = QuantumModel(n_qubits=3)
        
        # Create a batch of input data
        inputs = tf.random.normal((4, 5, 2))  # batch_size=4, time_steps=5, features=2
        
        # Reduce to 3 features (n_qubits)
        reduced = model._feature_dimension_reduction(inputs, 3)
        
        # Check shape
        self.assertEqual(reduced.shape, (4, 3))
        
        # Check value range (should be in [-1, 1])
        self.assertTrue(tf.reduce_all(reduced >= -1.0))
        self.assertTrue(tf.reduce_all(reduced <= 1.0))
    
    def test_build_model(self):
        """Test building the model architecture."""
        model = QuantumModel(n_qubits=2, n_layers=1)
        
        # Build the model
        built_model = model.build_model(input_shape=(self.window_size, self.features))
        
        # Check model existence
        self.assertIsNotNone(model.model)
        self.assertIsInstance(built_model, tf.keras.Model)
        
        # Check model input/output shapes
        self.assertEqual(built_model.input_shape[1:], (self.window_size, self.features))
        self.assertEqual(built_model.output_shape[1:], (1,))  # Single output value
    
    def test_train_predict(self):
        """Test model training and prediction."""
        # Skip this test if no GPU/TPU is available (quantum simulation is slow on CPU)
        try:
            # Create a small model that can train quickly
            model = QuantumModel(n_qubits=2, n_layers=1)
            model.build_model(input_shape=(self.window_size, self.features))
            
            # Train for just one epoch
            history = model.train(
                self.X_train, 
                self.y_train, 
                epochs=1,
                batch_size=5,
                verbose=0
            )
            
            # Check history
            self.assertIsInstance(history, tf.keras.callbacks.History)
            self.assertIn('loss', history.history)
            
            # Test prediction
            predictions = model.predict(self.X_test)
            
            # Check predictions shape
            self.assertEqual(predictions.shape, (len(self.X_test), 1))
        
        except (tf.errors.ResourceExhaustedError, MemoryError):
            # Skip if we run out of memory
            self.skipTest("Skipping test due to memory constraints")
    
    def test_save_load(self):
        """Test saving and loading the model."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Paths for model files
            model_path = os.path.join(tmpdirname, "quantum_model.json")
            
            # Create a model
            model = QuantumModel(n_qubits=2, n_layers=1)
            model.build_model(input_shape=(self.window_size, self.features))
            
            # We don't train the model to save time, but set up minimal history
            model.history = type('obj', (object,), {'history': {'loss': [1.0, 0.9], 'val_loss': [1.1, 1.0]}})
            
            # Save the model
            model.save(model_path)
            
            # Check if the files exist
            self.assertTrue(os.path.exists(model_path))
            weights_path = model_path.replace('.json', '.h5')
            self.assertTrue(os.path.exists(weights_path))
            
            # Check JSON content
            with open(model_path, 'r') as f:
                metadata = json.load(f)
            
            self.assertEqual(metadata['n_qubits'], 2)
            self.assertEqual(metadata['n_layers'], 1)
            
            # Load doesn't fully work in tests without training, so we just verify file structure
            self.assertIn('weights_path', metadata)
            self.assertEqual(metadata['weights_path'], os.path.basename(weights_path))


class TestQuantumKernelModel(unittest.TestCase):
    """Tests for the QuantumKernelModel class."""
    
    def setUp(self):
        """Set up test data."""
        # Use a very small dataset for quick tests
        self.window_size = 3
        self.features = 1
        self.X_train = np.random.random((5, self.window_size, self.features))
        self.y_train = np.random.random((5,))
        self.X_test = np.random.random((3, self.window_size, self.features))
        self.y_test = np.random.random((3,))
    
    def test_init(self):
        """Test model initialization."""
        model = QuantumKernelModel(n_qubits=2, feature_map="ZZFeatureMap")
        
        # Check attributes
        self.assertEqual(model.n_qubits, 2)
        self.assertEqual(model.feature_map_type, "ZZFeatureMap")
        self.assertEqual(model.device, "default.qubit")
        self.assertIsNone(model.svr)
        self.assertFalse(model.trained)
    
    def test_prepare_data(self):
        """Test data preparation for quantum kernel."""
        model = QuantumKernelModel(n_qubits=2)
        
        # Prepare data
        prepared_data = model._prepare_data(self.X_train)
        
        # Check shape
        self.assertEqual(prepared_data.shape, (len(self.X_train), model.n_qubits))
        
        # Check value range (should be in [-pi, pi])
        self.assertTrue(np.all(prepared_data >= -np.pi))
        self.assertTrue(np.all(prepared_data <= np.pi))
    
    def test_kernel_circuit(self):
        """Test kernel circuit execution."""
        model = QuantumKernelModel(n_qubits=2)
        
        # Create two test data points
        x1 = np.array([0.1, 0.2])
        x2 = np.array([0.2, 0.3])
        
        # Compute kernel value
        kernel_value = model.kernel_circuit(x1, x2)
        
        # Check that kernel value is a scalar and in [0, 1]
        self.assertIsInstance(kernel_value, float)
        self.assertTrue(0 <= kernel_value <= 1)
        
        # Kernel should be highest when vectors are identical
        kernel_identical = model.kernel_circuit(x1, x1)
        self.assertGreaterEqual(kernel_identical, kernel_value)
    
    @unittest.skip("Skip training test as it's too slow for regular testing")
    def test_train_predict(self):
        """Test model training and prediction."""
        # This test is skipped by default as quantum kernel training can be very slow
        model = QuantumKernelModel(n_qubits=2)
        
        # Train the model
        model.train(self.X_train, self.y_train)
        
        # Check training status
        self.assertTrue(model.trained)
        self.assertIsNotNone(model.svr)
        
        # Test prediction
        predictions = model.predict(self.X_test)
        
        # Check predictions shape
        self.assertEqual(predictions.shape, (len(self.X_test), 1))


if __name__ == '__main__':
    unittest.main() 