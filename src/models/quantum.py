"""Quantum machine learning models for stock prediction.

This module implements quantum machine learning models using PennyLane.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt
import os
import json
import pennylane as qml
from pennylane import numpy as pnp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuantumCircuit:
    """Quantum circuit for stock prediction."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        device: str = "default.qubit"
    ):
        """Initialize quantum circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            device: PennyLane device to use
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        self.dev = qml.device(device, wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Create the quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface="tf")
        
        logger.info(f"Initialized quantum circuit with {n_qubits} qubits and {n_layers} layers")
        
    def _circuit(self, inputs, weights):
        """Quantum circuit definition.
        
        Args:
            inputs: Input data encoded into the circuit
            weights: Trainable weights for the circuit
        
        Returns:
            Expectation values of observables
        """
        # Encode the inputs into the circuit
        self._encode_inputs(inputs)
        
        # Variational circuit
        self._variational_circuit(weights)
        
        # Return expectation values
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]
    
    def _encode_inputs(self, inputs):
        """Encode classical inputs into the quantum circuit.
        
        Args:
            inputs: Input values to encode
        """
        # Normalize and scale inputs
        scaled_inputs = pnp.array(inputs) * pnp.pi
        
        # Angle encoding
        for i in range(min(len(scaled_inputs), self.n_qubits)):
            qml.RY(scaled_inputs[i], wires=i)
    
    def _variational_circuit(self, weights):
        """Apply the variational part of the circuit.
        
        Args:
            weights: Trainable weights
        """
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            
            # Entanglement
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            
            # Connect the last qubit to the first to form a ring
            if self.n_qubits > 1:
                qml.CNOT(wires=[self.n_qubits - 1, 0])


class QuantumModel:
    """Quantum model for stock price prediction using PennyLane."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        learning_rate: float = 0.01,
        n_features: int = 1,
        device: str = "default.qubit"
    ):
        """Initialize quantum model.
        
        Args:
            n_qubits: Number of qubits to use
            n_layers: Number of variational layers
            learning_rate: Learning rate for the optimizer
            n_features: Number of input features
            device: PennyLane device to use
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_features = n_features
        self.device = device
        self.model = None
        self.history = None
        self.qcircuit = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers, device=device)
        
        logger.info(f"Initialized quantum model with {n_qubits} qubits, {n_layers} layers")
    
    def _feature_dimension_reduction(self, inputs, n_features_to_encode):
        """Reduce feature dimensions to match the number of qubits.
        
        Args:
            inputs: Input data with shape (batch_size, time_steps, features)
            n_features_to_encode: Number of features to encode in the quantum circuit
        
        Returns:
            Reduced feature array with shape (batch_size, n_features_to_encode)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Flatten the inputs for each batch item
        flattened = tf.reshape(inputs, [batch_size, -1])
        
        # If the flattened dimension is larger than what we need, use a linear layer to reduce
        if tf.shape(flattened)[1] > n_features_to_encode:
            reduction_layer = tf.keras.layers.Dense(n_features_to_encode, activation='tanh')
            reduced = reduction_layer(flattened)
        else:
            # Pad with zeros if needed
            padding_size = n_features_to_encode - tf.shape(flattened)[1]
            if padding_size > 0:
                reduced = tf.pad(flattened, [[0, 0], [0, padding_size]])
            else:
                reduced = flattened
                
        # Normalize to [-1, 1] range for angle encoding
        reduced = tf.tanh(reduced)
        
        return reduced
    
    def build_model(self, input_shape):
        """Build the hybrid quantum-classical model.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
        
        Returns:
            Keras model
        """
        # Create model inputs
        inputs = Input(shape=input_shape, name="inputs")
        
        # Initial classical preprocessing
        x = tf.keras.layers.LSTM(32, return_sequences=False)(inputs)
        x = Dropout(0.2)(x)
        
        # Dimension reduction to match number of qubits
        reduced_features = self._feature_dimension_reduction(inputs, self.n_qubits)
        
        # Initialize variational circuit weights
        weight_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        weights = tf.Variable(
            initial_value=weight_init(
                shape=self.qcircuit.weight_shapes["weights"],
                dtype=tf.float32
            ),
            trainable=True,
            name="quantum_weights"
        )
        
        # Quantum circuit layer implemented as a Lambda layer
        quantum_layer = tf.keras.layers.Lambda(
            lambda x: self.qcircuit.qnode(x, weights),
            name="quantum_layer"
        )
        quantum_outputs = quantum_layer(reduced_features)
        
        # Final classical post-processing
        outputs = Dense(16, activation='relu')(quantum_outputs)
        outputs = Dense(1, activation='linear', name="predictions")(outputs)
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        self.model = model
        logger.info(f"Built quantum model with input shape {input_shape}")
        
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
        """Train the quantum model.
        
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
        logger.info(f"Training quantum model with {len(X_train)} samples, {epochs} epochs, {batch_size} batch size")
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
        
        # For Keras models with custom quantum layers, save the weights separately
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model weights
        weights_path = path.replace('.json', '.h5')
        self.model.save_weights(weights_path)
        
        # Save model metadata and history
        metadata = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "n_features": self.n_features,
            "device": self.device,
            "input_shape": self.model.input_shape[1:],
            "weights_path": os.path.basename(weights_path),
            "history": self.history.history if self.history is not None else None
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f)
        
        logger.info(f"Model weights saved to {weights_path}")
        logger.info(f"Model metadata saved to {path}")
    
    def load(self, path: str) -> None:
        """Load a pre-trained model.
        
        Args:
            path: Path to the saved model metadata
        """
        # Load model metadata
        with open(path, 'r') as f:
            metadata = json.load(f)
        
        # Set model properties
        self.n_qubits = metadata["n_qubits"]
        self.n_layers = metadata["n_layers"]
        self.learning_rate = metadata["learning_rate"]
        self.n_features = metadata["n_features"]
        self.device = metadata["device"]
        
        # Initialize quantum circuit with loaded parameters
        self.qcircuit = QuantumCircuit(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            device=self.device
        )
        
        # Build the model architecture
        self.build_model(metadata["input_shape"])
        
        # Load model weights
        weights_path = os.path.join(os.path.dirname(path), metadata["weights_path"])
        self.model.load_weights(weights_path)
        
        # Load history if available
        if metadata["history"] is not None:
            self.history = type('obj', (object,), {'history': metadata["history"]})
        
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
        ax.set_title('Quantum Model Training and Validation Loss')
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


class QuantumKernelModel:
    """Quantum kernel-based model for stock prediction.
    
    This model uses a quantum kernel for support vector regression.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        feature_map: str = "ZZFeatureMap",
        device: str = "default.qubit",
        C: float = 1.0,
        epsilon: float = 0.1
    ):
        """Initialize quantum kernel model.
        
        Args:
            n_qubits: Number of qubits
            feature_map: Type of quantum feature map ('ZZFeatureMap' or 'AngleEmbedding')
            device: PennyLane device to use
            C: Regularization parameter for SVR
            epsilon: Epsilon parameter for SVR
        """
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map
        self.device = device
        self.C = C
        self.epsilon = epsilon
        self.dev = qml.device(device, wires=n_qubits)
        self.svr = None
        self.trained = False
        self.feature_scaler = None
        
        # Create the quantum kernel
        self._create_kernel()
        
        logger.info(f"Initialized quantum kernel model with {n_qubits} qubits and {feature_map} feature map")
    
    def _create_kernel(self):
        """Create the quantum kernel based on the specified feature map."""
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            """Quantum circuit for kernel evaluation.
            
            Args:
                x1: First data point
                x2: Second data point
            
            Returns:
                Kernel value (inner product in Hilbert space)
            """
            # Apply feature map to the first data point
            self._apply_feature_map(x1)
            
            # Apply inverse operations to create Hermitian conjugate
            qml.adjoint(self._apply_feature_map)(x2)
            
            # Return kernel value (overlap between states)
            return qml.probs(wires=range(self.n_qubits))[0]
        
        self.kernel_circuit = kernel_circuit
    
    def _apply_feature_map(self, x):
        """Apply the selected feature map.
        
        Args:
            x: Input data point
        """
        # Ensure x is properly sized
        x = np.asarray(x)
        x = x[:self.n_qubits] if len(x) > self.n_qubits else x
        
        if self.feature_map_type == "ZZFeatureMap":
            # ZZFeatureMap: Apply Hadamards followed by rotations and entangling
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # First rotation layer
            for i in range(self.n_qubits):
                if i < len(x):
                    qml.RZ(x[i], wires=i)
            
            # Entanglement layer
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    if i < len(x) and j < len(x):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(x[i] * x[j], wires=j)
                        qml.CNOT(wires=[i, j])
            
        elif self.feature_map_type == "AngleEmbedding":
            # Simple angle embedding
            for i in range(min(self.n_qubits, len(x))):
                qml.RY(x[i], wires=i)
                qml.RZ(x[i], wires=i)
            
            # Add entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        else:
            raise ValueError(f"Unknown feature map type: {self.feature_map_type}")
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix between data points.
        
        Args:
            X1: First set of data points
            X2: Second set of data points (if None, use X1)
        
        Returns:
            Kernel matrix
        """
        X2 = X1 if X2 is None else X2
        
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel_circuit(X1[i], X2[j])
        
        return K
    
    def _prepare_data(self, X):
        """Prepare data for quantum kernel.
        
        Args:
            X: Input data with shape (n_samples, time_steps, features)
        
        Returns:
            Flattened and reduced data suitable for quantum processing
        """
        # Flatten the time series data (each sample is a flattened sequence)
        n_samples = X.shape[0]
        flattened_X = X.reshape(n_samples, -1)
        
        # Reduce dimension to match the number of qubits using PCA if needed
        if flattened_X.shape[1] > self.n_qubits:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                flattened_X = self.feature_scaler.fit_transform(flattened_X)
            else:
                flattened_X = self.feature_scaler.transform(flattened_X)
            
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=self.n_qubits)
                reduced_X = self.pca.fit_transform(flattened_X)
            else:
                reduced_X = self.pca.transform(flattened_X)
            
            # Normalize to [-pi, pi] for angle encoding
            reduced_X = np.clip(reduced_X, -1, 1) * np.pi
        else:
            # If already smaller than n_qubits, just normalize
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
                reduced_X = self.feature_scaler.fit_transform(flattened_X)
            else:
                reduced_X = self.feature_scaler.transform(flattened_X)
            
            reduced_X = np.clip(reduced_X, -1, 1) * np.pi
        
        return reduced_X
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the quantum kernel model.
        
        Args:
            X_train: Training data
            y_train: Target values
        """
        from sklearn.svm import SVR
        
        logger.info(f"Training quantum kernel model with {len(X_train)} samples")
        
        # Prepare data for quantum kernel
        X_train_prepared = self._prepare_data(X_train)
        
        # Compute the training kernel matrix
        K_train = self._compute_kernel_matrix(X_train_prepared)
        
        # Create and train the SVR with the precomputed kernel
        self.svr = SVR(kernel='precomputed', C=self.C, epsilon=self.epsilon)
        self.svr.fit(K_train, y_train)
        
        self.X_train_prepared = X_train_prepared  # Store for prediction
        self.trained = True
        
        logger.info("Quantum kernel model training completed")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using the trained quantum kernel model.
        
        Args:
            X_test: Test data
        
        Returns:
            Predicted values
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making quantum kernel predictions on {len(X_test)} samples")
        
        # Prepare test data
        X_test_prepared = self._prepare_data(X_test)
        
        # Compute the kernel matrix between test and training data
        K_test = self._compute_kernel_matrix(X_test_prepared, self.X_train_prepared)
        
        # Make predictions using SVR
        predictions = self.svr.predict(K_test)
        
        return predictions.reshape(-1, 1)
    
    def save(self, path: str) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        import pickle
        
        # Save metadata and SVR model
        model_data = {
            "n_qubits": self.n_qubits,
            "feature_map_type": self.feature_map_type,
            "device": self.device,
            "C": self.C,
            "epsilon": self.epsilon,
            "svr": self.svr,
            "feature_scaler": self.feature_scaler
        }
        
        if hasattr(self, 'pca'):
            model_data["pca"] = self.pca
        
        # We need to store the training data for prediction
        model_data["X_train_prepared"] = self.X_train_prepared
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Quantum kernel model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load a pre-trained model.
        
        Args:
            path: Path to the saved model
        """
        import pickle
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_qubits = model_data["n_qubits"]
        self.feature_map_type = model_data["feature_map_type"]
        self.device = model_data["device"]
        self.C = model_data["C"]
        self.epsilon = model_data["epsilon"]
        self.svr = model_data["svr"]
        self.feature_scaler = model_data["feature_scaler"]
        
        if "pca" in model_data:
            self.pca = model_data["pca"]
        
        self.X_train_prepared = model_data["X_train_prepared"]
        self.trained = True
        
        # Recreate the device and kernel
        self.dev = qml.device(self.device, wires=self.n_qubits)
        self._create_kernel()
        
        logger.info(f"Quantum kernel model loaded from {path}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: True values
        
        Returns:
            Mean squared error
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        
        logger.info(f"Quantum kernel model test MSE: {mse:.6f}")
        
        return mse


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy data
    X_train = np.random.random((10, 60, 1))
    y_train = np.random.random((10,))
    
    # Create and train model (with small dataset for demonstration)
    model = QuantumModel(n_qubits=4, n_layers=1)
    model.build_model(input_shape=(60, 1))
    model.train(X_train, y_train, epochs=2, verbose=1)
    
    # Make predictions
    X_test = np.random.random((5, 60, 1))
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}") 