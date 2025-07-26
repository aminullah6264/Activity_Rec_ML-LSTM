"""LSTM model for activity recognition using temporal features."""

import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import scipy.io as sio
from tqdm import tqdm

from .config import config

logger = logging.getLogger(__name__)


class LSTMModel:
    """Multi-layer LSTM model for activity recognition."""
    
    def __init__(self, config_override: dict = None):
        """
        Initialize the LSTM model.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = config
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.model = None
        self.class_names = []
        self.history = None
        
        # Setup GPU if available
        self._setup_gpu()
    
    def _setup_gpu(self) -> None:
        """Setup GPU configuration for TensorFlow."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        else:
            logger.info("No GPU found, using CPU")
    
    def build_model(self) -> tf.keras.Model:
        """
        Build the multi-layer LSTM model.
        
        Returns:
            Compiled Keras model
        """
        n_classes = self.config.get('model.lstm.n_classes', 101)
        chunk_size = self.config.get('model.lstm.chunk_size', 1024)
        n_chunks = self.config.get('model.lstm.n_chunks', 15)
        rnn_size = self.config.get('model.lstm.rnn_size', 256)
        n_layers = self.config.get('model.lstm.n_layers', 2)
        learning_rate = self.config.get('model.lstm.learning_rate', 0.001)
        
        # Input layer
        inputs = tf.keras.Input(
            shape=(n_chunks, chunk_size), 
            name='features'
        )
        
        # Dense layer for feature transformation
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(rnn_size, activation='relu'),
            name='feature_transform'
        )(inputs)
        
        # Multi-layer LSTM
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)  # Only last layer doesn't return sequences
            x = tf.keras.layers.LSTM(
                rnn_size,
                return_sequences=return_sequences,
                dropout=0.2,
                recurrent_dropout=0.2,
                name=f'lstm_{i+1}'
            )(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            n_classes, 
            activation='softmax',
            name='classification'
        )(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ActivityLSTM')
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        logger.info("LSTM model built successfully")
        return model
    
    def prepare_data(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepare and split data for training.
        
        Args:
            features: Feature array of shape (n_samples, feature_dim)
            labels: One-hot encoded labels of shape (n_samples, n_classes)
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        # Reshape features for LSTM input
        chunk_size = self.config.get('model.lstm.chunk_size', 1024)
        n_chunks = self.config.get('model.lstm.n_chunks', 15)
        
        features_reshaped = features.reshape((-1, n_chunks, chunk_size))
        
        # Split data
        test_size = self.config.get('training.test_size', 0.2)
        validation_size = self.config.get('training.validation_size', 0.2)
        random_state = self.config.get('training.random_state', 42)
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features_reshaped, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels.argmax(axis=1)
        )
        
        # Second split: train vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train_val.argmax(axis=1)
        )
        
        train_data = {'X': X_train, 'y': y_train}
        val_data = {'X': X_val, 'y': y_val}
        test_data = {'X': X_test, 'y': y_test}
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, "
                   f"Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return train_data, val_data, test_data
    
    def train(
        self, 
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        model_save_path: str = None
    ) -> tf.keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            model_save_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        epochs = self.config.get('model.lstm.epochs', 500)
        batch_size = self.config.get('model.lstm.batch_size', 128)
        
        if model_save_path is None:
            model_save_path = self.config.get('model.lstm.model_save_path', 'models/trained')
        
        model_save_path = Path(model_save_path)
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save_path / 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=20,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                model_save_path / 'training_log.csv',
                append=False
            )
        ]
        
        logger.info("Starting model training...")
        start_time = time.time()
        
        # Train model
        self.history = self.model.fit(
            train_data['X'], train_data['y'],
            validation_data=(val_data['X'], val_data['y']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.model.save(model_save_path / 'final_model.h5')
        
        return self.history
    
    def evaluate(
        self, 
        test_data: Dict[str, np.ndarray],
        class_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data dictionary
            class_names: List of class names for detailed metrics
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or load_model() first.")
        
        X_test, y_test = test_data['X'], test_data['y']
        
        # Get predictions
        logger.info("Evaluating model...")
        y_pred_proba = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_top5_acc = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_top5_accuracy': test_top5_acc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true,
            'prediction_probabilities': y_pred_proba
        }
        
        if class_names:
            results['classification_report'] = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True
            )
        
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Top-5 Accuracy: {test_top5_acc:.4f}")
        
        return results
    
    def predict_video(
        self, 
        features: np.ndarray,
        class_names: list = None
    ) -> Dict[str, Any]:
        """
        Predict activity for a single video's features.
        
        Args:
            features: Video features array
            class_names: List of class names
            
        Returns:
            Prediction results dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() or load_model() first.")
        
        # Reshape features for model input
        chunk_size = self.config.get('model.lstm.chunk_size', 1024)
        n_chunks = self.config.get('model.lstm.n_chunks', 15)
        
        features_reshaped = features.reshape(1, n_chunks, chunk_size)
        
        # Get prediction
        prediction_proba = self.model.predict(features_reshaped, verbose=0)
        predicted_class = np.argmax(prediction_proba, axis=1)[0]
        confidence = prediction_proba[0, predicted_class]
        
        results = {
            'predicted_class_idx': predicted_class,
            'confidence': confidence,
            'all_probabilities': prediction_proba[0]
        }
        
        if class_names and predicted_class < len(class_names):
            results['predicted_class_name'] = class_names[predicted_class]
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("No model to save. Build or train a model first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a model from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def save_results(
        self, 
        results: Dict[str, Any], 
        output_path: str
    ) -> None:
        """Save evaluation results to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions and labels
        np.save(output_path / 'predictions.npy', results['predictions'])
        np.save(output_path / 'true_labels.npy', results['true_labels'])
        np.save(output_path / 'prediction_probabilities.npy', results['prediction_probabilities'])
        
        # Save confusion matrix
        np.save(output_path / 'confusion_matrix.npy', results['confusion_matrix'])
        
        # Save in MATLAB format for compatibility
        sio.savemat(
            output_path / 'results.mat',
            {
                'predictions': results['predictions'],
                'true_labels': results['true_labels'],
                'confusion_matrix': results['confusion_matrix']
            }
        )
        
        # Save classification report if available
        if 'classification_report' in results:
            import json
            with open(output_path / 'classification_report.json', 'w') as f:
                json.dump(results['classification_report'], f, indent=2)
        
        logger.info(f"Results saved to {output_path}")