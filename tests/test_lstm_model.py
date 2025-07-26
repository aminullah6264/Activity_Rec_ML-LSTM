"""Tests for LSTM model module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from activity_recognition.lstm_model import LSTMModel


class TestLSTMModel:
    """Test LSTM model functionality."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = LSTMModel()
        
        assert model.model is None
        assert model.class_names == []
        assert model.history is None
    
    def test_model_building(self):
        """Test model building with default parameters."""
        model = LSTMModel()
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        
        # Check model structure
        assert len(built_model.layers) > 0
        assert built_model.input_shape == (None, 15, 1024)  # (batch, n_chunks, chunk_size)
        assert built_model.output_shape == (None, 101)  # (batch, n_classes)
    
    def test_model_building_with_custom_config(self):
        """Test model building with custom configuration."""
        config_override = {
            'model.lstm.n_classes': 50,
            'model.lstm.chunk_size': 512,
            'model.lstm.n_chunks': 10,
            'model.lstm.rnn_size': 128
        }
        
        model = LSTMModel(config_override)
        built_model = model.build_model()
        
        assert built_model.input_shape == (None, 10, 512)
        assert built_model.output_shape == (None, 50)
    
    def test_data_preparation(self):
        """Test data preparation and splitting."""
        # Create dummy data
        n_samples = 100
        feature_dim = 15360  # 15 * 1024
        n_classes = 10
        
        features = np.random.randn(n_samples, feature_dim)
        labels = np.eye(n_classes)[np.random.randint(0, n_classes, n_samples)]
        
        model = LSTMModel({'model.lstm.n_classes': n_classes})
        train_data, val_data, test_data = model.prepare_data(features, labels)
        
        # Check data shapes
        assert train_data['X'].ndim == 3  # (samples, chunks, chunk_size)
        assert train_data['y'].shape[1] == n_classes
        
        # Check splits
        total_samples = (
            train_data['X'].shape[0] + 
            val_data['X'].shape[0] + 
            test_data['X'].shape[0]
        )
        assert total_samples == n_samples
        
        # Check that test set is approximately 20% of data
        assert abs(test_data['X'].shape[0] / n_samples - 0.2) < 0.05
    
    def test_prediction_single_video(self):
        """Test prediction on single video features."""
        # Create dummy model and data
        model = LSTMModel({'model.lstm.n_classes': 5})
        model.build_model()
        
        # Create dummy features
        feature_dim = 15360
        features = np.random.randn(feature_dim)
        
        class_names = ['class1', 'class2', 'class3', 'class4', 'class5']
        
        # Test prediction
        result = model.predict_video(features, class_names)
        
        assert 'predicted_class_idx' in result
        assert 'confidence' in result
        assert 'all_probabilities' in result
        assert 'predicted_class_name' in result
        
        assert 0 <= result['predicted_class_idx'] < 5
        assert 0 <= result['confidence'] <= 1
        assert len(result['all_probabilities']) == 5
        assert result['predicted_class_name'] in class_names
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Build model
        model = LSTMModel({'model.lstm.n_classes': 5})
        model.build_model()
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            model_path = f.name
        
        try:
            model.save_model(model_path)
            
            # Create new model and load
            model2 = LSTMModel()
            model2.load_model(model_path)
            
            assert model2.model is not None
            
            # Test that both models have same architecture
            assert len(model.model.layers) == len(model2.model.layers)
            
        finally:
            Path(model_path).unlink()
    
    def test_evaluation_metrics(self):
        """Test model evaluation with dummy data."""
        # Create small model for faster testing
        model = LSTMModel({
            'model.lstm.n_classes': 3,
            'model.lstm.rnn_size': 32,
            'model.lstm.n_chunks': 5,
            'model.lstm.chunk_size': 64
        })
        model.build_model()
        
        # Create dummy test data
        n_samples = 20
        feature_dim = 5 * 64  # n_chunks * chunk_size
        
        X_test = np.random.randn(n_samples, 5, 64)
        y_test = np.eye(3)[np.random.randint(0, 3, n_samples)]
        
        test_data = {'X': X_test, 'y': y_test}
        class_names = ['action1', 'action2', 'action3']
        
        # Test evaluation
        results = model.evaluate(test_data, class_names)
        
        # Check results structure
        assert 'test_loss' in results
        assert 'test_accuracy' in results
        assert 'test_top5_accuracy' in results
        assert 'confusion_matrix' in results
        assert 'predictions' in results
        assert 'true_labels' in results
        assert 'prediction_probabilities' in results
        assert 'classification_report' in results
        
        # Check shapes
        assert results['confusion_matrix'].shape == (3, 3)
        assert len(results['predictions']) == n_samples
        assert len(results['true_labels']) == n_samples
        assert results['prediction_probabilities'].shape == (n_samples, 3)
    
    def test_config_override(self):
        """Test configuration override functionality."""
        config_override = {
            'model.lstm.batch_size': 64,
            'model.lstm.learning_rate': 0.01,
            'model.lstm.epochs': 10
        }
        
        model = LSTMModel(config_override)
        
        assert model.config.get('model.lstm.batch_size') == 64
        assert model.config.get('model.lstm.learning_rate') == 0.01
        assert model.config.get('model.lstm.epochs') == 10
        
        # Check that other values remain default
        assert model.config.get('model.lstm.n_classes') == 101