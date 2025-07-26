"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path
import yaml

from activity_recognition.config import Config


class TestConfig:
    """Test configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = Config()
        
        # Test basic structure
        assert 'data' in config.all
        assert 'model' in config.all
        assert 'processing' in config.all
        assert 'training' in config.all
        
        # Test default values
        assert config.get('model.lstm.n_classes') == 101
        assert config.get('processing.frames_per_segment') == 30
        assert config.get('training.test_size') == 0.2
    
    def test_get_set_operations(self):
        """Test get and set operations."""
        config = Config()
        
        # Test getting existing value
        batch_size = config.get('model.lstm.batch_size')
        assert batch_size == 128
        
        # Test getting non-existing value with default
        assert config.get('non.existing.key', 'default') == 'default'
        
        # Test setting value
        config.set('model.lstm.batch_size', 64)
        assert config.get('model.lstm.batch_size') == 64
        
        # Test setting new nested key
        config.set('new.nested.key', 'value')
        assert config.get('new.nested.key') == 'value'
    
    def test_file_operations(self):
        """Test loading and saving configuration files."""
        # Create temporary config file
        test_config = {
            'model': {
                'lstm': {
                    'batch_size': 256,
                    'epochs': 100
                }
            },
            'new_section': {
                'new_key': 'new_value'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            config_file = f.name
        
        try:
            # Test loading from file
            config = Config(config_file)
            
            # Test that values were loaded
            assert config.get('model.lstm.batch_size') == 256
            assert config.get('model.lstm.epochs') == 100
            assert config.get('new_section.new_key') == 'new_value'
            
            # Test that default values are still present
            assert config.get('model.lstm.n_classes') == 101
            
            # Test saving to file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
                save_file = f2.name
            
            config.save_to_file(save_file)
            
            # Load the saved file and verify
            config2 = Config(save_file)
            assert config2.get('model.lstm.batch_size') == 256
            assert config2.get('new_section.new_key') == 'new_value'
            
        finally:
            # Cleanup
            Path(config_file).unlink()
            if 'save_file' in locals():
                Path(save_file).unlink()
    
    def test_nested_update(self):
        """Test that nested updates work correctly."""
        config = Config()
        
        # Original values
        original_epochs = config.get('model.lstm.epochs')
        original_n_classes = config.get('model.lstm.n_classes')
        
        # Update with partial config
        partial_config = {
            'model': {
                'lstm': {
                    'epochs': 200
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(partial_config, f)
            config_file = f.name
        
        try:
            config.load_from_file(config_file)
            
            # Check that epochs was updated
            assert config.get('model.lstm.epochs') == 200
            
            # Check that other values are preserved
            assert config.get('model.lstm.n_classes') == original_n_classes
            
        finally:
            Path(config_file).unlink()