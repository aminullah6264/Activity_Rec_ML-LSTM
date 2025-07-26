"""Configuration module for activity recognition system."""

from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Configuration class for the activity recognition system."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration with default values."""
        self.config_path = config_path
        self._config = self._load_default_config()
        
        if config_path:
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            "data": {
                "dataset_folder": "data/raw",
                "output_folder": "data/processed",
                "class_names_file": "config/class_names.txt"
            },
            "model": {
                "flownet": {
                    "caffemodel": "models/pretrained/FlowNet2_weights.caffemodel.h5",
                    "deploy_proto": "models/pretrained/FlowNet2_deploy.prototxt.template",
                    "width": 512,
                    "height": 386,
                    "gpu_id": 0
                },
                "lstm": {
                    "n_classes": 101,
                    "chunk_size": 1024,
                    "n_chunks": 15,
                    "rnn_size": 256,
                    "n_layers": 2,
                    "epochs": 500,
                    "batch_size": 128,
                    "validation_batch_size": 64,
                    "learning_rate": 0.001,
                    "model_save_path": "models/trained"
                }
            },
            "processing": {
                "frames_per_segment": 30,
                "feature_dim": 15360,
                "max_pool_features": True
            },
            "training": {
                "test_size": 0.2,
                "validation_size": 0.2,
                "random_state": 42
            }
        }
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                self._update_config(self._config, file_config)
    
    def _update_config(self, base_config: Dict, update_config: Dict) -> None:
        """Recursively update configuration."""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get(self, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.lstm.batch_size')."""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to YAML file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
config = Config()