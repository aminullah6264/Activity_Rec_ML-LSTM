"""
Activity Recognition using Temporal Optical Flow Convolutional Features and Multi-Layer LSTM

A modernized implementation for activity recognition in videos using optical flow features
extracted via FlowNet2 and classified using multi-layer LSTM networks.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Conditional imports to handle missing dependencies gracefully
__all__ = []

try:
    from .feature_extractor import FeatureExtractor
    __all__.append("FeatureExtractor")
except ImportError:
    pass

try:
    from .lstm_model import LSTMModel
    __all__.append("LSTMModel")
except ImportError:
    pass

try:
    from .video_processor import VideoProcessor
    __all__.append("VideoProcessor")
except ImportError:
    pass