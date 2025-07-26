"""Video processing module for real-time activity recognition."""

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import logging

import cv2
import numpy as np
from tqdm import tqdm

from .config import config
from .feature_extractor import FeatureExtractor
from .lstm_model import LSTMModel

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos for real-time activity recognition."""
    
    def __init__(
        self, 
        model_path: str = None,
        class_names_file: str = None,
        config_override: dict = None
    ):
        """
        Initialize the video processor.
        
        Args:
            model_path: Path to trained LSTM model
            class_names_file: Path to class names file
            config_override: Optional configuration overrides
        """
        self.config = config
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.feature_extractor = None
        self.lstm_model = None
        self.class_names = []
        
        # Initialize components
        self._setup_feature_extractor()
        self._setup_lstm_model(model_path)
        self._load_class_names(class_names_file)
    
    def _setup_feature_extractor(self) -> None:
        """Setup the feature extractor."""
        try:
            self.feature_extractor = FeatureExtractor()
            logger.info("Feature extractor initialized successfully")
        except ImportError:
            logger.warning("Caffe not available. Feature extraction will be disabled.")
            self.feature_extractor = None
    
    def _setup_lstm_model(self, model_path: str = None) -> None:
        """Setup the LSTM model."""
        self.lstm_model = LSTMModel()
        
        if model_path:
            try:
                self.lstm_model.load_model(model_path)
                logger.info(f"LSTM model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                self.lstm_model.build_model()
                logger.info("Built new LSTM model instead")
        else:
            self.lstm_model.build_model()
            logger.info("Built new LSTM model")
    
    def _load_class_names(self, class_names_file: str = None) -> None:
        """Load class names from file."""
        if class_names_file is None:
            class_names_file = self.config.get('data.class_names_file')
        
        class_names_path = Path(class_names_file)
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.class_names)} class names")
        else:
            logger.warning(f"Class names file not found: {class_names_file}")
            # Create default class names if file doesn't exist
            n_classes = self.config.get('model.lstm.n_classes', 101)
            self.class_names = [f"Class_{i:03d}" for i in range(n_classes)]
    
    def process_video_file(
        self, 
        video_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a video file and return activity predictions for each segment.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video with annotations
            show_progress: Whether to show processing progress
            
        Returns:
            List of prediction results for each segment
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not available")
        
        predictions = []
        
        # Open video
        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Setup video writer if output path provided
        video_writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get video properties
            fps = int(vidcap.get(cv2.CAP_PROP_FPS))
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
        
        try:
            video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_per_segment = self.config.get('processing.frames_per_segment', 30)
            
            logger.info(f"Processing video: {video_path.name}")
            logger.info(f"Video length: {video_length} frames")
            
            if show_progress:
                pbar = tqdm(total=video_length, desc="Processing video")
            
            segment_features = []
            frame_count = 0
            
            while True:
                ret, frame = vidcap.read()
                if not ret:
                    break
                
                # Extract features for every pair of frames
                if frame_count % 2 == 0 and frame_count < video_length - 1:
                    # Get next frame for optical flow
                    ret_next, next_frame = vidcap.read()
                    if ret_next:
                        # Extract features using feature extractor
                        features = self._extract_frame_features(frame, next_frame)
                        if features is not None:
                            segment_features.append(features)
                        
                        frame_count += 1  # Account for the next frame
                
                # Process segment when we have enough frames
                if len(segment_features) == frames_per_segment:
                    # Predict activity for this segment
                    prediction = self._predict_segment(segment_features)
                    predictions.append(prediction)
                    
                    # Annotate frames if output video requested
                    if video_writer:
                        self._annotate_segment_frames(
                            vidcap, frame_count - frames_per_segment, 
                            frames_per_segment, prediction, video_writer
                        )
                    
                    segment_features = []
                
                frame_count += 1
                if show_progress:
                    pbar.update(1)
            
            # Handle remaining features
            if segment_features:
                prediction = self._predict_segment(segment_features)
                predictions.append(prediction)
                
                if video_writer:
                    remaining_frames = len(segment_features)
                    self._annotate_segment_frames(
                        vidcap, frame_count - remaining_frames,
                        remaining_frames, prediction, video_writer
                    )
            
            if show_progress:
                pbar.close()
        
        finally:
            vidcap.release()
            if video_writer:
                video_writer.release()
        
        logger.info(f"Processed {len(predictions)} segments")
        return predictions
    
    def _extract_frame_features(
        self, 
        frame1: np.ndarray, 
        frame2: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract features from a pair of frames."""
        try:
            # Resize frames to match FlowNet2 requirements
            width = self.config.get('model.flownet.width', 512)
            height = self.config.get('model.flownet.height', 386)
            
            frame1_resized = cv2.resize(frame1, (width, height))
            frame2_resized = cv2.resize(frame2, (width, height))
            
            # Use feature extractor's internal method
            # Prepare input data
            input_data = self.feature_extractor._prepare_input_data(frame1_resized, frame2_resized)
            
            # Run network forward pass
            input_dict = {
                self.feature_extractor.net.inputs[i]: input_data[i] 
                for i in range(len(input_data))
            }
            
            self.feature_extractor.net.forward(**input_dict)
            
            # Extract features from conv6_1 layer
            blob = np.squeeze(self.feature_extractor.net.blobs['conv6_1'].data).transpose(1, 2, 0)
            
            # Reshape and max pool features
            arr = np.array(blob)
            arr_reshaped = arr.reshape([56, 1024])
            bb = np.matrix(arr_reshaped)
            features = bb.max(0)
            
            return features
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None
    
    def _predict_segment(self, segment_features: List[np.ndarray]) -> Dict[str, Any]:
        """Predict activity for a segment of features."""
        try:
            # Convert features to proper format
            features_array = np.asarray(segment_features)
            feature_dim = self.config.get('processing.feature_dim', 15360)
            
            # Reshape for LSTM input
            features_flat = features_array.reshape([1, feature_dim])
            
            # Get prediction
            prediction = self.lstm_model.predict_video(
                features_flat, self.class_names
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'predicted_class_idx': 0,
                'predicted_class_name': 'Unknown',
                'confidence': 0.0
            }
    
    def _annotate_segment_frames(
        self,
        vidcap: cv2.VideoCapture,
        start_frame: int,
        num_frames: int,
        prediction: Dict[str, Any],
        video_writer: cv2.VideoWriter
    ) -> None:
        """Annotate frames with prediction results."""
        # Get current position
        current_pos = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Setup text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)  # White text
        thickness = 2
        
        # Create annotation text
        class_name = prediction.get('predicted_class_name', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        text = f"Activity: {class_name} ({confidence:.2f})"
        
        # Annotate frames
        for i in range(num_frames):
            frame_idx = start_frame + i
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vidcap.read()
            
            if ret:
                # Add text annotation
                cv2.putText(
                    frame, text, (20, 30), font, font_scale, color, thickness
                )
                video_writer.write(frame)
        
        # Restore position
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    def process_video_stream(
        self,
        source: Union[int, str] = 0,
        display: bool = True,
        save_path: str = None
    ) -> None:
        """
        Process video stream (webcam or file) in real-time.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            display: Whether to display the video with annotations
            save_path: Optional path to save annotated video
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not available")
        
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Setup video writer if save path provided
        video_writer = None
        if save_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        try:
            logger.info("Starting real-time video processing. Press 'q' to quit.")
            
            frame_buffer = []
            segment_features = []
            frames_per_segment = self.config.get('processing.frames_per_segment', 30)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_buffer.append(frame.copy())
                
                # Extract features from consecutive frames
                if len(frame_buffer) >= 2:
                    features = self._extract_frame_features(
                        frame_buffer[-2], frame_buffer[-1]
                    )
                    if features is not None:
                        segment_features.append(features)
                    
                    # Keep only recent frames in buffer
                    if len(frame_buffer) > 2:
                        frame_buffer.pop(0)
                
                # Predict when we have enough features
                current_prediction = None
                if len(segment_features) >= frames_per_segment:
                    current_prediction = self._predict_segment(
                        segment_features[-frames_per_segment:]
                    )
                
                # Annotate frame
                if current_prediction:
                    class_name = current_prediction.get('predicted_class_name', 'Unknown')
                    confidence = current_prediction.get('confidence', 0.0)
                    text = f"Activity: {class_name} ({confidence:.2f})"
                    
                    cv2.putText(
                        frame, text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                
                # Save frame if requested
                if video_writer:
                    video_writer.write(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Activity Recognition', frame)
                    
                    # Check for quit command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        logger.info("Real-time processing stopped")
    
    def benchmark_video(
        self, 
        video_path: Union[str, Path]
    ) -> Dict[str, float]:
        """
        Benchmark processing speed on a video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with timing statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        start_time = time.time()
        predictions = self.process_video_file(video_path, show_progress=False)
        end_time = time.time()
        
        # Calculate statistics
        total_time = end_time - start_time
        num_segments = len(predictions)
        avg_time_per_segment = total_time / num_segments if num_segments > 0 else 0
        
        # Get video info
        vidcap = cv2.VideoCapture(str(video_path))
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_duration = video_length / fps
        vidcap.release()
        
        processing_fps = video_length / total_time
        real_time_factor = video_duration / total_time
        
        return {
            'total_processing_time': total_time,
            'video_duration': video_duration,
            'num_segments': num_segments,
            'avg_time_per_segment': avg_time_per_segment,
            'processing_fps': processing_fps,
            'real_time_factor': real_time_factor
        }