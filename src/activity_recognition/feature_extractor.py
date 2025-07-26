"""Feature extraction module for optical flow-based activity recognition."""

import os
import tempfile
from math import ceil
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .config import config

# Optional caffe import with fallback
try:
    import caffe
    CAFFE_AVAILABLE = True
except ImportError:
    CAFFE_AVAILABLE = False
    logging.warning("Caffe not available. Feature extraction will not work without it.")

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract temporal optical flow features using FlowNet2."""
    
    def __init__(self, config_override: dict = None):
        """
        Initialize the feature extractor.
        
        Args:
            config_override: Optional configuration overrides
        """
        if not CAFFE_AVAILABLE:
            raise ImportError("Caffe is required for feature extraction")
            
        self.config = config
        if config_override:
            for key, value in config_override.items():
                self.config.set(key, value)
        
        self.net = None
        self._setup_network()
    
    def _setup_network(self) -> None:
        """Setup the FlowNet2 network."""
        caffemodel = self.config.get('model.flownet.caffemodel')
        deploy_proto = self.config.get('model.flownet.deploy_proto')
        gpu_id = self.config.get('model.flownet.gpu_id', 0)
        
        if not os.path.exists(caffemodel):
            raise FileNotFoundError(f"Caffe model not found: {caffemodel}")
        if not os.path.exists(deploy_proto):
            raise FileNotFoundError(f"Deploy proto not found: {deploy_proto}")
        
        # Setup network parameters
        width = self.config.get('model.flownet.width', 512)
        height = self.config.get('model.flownet.height', 386)
        
        self.vars = self._create_network_vars(width, height)
        
        # Create temporary proto file with substituted variables
        self.tmp_proto = self._create_temp_proto(deploy_proto)
        
        # Initialize Caffe
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()
        caffe.set_logging_disabled()
        
        self.net = caffe.Net(self.tmp_proto.name, caffemodel, caffe.TEST)
        logger.info("FlowNet2 network initialized successfully")
    
    def _create_network_vars(self, width: int, height: int) -> dict:
        """Create network variable substitutions."""
        divisor = 64.0
        adapted_width = int(ceil(width / divisor) * divisor)
        adapted_height = int(ceil(height / divisor) * divisor)
        
        return {
            'TARGET_WIDTH': width,
            'TARGET_HEIGHT': height,
            'ADAPTED_WIDTH': adapted_width,
            'ADAPTED_HEIGHT': adapted_height,
            'SCALE_WIDTH': width / float(adapted_width),
            'SCALE_HEIGHT': height / float(adapted_height)
        }
    
    def _create_temp_proto(self, deploy_proto_path: str) -> tempfile.NamedTemporaryFile:
        """Create temporary proto file with variable substitutions."""
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.prototxt')
        
        with open(deploy_proto_path, 'r') as f:
            proto_lines = f.readlines()
        
        for line in proto_lines:
            for key, value in self.vars.items():
                tag = f"${key}$"
                line = line.replace(tag, str(value))
            tmp.write(line)
        
        tmp.flush()
        return tmp
    
    def extract_features_from_video(
        self, 
        video_path: Union[str, Path]
    ) -> Tuple[np.ndarray, str]:
        """
        Extract features from a single video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (features_array, class_name)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract class name from parent directory
        class_name = video_path.parent.name
        
        vidcap = cv2.VideoCapture(str(video_path))
        if not vidcap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_features = []
            frame_no = -1
            
            frames_per_segment = self.config.get('processing.frames_per_segment', 30)
            
            with tqdm(total=video_length//2, desc=f"Processing {video_path.name}") as pbar:
                while frame_no < video_length - 1:
                    frame_features = self._extract_frame_pair_features(
                        vidcap, frame_no
                    )
                    
                    if frame_features is not None:
                        video_features.append(frame_features)
                        
                        # Create segment every N frames
                        if (frame_no + 1) % frames_per_segment == 0:
                            if len(video_features) == frames_per_segment:
                                features_array = np.asarray(video_features)
                                yield features_array, class_name
                                video_features = []
                    
                    frame_no += 2  # Skip one frame for optical flow
                    pbar.update(1)
            
            # Handle remaining frames
            if video_features:
                features_array = np.asarray(video_features)
                yield features_array, class_name
        
        finally:
            vidcap.release()
    
    def _extract_frame_pair_features(
        self, 
        vidcap: cv2.VideoCapture, 
        frame_no: int
    ) -> Optional[np.ndarray]:
        """Extract features from a pair of consecutive frames."""
        # Read first frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret0, img0 = vidcap.read()
        
        # Read second frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + 1)
        ret1, img1 = vidcap.read()
        
        if not (ret0 and ret1):
            return None
        
        # Resize frames
        width = self.config.get('model.flownet.width', 512)
        height = self.config.get('model.flownet.height', 386)
        
        img0 = cv2.resize(img0, (width, height))
        img1 = cv2.resize(img1, (width, height))
        
        # Prepare input data
        input_data = self._prepare_input_data(img0, img1)
        
        # Run network forward pass
        input_dict = {
            self.net.inputs[i]: input_data[i] 
            for i in range(len(input_data))
        }
        
        self.net.forward(**input_dict)
        
        # Extract features from conv6_1 layer
        blob = np.squeeze(self.net.blobs['conv6_1'].data).transpose(1, 2, 0)
        
        # Reshape and max pool features
        arr = np.array(blob)
        arr_reshaped = arr.reshape([56, 1024])
        bb = np.matrix(arr_reshaped)
        features = bb.max(0)
        
        return features
    
    def _prepare_input_data(self, img0: np.ndarray, img1: np.ndarray) -> List[np.ndarray]:
        """Prepare input data for the network."""
        input_data = []
        
        for img in [img0, img1]:
            if len(img.shape) < 3:
                # Grayscale image
                input_data.append(img[np.newaxis, np.newaxis, :, :])
            else:
                # Color image - convert BGR to RGB and transpose
                input_data.append(
                    img[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
                )
        
        return input_data
    
    def extract_features_from_dataset(
        self, 
        dataset_path: Union[str, Path],
        output_path: Union[str, Path] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from entire dataset.
        
        Args:
            dataset_path: Path to dataset directory
            output_path: Optional path to save features
            
        Returns:
            Tuple of (features, labels)
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        all_features = []
        all_labels = []
        class_names = []
        
        # Process each class directory
        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in class_names:
                class_names.append(class_name)
            
            logger.info(f"Processing class: {class_name}")
            
            # Process each video in the class
            video_files = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
            
            for video_file in video_files:
                try:
                    for features, _ in self.extract_features_from_video(video_file):
                        all_features.append(features)
                        all_labels.append(class_name)
                        
                except Exception as e:
                    logger.error(f"Error processing {video_file}: {e}")
                    continue
        
        # Convert to arrays and create one-hot encoding
        features_array = self._process_features(all_features)
        labels_array = self._create_one_hot_labels(all_labels, class_names)
        
        # Save features if output path provided
        if output_path:
            self._save_features(features_array, labels_array, class_names, output_path)
        
        return features_array, labels_array
    
    def _process_features(self, features_list: List[np.ndarray]) -> np.ndarray:
        """Process and reshape features list."""
        total_features = []
        feature_dim = self.config.get('processing.feature_dim', 15360)
        
        for sample in features_list:
            total_features.append(sample.reshape([1, feature_dim]))
        
        return np.asarray(total_features).reshape([len(features_list), feature_dim])
    
    def _create_one_hot_labels(
        self, 
        labels: List[str], 
        class_names: List[str]
    ) -> np.ndarray:
        """Create one-hot encoded labels."""
        n_samples = len(labels)
        n_classes = len(class_names)
        one_hot = np.zeros([n_samples, n_classes], dtype=int)
        
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for i, label in enumerate(labels):
            if label in class_to_idx:
                one_hot[i, class_to_idx[label]] = 1
        
        return one_hot
    
    def _save_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        output_path: Union[str, Path]
    ) -> None:
        """Save extracted features to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        np.save(output_path / 'features.npy', features)
        np.save(output_path / 'labels.npy', labels)
        
        # Save class names
        with open(output_path / 'class_names.txt', 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        # Save in MATLAB format for compatibility
        sio.savemat(
            output_path / 'features.mat',
            {
                'TotalFeatures': features,
                'DatabaseLabel': labels
            },
            appendmat=True,
            format='5',
            long_field_names=False,
            do_compression=True,
            oned_as='row'
        )
        
        logger.info(f"Features saved to {output_path}")
    
    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, 'tmp_proto') and self.tmp_proto:
            try:
                os.unlink(self.tmp_proto.name)
            except:
                pass