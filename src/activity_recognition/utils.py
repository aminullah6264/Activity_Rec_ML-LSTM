"""Utility functions for the activity recognition system."""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_class_names_file(
    dataset_path: Union[str, Path], 
    output_path: Union[str, Path]
) -> List[str]:
    """
    Create class names file from dataset directory structure.
    
    Args:
        dataset_path: Path to dataset directory
        output_path: Path to save class names file
        
    Returns:
        List of class names
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Get class names from directory structure
    class_names = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir():
            class_names.append(item.name)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    logger.info(f"Created class names file with {len(class_names)} classes: {output_path}")
    return class_names


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the matrix
        title: Plot title
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm = confusion_matrix
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {output_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path]
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved to: {output_path}")


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Analyze model predictions and save detailed results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        class_names: List of class names
        output_dir: Directory to save analysis results
        
    Returns:
        Analysis results dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Top-k accuracy
    top5_accuracy = np.mean([
        true_label in pred_proba.argsort()[-5:]
        for true_label, pred_proba in zip(y_true, y_pred_proba)
    ])
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Confidence statistics
    confidence_stats = {
        'mean_confidence': np.mean(np.max(y_pred_proba, axis=1)),
        'std_confidence': np.std(np.max(y_pred_proba, axis=1)),
        'min_confidence': np.min(np.max(y_pred_proba, axis=1)),
        'max_confidence': np.max(np.max(y_pred_proba, axis=1))
    }
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'per_class_accuracy': per_class_accuracy.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'confidence_stats': confidence_stats
    }
    
    # Save detailed results
    with open(output_dir / 'detailed_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names, 
        output_dir / 'confusion_matrix.png'
    )
    
    plot_confusion_matrix(
        cm, class_names, 
        output_dir / 'confusion_matrix_normalized.png',
        normalize=True,
        title="Normalized Confusion Matrix"
    )
    
    # Plot per-class accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), per_class_accuracy)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(np.max(y_pred_proba, axis=1), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Detailed analysis saved to: {output_dir}")
    logger.info(f"Overall accuracy: {accuracy:.4f}")
    logger.info(f"Top-5 accuracy: {top5_accuracy:.4f}")
    
    return results


def validate_dataset_structure(dataset_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate dataset directory structure and return statistics.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dataset statistics dictionary
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    stats = {
        'total_classes': 0,
        'total_videos': 0,
        'classes': {},
        'video_extensions': set(),
        'issues': []
    }
    
    # Analyze each class directory
    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        video_files = []
        
        # Find video files
        for ext in ['.avi', '.mp4', '.mov', '.mkv']:
            video_files.extend(class_dir.glob(f'*{ext}'))
            video_files.extend(class_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            stats['issues'].append(f"No video files found in class '{class_name}'")
            continue
        
        # Collect statistics
        stats['classes'][class_name] = {
            'video_count': len(video_files),
            'video_files': [f.name for f in video_files]
        }
        
        stats['total_classes'] += 1
        stats['total_videos'] += len(video_files)
        
        # Track extensions
        for video_file in video_files:
            stats['video_extensions'].add(video_file.suffix.lower())
    
    stats['video_extensions'] = list(stats['video_extensions'])
    
    logger.info(f"Dataset validation completed:")
    logger.info(f"  Total classes: {stats['total_classes']}")
    logger.info(f"  Total videos: {stats['total_videos']}")
    logger.info(f"  Video extensions: {stats['video_extensions']}")
    
    if stats['issues']:
        logger.warning("Issues found:")
        for issue in stats['issues']:
            logger.warning(f"  - {issue}")
    
    return stats


def resize_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 386),
    fps: int = None
) -> None:
    """
    Resize video to target dimensions.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        target_size: Target (width, height)
        fps: Target FPS (None to keep original)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = original_fps
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, target_size)
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            resized_frame = cv2.resize(frame, target_size)
            out.write(resized_frame)
    
    finally:
        cap.release()
        out.release()
    
    logger.info(f"Video resized: {input_path} -> {output_path}")


def calculate_dataset_statistics(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for extracted features dataset.
    
    Args:
        features: Feature array
        labels: Label array
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'feature_shape': features.shape,
        'label_shape': labels.shape,
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        'n_classes': labels.shape[1],
        'feature_stats': {
            'mean': np.mean(features),
            'std': np.std(features),
            'min': np.min(features),
            'max': np.max(features)
        },
        'class_distribution': np.sum(labels, axis=0).tolist(),
        'samples_per_class': {
            'mean': np.mean(np.sum(labels, axis=0)),
            'std': np.std(np.sum(labels, axis=0)),
            'min': np.min(np.sum(labels, axis=0)),
            'max': np.max(np.sum(labels, axis=0))
        }
    }
    
    return stats