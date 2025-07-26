"""Command-line interface for activity recognition system."""

import logging
from pathlib import Path
from typing import Optional
import sys

import click
import numpy as np

from .config import config
from .feature_extractor import FeatureExtractor
from .lstm_model import LSTMModel
from .video_processor import VideoProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config_file: Optional[str], verbose: bool):
    """Activity Recognition using Temporal Optical Flow and Multi-Layer LSTM."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config_file:
        config.load_from_file(config_file)
        logger.info(f"Loaded configuration from {config_file}")


@cli.command()
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True),
              help='Path to dataset directory')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for extracted features')
@click.option('--flownet-model', type=click.Path(exists=True),
              help='Path to FlowNet2 caffemodel file')
@click.option('--flownet-proto', type=click.Path(exists=True),
              help='Path to FlowNet2 deploy prototxt file')
@click.option('--gpu-id', type=int, default=0, help='GPU ID to use')
def extract_features(
    dataset: str,
    output: str,
    flownet_model: Optional[str],
    flownet_proto: Optional[str],
    gpu_id: int
):
    """Extract temporal optical flow features from video dataset."""
    logger.info("Starting feature extraction...")
    
    # Update configuration with provided paths
    config_override = {'model.flownet.gpu_id': gpu_id}
    if flownet_model:
        config_override['model.flownet.caffemodel'] = flownet_model
    if flownet_proto:
        config_override['model.flownet.deploy_proto'] = flownet_proto
    
    try:
        # Initialize feature extractor
        extractor = FeatureExtractor(config_override)
        
        # Extract features
        features, labels = extractor.extract_features_from_dataset(
            dataset_path=dataset,
            output_path=output
        )
        
        logger.info(f"Feature extraction completed successfully!")
        logger.info(f"Extracted features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Features saved to: {output}")
        
    except ImportError:
        logger.error("Caffe is not available. Please install Caffe to use feature extraction.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--features', '-f', required=True, type=click.Path(exists=True),
              help='Path to extracted features file (.npy)')
@click.option('--labels', '-l', required=True, type=click.Path(exists=True),
              help='Path to labels file (.npy)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for trained model')
@click.option('--epochs', type=int, help='Number of training epochs')
@click.option('--batch-size', type=int, help='Training batch size')
@click.option('--learning-rate', type=float, help='Learning rate')
@click.option('--n-classes', type=int, help='Number of classes')
def train_model(
    features: str,
    labels: str,
    output: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: Optional[float],
    n_classes: Optional[int]
):
    """Train LSTM model on extracted features."""
    logger.info("Starting model training...")
    
    # Update configuration with provided parameters
    config_override = {}
    if epochs:
        config_override['model.lstm.epochs'] = epochs
    if batch_size:
        config_override['model.lstm.batch_size'] = batch_size
    if learning_rate:
        config_override['model.lstm.learning_rate'] = learning_rate
    if n_classes:
        config_override['model.lstm.n_classes'] = n_classes
    
    try:
        # Load data
        logger.info("Loading features and labels...")
        X = np.load(features)
        y = np.load(labels)
        
        logger.info(f"Loaded features shape: {X.shape}")
        logger.info(f"Loaded labels shape: {y.shape}")
        
        # Initialize model
        model = LSTMModel(config_override)
        
        # Prepare data
        train_data, val_data, test_data = model.prepare_data(X, y)
        
        # Train model
        history = model.train(train_data, val_data, output)
        
        # Evaluate model
        results = model.evaluate(test_data)
        
        # Save results
        model.save_results(results, output)
        
        logger.info("Training completed successfully!")
        logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Test top-5 accuracy: {results['test_top5_accuracy']:.4f}")
        logger.info(f"Model and results saved to: {output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--video', '-v', required=True, type=click.Path(exists=True),
              help='Path to input video file')
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model file')
@click.option('--output', '-o', type=click.Path(),
              help='Path to output annotated video')
@click.option('--class-names', type=click.Path(exists=True),
              help='Path to class names file')
@click.option('--flownet-model', type=click.Path(exists=True),
              help='Path to FlowNet2 caffemodel file')
@click.option('--flownet-proto', type=click.Path(exists=True),
              help='Path to FlowNet2 deploy prototxt file')
@click.option('--gpu-id', type=int, default=0, help='GPU ID to use')
@click.option('--benchmark', is_flag=True, help='Run benchmark test')
def test_video(
    video: str,
    model: str,
    output: Optional[str],
    class_names: Optional[str],
    flownet_model: Optional[str],
    flownet_proto: Optional[str],
    gpu_id: int,
    benchmark: bool
):
    """Test activity recognition on a video file."""
    logger.info("Starting video testing...")
    
    # Update configuration
    config_override = {'model.flownet.gpu_id': gpu_id}
    if flownet_model:
        config_override['model.flownet.caffemodel'] = flownet_model
    if flownet_proto:
        config_override['model.flownet.deploy_proto'] = flownet_proto
    
    try:
        # Initialize video processor
        processor = VideoProcessor(
            model_path=model,
            class_names_file=class_names,
            config_override=config_override
        )
        
        if benchmark:
            # Run benchmark
            logger.info("Running benchmark...")
            stats = processor.benchmark_video(video)
            
            logger.info("Benchmark Results:")
            logger.info(f"  Total processing time: {stats['total_processing_time']:.2f}s")
            logger.info(f"  Video duration: {stats['video_duration']:.2f}s")
            logger.info(f"  Processing FPS: {stats['processing_fps']:.2f}")
            logger.info(f"  Real-time factor: {stats['real_time_factor']:.2f}x")
            logger.info(f"  Number of segments: {stats['num_segments']}")
            logger.info(f"  Average time per segment: {stats['avg_time_per_segment']:.3f}s")
        else:
            # Process video
            predictions = processor.process_video_file(video, output)
            
            logger.info("Video processing completed!")
            logger.info(f"Processed {len(predictions)} segments")
            
            # Show prediction summary
            if predictions:
                logger.info("\nPrediction Summary:")
                for i, pred in enumerate(predictions):
                    class_name = pred.get('predicted_class_name', 'Unknown')
                    confidence = pred.get('confidence', 0.0)
                    logger.info(f"  Segment {i+1}: {class_name} ({confidence:.3f})")
            
            if output:
                logger.info(f"Annotated video saved to: {output}")
        
    except ImportError:
        logger.error("Caffe is not available. Please install Caffe to use video testing.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Video testing failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--source', '-s', default=0, 
              help='Video source (0 for webcam, or path to video file)')
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to trained model file')
@click.option('--class-names', type=click.Path(exists=True),
              help='Path to class names file')
@click.option('--save', type=click.Path(), help='Path to save output video')
@click.option('--no-display', is_flag=True, help='Disable video display')
def stream(
    source,
    model: str,
    class_names: Optional[str],
    save: Optional[str],
    no_display: bool
):
    """Process video stream in real-time."""
    logger.info("Starting real-time stream processing...")
    
    try:
        # Convert source to int if it's a digit
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        # Initialize video processor
        processor = VideoProcessor(
            model_path=model,
            class_names_file=class_names
        )
        
        # Process stream
        processor.process_video_stream(
            source=source,
            display=not no_display,
            save_path=save
        )
        
    except KeyboardInterrupt:
        logger.info("Stream processing interrupted by user")
    except ImportError:
        logger.error("Caffe is not available. Please install Caffe to use stream processing.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Stream processing failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config-path', '-c', required=True, type=click.Path(),
              help='Path to save default configuration file')
def init_config(config_path: str):
    """Generate a default configuration file."""
    config.save_to_file(config_path)
    logger.info(f"Default configuration saved to: {config_path}")


@cli.command()
def info():
    """Show system and configuration information."""
    logger.info("Activity Recognition System Information")
    logger.info("=" * 50)
    
    # System info
    import tensorflow as tf
    import cv2
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    
    # GPU info
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        logger.info(f"  GPU {i}: {gpu}")
    
    # Caffe availability
    try:
        import caffe
        logger.info("Caffe: Available")
    except ImportError:
        logger.info("Caffe: Not available")
    
    # Current configuration
    logger.info("\nCurrent Configuration:")
    logger.info("-" * 30)
    for key, value in config.all.items():
        logger.info(f"{key}: {value}")


if __name__ == '__main__':
    cli()