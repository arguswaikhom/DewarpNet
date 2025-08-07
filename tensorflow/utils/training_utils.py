"""
Training utilities for TensorFlow DewarpNet implementation.
Provides utilities for logging, visualization, and training management.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from datetime import datetime


class TrainingLogger:
    """Enhanced logging utility for training progress."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.txt")
        
        # Setup logging
        self.logger = logging.getLogger(f"trainer_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize log file
        self.write_header()
    
    def write_header(self):
        """Write experiment header to log file."""
        with open(self.log_file, 'a') as f:
            f.write(f'\n{"="*60}\n')
            f.write(f'Experiment: {self.experiment_name}\n')
            f.write(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'{"="*60}\n')
    
    def log_epoch(self, epoch: int, phase: str, losses: Dict[str, float], 
                  lr: float, additional_info: Optional[Dict[str, Any]] = None):
        """Log epoch results.
        
        Args:
            epoch: Current epoch number
            phase: Training phase ('Train' or 'Val')
            losses: Dictionary of loss values
            lr: Current learning rate
            additional_info: Additional information to log
        """
        # Format loss string
        loss_str = " ".join([f"{k}: {v:.6f}" for k, v in losses.items()])
        
        # Create log message
        log_msg = f"{phase} Epoch: {epoch} LRate: {lr:.8f} {loss_str}"
        
        if additional_info:
            info_str = " ".join([f"{k}: {v}" for k, v in additional_info.items()])
            log_msg += f" {info_str}"
        
        # Write to file and logger
        with open(self.log_file, 'a') as f:
            f.write(f"\n{log_msg}")
        
        self.logger.info(log_msg)
    
    def log_checkpoint(self, epoch: int, checkpoint_path: str, is_best: bool = False):
        """Log checkpoint saving.
        
        Args:
            epoch: Current epoch
            checkpoint_path: Path to saved checkpoint
            is_best: Whether this is the best model so far
        """
        status = "BEST" if is_best else "REGULAR"
        msg = f"Checkpoint saved [{status}] Epoch: {epoch} Path: {checkpoint_path}"
        
        with open(self.log_file, 'a') as f:
            f.write(f"\n{msg}")
        
        self.logger.info(msg)
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_str = json.dumps(config, indent=2)
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nTraining Configuration:\n{config_str}\n")
        
        self.logger.info(f"Training configuration logged")


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], count: int = 1):
        """Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric values
            count: Number of samples (for averaging)
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            
            self.metrics[name] += float(value) * count
            self.counts[name] += count
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics.
        
        Returns:
            Dictionary of average metric values
        """
        averages = {}
        for name in self.metrics:
            if self.counts[name] > 0:
                averages[name] = self.metrics[name] / self.counts[name]
            else:
                averages[name] = 0.0
        
        return averages
    
    def get_current(self) -> Dict[str, float]:
        """Get current accumulated values.
        
        Returns:
            Dictionary of current metric values
        """
        return dict(self.metrics)


class LearningRateScheduler:
    """Custom learning rate scheduler for TensorFlow."""
    
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, 
                 mode: str = 'min', factor: float = 0.5, 
                 patience: int = 5, min_lr: float = 1e-8,
                 verbose: bool = True):
        """Initialize learning rate scheduler.
        
        Args:
            optimizer: TensorFlow optimizer
            mode: 'min' or 'max' for monitoring metric
            factor: Factor to reduce learning rate by
            patience: Number of epochs to wait before reducing
            min_lr: Minimum learning rate
            verbose: Whether to print when reducing learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.logger = logging.getLogger(__name__)
    
    def step(self, metric: float) -> bool:
        """Step the scheduler with current metric value.
        
        Args:
            metric: Current metric value to monitor
            
        Returns:
            True if learning rate was reduced, False otherwise
        """
        improved = False
        
        if self.mode == 'min':
            improved = metric < self.best_metric
        else:
            improved = metric > self.best_metric
        
        if improved:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            current_lr = float(self.optimizer.learning_rate.numpy())
            new_lr = max(current_lr * self.factor, self.min_lr)
            
            if new_lr < current_lr:
                self.optimizer.learning_rate.assign(new_lr)
                self.wait = 0
                
                if self.verbose:
                    self.logger.info(
                        f"Reducing learning rate from {current_lr:.8f} to {new_lr:.8f}"
                    )
                
                return True
        
        return False


class CheckpointManager:
    """Manage model checkpoints with automatic cleanup."""
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load information about existing checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            return
        
        # Find all checkpoint files
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.ckpt.index'):
                checkpoint_path = os.path.join(self.checkpoint_dir, file[:-6])  # Remove .index
                if os.path.exists(f"{checkpoint_path}.data-00000-of-00001"):
                    checkpoint_files.append(checkpoint_path)
        
        # Sort by modification time
        checkpoint_files.sort(key=lambda x: os.path.getmtime(f"{x}.index"))
        
        self.checkpoints = checkpoint_files
    
    def save_checkpoint(self, checkpoint: tf.train.Checkpoint, 
                       filename: str, is_best: bool = False) -> str:
        """Save a checkpoint.
        
        Args:
            checkpoint: TensorFlow checkpoint object
            filename: Filename for the checkpoint
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Save checkpoint
        checkpoint.save(checkpoint_path)
        
        # Add to list if not best (best checkpoints are kept separately)
        if not is_best:
            self.checkpoints.append(checkpoint_path)
            
            # Clean up old checkpoints
            while len(self.checkpoints) > self.max_to_keep:
                old_checkpoint = self.checkpoints.pop(0)
                self._remove_checkpoint(old_checkpoint)
        
        return checkpoint_path
    
    def _remove_checkpoint(self, checkpoint_path: str):
        """Remove a checkpoint and its associated files.
        
        Args:
            checkpoint_path: Path to checkpoint to remove
        """
        # Remove all associated files
        for ext in ['.index', '.data-00000-of-00001']:
            file_path = f"{checkpoint_path}{ext}"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # Ignore errors
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if self.checkpoints:
            return self.checkpoints[-1]
        return None
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint paths
        """
        return list(self.checkpoints)


class TensorBoardLogger:
    """Enhanced TensorBoard logging utilities."""
    
    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
    
    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log scalar values.
        
        Args:
            scalars: Dictionary of scalar values
            step: Current step/epoch
        """
        with self.writer.as_default():
            for name, value in scalars.items():
                tf.summary.scalar(name, value, step=step)
            self.writer.flush()
    
    def log_images(self, images: tf.Tensor, name: str, step: int, 
                   max_outputs: int = 8):
        """Log images to TensorBoard.
        
        Args:
            images: Tensor of images [batch, height, width, channels]
            name: Name for the image summary
            step: Current step/epoch
            max_outputs: Maximum number of images to log
        """
        with self.writer.as_default():
            tf.summary.image(name, images, step=step, max_outputs=max_outputs)
            self.writer.flush()
    
    def log_histogram(self, values: tf.Tensor, name: str, step: int):
        """Log histogram of values.
        
        Args:
            values: Tensor of values
            name: Name for the histogram
            step: Current step/epoch
        """
        with self.writer.as_default():
            tf.summary.histogram(name, values, step=step)
            self.writer.flush()
    
    def log_model_weights(self, model: tf.keras.Model, step: int):
        """Log model weight histograms.
        
        Args:
            model: TensorFlow model
            step: Current step/epoch
        """
        with self.writer.as_default():
            for layer in model.layers:
                for weight in layer.weights:
                    tf.summary.histogram(f"weights/{weight.name}", weight, step=step)
            self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


def get_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: TensorFlow optimizer
        
    Returns:
        Current learning rate
    """
    return float(optimizer.learning_rate.numpy())


def create_experiment_name(arch: str, dataset: str, loss_type: str, 
                          augmentations: str, training_mode: str) -> str:
    """Create experiment name following PyTorch convention.
    
    Args:
        arch: Model architecture
        dataset: Dataset name
        loss_type: Loss function type
        augmentations: Augmentation strategy
        training_mode: Training mode (scratch, pretrained, etc.)
        
    Returns:
        Formatted experiment name
    """
    return f"{arch}_{dataset}_{loss_type}_{augmentations}_{training_mode}"


def setup_training_directories(base_dir: str, experiment_name: str) -> Dict[str, str]:
    """Setup training directories.
    
    Args:
        base_dir: Base directory for training outputs
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        'base': base_dir,
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'tensorboard': os.path.join(base_dir, 'tensorboard', experiment_name),
        'outputs': os.path.join(base_dir, 'outputs')
    }
    
    # Create directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def save_training_config(config: Dict[str, Any], save_path: str):
    """Save training configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)  # Test if serializable
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for monitoring metric
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, metric: float, model: tf.keras.Model) -> bool:
        """Check if training should stop.
        
        Args:
            metric: Current metric value
            model: Model to potentially restore weights for
            
        Returns:
            True if training should stop, False otherwise
        """
        improved = False
        
        if self.mode == 'min':
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)
        
        if improved:
            self.best_metric = metric
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.get_weights()
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.set_weights(self.best_weights)
                self.logger.info("Restored best weights")
            
            self.logger.info(f"Early stopping triggered after {self.patience} epochs")
            return True
        
        return False


def compute_model_flops(model: tf.keras.Model, input_shape: Tuple[int, ...]) -> int:
    """Compute approximate FLOPs for a model.
    
    Args:
        model: TensorFlow model
        input_shape: Input shape (without batch dimension)
        
    Returns:
        Approximate number of FLOPs
    """
    # This is a simplified FLOP calculation
    # For more accurate results, use tf.profiler or specialized tools
    
    total_flops = 0
    
    # Build model if not built
    if not model.built:
        dummy_input = tf.random.normal((1,) + input_shape)
        _ = model(dummy_input)
    
    for layer in model.layers:
        if hasattr(layer, 'kernel_size') and hasattr(layer, 'filters'):
            # Convolutional layer
            if hasattr(layer, 'input_shape') and layer.input_shape:
                input_h, input_w = layer.input_shape[1:3]
                kernel_h, kernel_w = layer.kernel_size
                in_channels = layer.input_shape[-1]
                out_channels = layer.filters
                
                # Approximate FLOPs for convolution
                flops = input_h * input_w * kernel_h * kernel_w * in_channels * out_channels
                total_flops += flops
        
        elif hasattr(layer, 'units'):
            # Dense layer
            if hasattr(layer, 'input_shape') and layer.input_shape:
                input_size = layer.input_shape[-1]
                output_size = layer.units
                
                # FLOPs for matrix multiplication
                flops = input_size * output_size
                total_flops += flops
    
    return total_flops