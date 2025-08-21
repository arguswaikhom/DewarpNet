"""
TensorFlow utility functions for DewarpNet training.
Equivalent to PyTorch utils.py
"""

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Please install TensorFlow 2.x")
    import sys
    sys.exit(1)

import os
import numpy as np
from typing import Dict, Any


def get_lr_tf(optimizer):
    """
    Extract current learning rate from TensorFlow optimizer.
    
    Args:
        optimizer: TensorFlow optimizer
        
    Returns:
        Current learning rate as float
    """
    if hasattr(optimizer, '_decayed_lr'):
        return float(optimizer._decayed_lr(tf.float32).numpy())
    elif hasattr(optimizer, 'learning_rate'):
        lr = optimizer.learning_rate
        if callable(lr):
            return float(lr(optimizer.iterations).numpy())
        else:
            return float(lr.numpy())
    else:
        return 0.0


def show_wc_tnsboard_tf(step: int, images: tf.Tensor, labels: tf.Tensor, predictions: tf.Tensor,
                       max_images: int = 8, input_name: str = 'Inputs', 
                       label_name: str = 'Labels', pred_name: str = 'Predictions'):
    """
    Log world coordinate images to TensorBoard.
    
    Args:
        step: Current training step
        images: Input RGB images
        labels: Ground truth world coordinates
        predictions: Predicted world coordinates
        max_images: Maximum number of images to log
        input_name: Name for input images
        label_name: Name for label images
        pred_name: Name for prediction images
    """
    # Limit number of images
    batch_size = tf.shape(images)[0]
    num_images = tf.minimum(batch_size, max_images)
    
    # Select subset of images
    images_subset = images[:num_images]
    labels_subset = labels[:num_images]
    predictions_subset = predictions[:num_images]
    
    # Convert BGR to RGB for display (images are in BGR format)
    images_rgb = images_subset[..., ::-1]
    
    # Log to TensorBoard
    with tf.name_scope("WC_Visualization"):
        tf.summary.image(input_name, images_rgb, step=step, max_outputs=num_images)
        tf.summary.image(label_name, labels_subset, step=step, max_outputs=num_images)
        tf.summary.image(pred_name, predictions_subset, step=step, max_outputs=num_images)


def show_unwarp_tnsboard_tf(step: int, unwarp_pred: tf.Tensor, unwarp_gt: tf.Tensor,
                           max_images: int = 8, gt_name: str = 'GT Unwarp', 
                           pred_name: str = 'Pred Unwarp'):
    """
    Log unwarped images to TensorBoard.
    
    Args:
        step: Current training step
        unwarp_pred: Predicted unwarped images
        unwarp_gt: Ground truth unwarped images
        max_images: Maximum number of images to log
        gt_name: Name for ground truth images
        pred_name: Name for prediction images
    """
    # Limit number of images
    batch_size = tf.shape(unwarp_pred)[0]
    num_images = tf.minimum(batch_size, max_images)
    
    # Select subset of images
    unwarp_pred_subset = unwarp_pred[:num_images]
    unwarp_gt_subset = unwarp_gt[:num_images]
    
    # Ensure values are in [0, 1] range for TensorBoard
    unwarp_pred_subset = tf.clip_by_value(unwarp_pred_subset, 0.0, 1.0)
    unwarp_gt_subset = tf.clip_by_value(unwarp_gt_subset, 0.0, 1.0)
    
    # Log to TensorBoard
    with tf.name_scope("BM_Visualization"):
        tf.summary.image(gt_name, unwarp_gt_subset, step=step, max_outputs=num_images)
        tf.summary.image(pred_name, unwarp_pred_subset, step=step, max_outputs=num_images)


def convert_checkpoint_tf(checkpoint_path: str, target_path: str):
    """
    Convert PyTorch checkpoint to TensorFlow format.
    Note: This is a placeholder for future implementation.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        target_path: Path to save TensorFlow checkpoint
    """
    raise NotImplementedError("PyTorch to TensorFlow checkpoint conversion not implemented")


def save_tf_checkpoint(model, optimizer, epoch: int, val_loss: float, train_loss: float,
                      save_path: str, additional_state: Dict[str, Any] = None):
    """
    Save TensorFlow checkpoint with model, optimizer, and training state.
    
    Args:
        model: TensorFlow model
        optimizer: TensorFlow optimizer
        epoch: Current epoch
        val_loss: Validation loss
        train_loss: Training loss
        save_path: Path to save checkpoint
        additional_state: Additional state variables to save
    """
    # Create checkpoint object
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=tf.Variable(epoch),
        val_loss=tf.Variable(val_loss),
        train_loss=tf.Variable(train_loss)
    )
    
    # Add additional state if provided
    if additional_state:
        for key, value in additional_state.items():
            setattr(checkpoint, key, tf.Variable(value))
    
    # Save checkpoint
    checkpoint.save(save_path)
    return save_path


def load_tf_checkpoint(model, optimizer, checkpoint_path: str):
    """
    Load TensorFlow checkpoint.
    
    Args:
        model: TensorFlow model
        optimizer: TensorFlow optimizer
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dictionary with loaded state (epoch, val_loss, train_loss, etc.)
    """
    # Create checkpoint object
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=tf.Variable(0),
        val_loss=tf.Variable(0.0),
        train_loss=tf.Variable(0.0)
    )
    
    # Restore checkpoint
    status = checkpoint.restore(checkpoint_path)
    
    # Extract state
    state = {
        'epoch': int(checkpoint.epoch.numpy()),
        'val_loss': float(checkpoint.val_loss.numpy()),
        'train_loss': float(checkpoint.train_loss.numpy())
    }
    
    return state


def create_experiment_name(activation: str, dataset: str, loss_params: str, 
                          augmentations: str, train_start: str) -> str:
    """
    Create experiment name following the pattern from PyTorch version.
    
    Args:
        activation: Activation function name
        dataset: Dataset name
        loss_params: Loss function parameters
        augmentations: Augmentation description
        train_start: Training start description
        
    Returns:
        Experiment name string
    """
    return f"{activation}_{dataset}_{loss_params}_{augmentations}_{train_start}"


def setup_gpu_tf():
    """
    Setup GPU configuration for TensorFlow.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    else:
        print("No GPUs found, using CPU")
        return False


def print_model_summary(model, input_shape: tuple = None):
    """
    Print model summary with parameter count.
    
    Args:
        model: TensorFlow model
        input_shape: Input shape for the model
    """
    if input_shape:
        # Build model with specified input shape
        model.build(input_shape)
    
    model.summary()
    
    # Calculate total parameters
    total_params = sum([np.prod(var.shape) for var in model.trainable_variables])
    print(f"Total trainable parameters: {total_params:,}")


def create_callbacks(log_dir: str, save_dir: str, experiment_name: str, 
                    patience: int = 5, factor: float = 0.5, monitor: str = 'val_loss'):
    """
    Create common training callbacks.
    
    Args:
        log_dir: Directory for TensorBoard logs
        save_dir: Directory for saving checkpoints
        experiment_name: Name of the experiment
        patience: Patience for learning rate reduction
        factor: Factor for learning rate reduction
        monitor: Metric to monitor for LR reduction
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # TensorBoard callback
    if log_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, experiment_name),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
    
    # Model checkpoint callback
    if save_dir:
        checkpoint_path = os.path.join(save_dir, f"{experiment_name}_best.h5")
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
    
    # Learning rate reduction callback
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        verbose=1,
        mode='min'
    )
    callbacks.append(lr_callback)
    
    return callbacks


if __name__ == "__main__":
    # Test utility functions
    print("Testing TensorFlow utility functions...")
    
    # Test GPU setup
    gpu_available = setup_gpu_tf()
    print(f"GPU available: {gpu_available}")
    
    # Test model summary (would need actual model)
    print("✓ TensorFlow utility functions test passed!")