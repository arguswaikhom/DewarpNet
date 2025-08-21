"""
TensorFlow model factory for DewarpNet architectures.
Provides equivalent functionality to PyTorch models/__init__.py
"""

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Please install TensorFlow 2.x")
    import sys
    sys.exit(1)

from .unetnc_tf import create_unet_model, UnetGenerator
from .densenetccnl_tf import create_dnetccnl_model, DnetCCNL


def get_model(arch: str, n_classes: int, in_channels: int = 3, img_size: int = 256):
    """
    Get model architecture equivalent to PyTorch implementation.
    
    Args:
        arch: Architecture name ('unetnc_tf' or 'dnetccnl_tf')
        n_classes: Number of output classes/channels
        in_channels: Number of input channels
        img_size: Input image size
        
    Returns:
        TensorFlow/Keras model
    """
    
    if arch.lower() == 'unetnc' or arch.lower() == 'unetnc_tf':
        # UNet for world coordinate regression
        # Input: RGB image (256x256x3) -> Output: World coordinates (256x256x3)
        input_shape = (img_size, img_size, in_channels)
        model = create_unet_model(
            input_nc=in_channels,
            output_nc=n_classes,
            num_downs=7,  # 7 downsampling layers for 256x256 input
            ngf=64,
            input_shape=input_shape
        )
        return model
        
    elif arch.lower() == 'dnetccnl' or arch.lower() == 'dnetccnl_tf':
        # DenseNet for backward mapping
        # Input: Coordinate data (128x128x1) -> Output: Backward mapping (128x128x2)
        input_shape = (img_size, img_size, in_channels)
        model = create_dnetccnl_model(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=n_classes,
            filters=32,
            input_shape=input_shape
        )
        return model
        
    else:
        raise ValueError(f"Unknown architecture: {arch}. Supported: 'unetnc_tf', 'dnetccnl_tf'")


def load_pretrained_weights(model, checkpoint_path: str, convert_from_pytorch: bool = False):
    """
    Load weights from checkpoint.
    
    Args:
        model: TensorFlow model
        checkpoint_path: Path to checkpoint file
        convert_from_pytorch: Whether to convert from PyTorch state dict format
    """
    if convert_from_pytorch:
        # This would require additional conversion logic for PyTorch -> TF
        raise NotImplementedError("PyTorch to TensorFlow weight conversion not implemented")
    else:
        # Load TensorFlow checkpoint
        model.load_weights(checkpoint_path)
    
    return model


def save_model_checkpoint(model, optimizer, epoch: int, val_loss: float, train_loss: float, 
                         experiment_name: str, checkpoint_dir: str, is_best: bool = False):
    """
    Save model checkpoint in TensorFlow format.
    
    Args:
        model: TensorFlow model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        train_loss: Training loss  
        experiment_name: Name of experiment
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    import os
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Create checkpoint
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=tf.Variable(epoch),
        val_loss=tf.Variable(val_loss),
        train_loss=tf.Variable(train_loss)
    )
    
    # Save checkpoint
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{experiment_name}")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_{experiment_name}")
    
    checkpoint.save(checkpoint_path)
    
    # Also save the model architecture
    model_path = checkpoint_path + "_model"
    model.save_weights(model_path)
    
    return checkpoint_path


def restore_checkpoint(model, optimizer, checkpoint_path: str):
    """
    Restore model and optimizer from checkpoint.
    
    Args:
        model: TensorFlow model
        optimizer: Optimizer 
        checkpoint_path: Path to checkpoint
        
    Returns:
        Tuple of (epoch, val_loss, train_loss)
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
    checkpoint.restore(checkpoint_path)
    
    return int(checkpoint.epoch.numpy()), float(checkpoint.val_loss.numpy()), float(checkpoint.train_loss.numpy())


# Model registry for easy access
MODEL_REGISTRY = {
    'unetnc': {
        'class': UnetGenerator,
        'factory': create_unet_model,
        'default_params': {
            'input_nc': 3,
            'output_nc': 3,
            'num_downs': 7,
            'ngf': 64
        }
    },
    'dnetccnl': {
        'class': DnetCCNL,
        'factory': create_dnetccnl_model,
        'default_params': {
            'img_size': 128,
            'in_channels': 1,
            'out_channels': 2,
            'filters': 32
        }
    }
}


if __name__ == "__main__":
    # Test model creation
    print("Testing TensorFlow model factory...")
    
    # Test UNet
    print("\nTesting UNet model creation...")
    unet = get_model('unetnc_tf', n_classes=3, in_channels=3, img_size=256)
    print(f"UNet created: {unet.name}")
    
    # Test DenseNet
    print("\nTesting DenseNet model creation...")
    dnet = get_model('dnetccnl_tf', n_classes=2, in_channels=1, img_size=128)
    print(f"DenseNet created: {dnet.name}")
    
    print("✓ Model factory tests passed!")