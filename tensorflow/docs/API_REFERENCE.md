# API Reference

This document provides detailed API documentation for the TensorFlow DewarpNet implementation.

## Table of Contents

- [Models](#models)
- [Data Loaders](#data-loaders)
- [Loss Functions](#loss-functions)
- [Training Utilities](#training-utilities)
- [Inference Pipeline](#inference-pipeline)
- [Utility Functions](#utility-functions)

## Models

### UNet Generator

The UNet Generator is used for world coordinate prediction.

#### `models.unet_tf.UnetGenerator`

```python
class UnetGenerator(tf.keras.Model):
    """
    UNet-based generator for world coordinate prediction.
    
    Args:
        input_nc (int): Number of input channels (default: 3)
        output_nc (int): Number of output channels (default: 3)
        num_downs (int): Number of downsampling layers (default: 7)
        ngf (int): Number of generator filters in first conv layer (default: 64)
        norm_layer: Normalization layer type (default: BatchNormalization)
        use_dropout (bool): Whether to use dropout (default: False)
    """
```

**Methods:**

```python
def __init__(self, input_nc=3, output_nc=3, num_downs=7, ngf=64, 
             norm_layer=tf.keras.layers.BatchNormalization, use_dropout=False):
    """Initialize the UNet generator."""

def call(self, inputs, training=None):
    """
    Forward pass through the UNet.
    
    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels)
        training (bool): Whether in training mode
        
    Returns:
        tf.Tensor: Output tensor of shape (batch_size, height, width, output_nc)
    """

def get_config(self):
    """Get model configuration for serialization."""
```

**Example Usage:**

```python
import tensorflow as tf
from models.unet_tf import UnetGenerator

# Create UNet model
model = UnetGenerator(
    input_nc=3,
    output_nc=3,
    num_downs=7,
    ngf=64
)

# Build model
model.build(input_shape=(None, 256, 256, 3))

# Forward pass
input_tensor = tf.random.normal((1, 256, 256, 3))
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # (1, 256, 256, 3)
```

### UNet Skip Connection Block

#### `models.unet_tf.UnetSkipConnectionBlock`

```python
class UnetSkipConnectionBlock(tf.keras.layers.Layer):
    """
    UNet skip connection block with encoder-decoder structure.
    
    Args:
        outer_nc (int): Number of filters in outer conv layer
        inner_nc (int): Number of filters in inner conv layer
        input_nc (int): Number of input channels (default: None)
        submodule: Inner UNet block (default: None)
        outermost (bool): Whether this is the outermost block (default: False)
        innermost (bool): Whether this is the innermost block (default: False)
        norm_layer: Normalization layer type
        use_dropout (bool): Whether to use dropout (default: False)
    """
```

### DenseNet Encoder

#### `models.densenet_tf.waspDenseEncoder128`

```python
class waspDenseEncoder128(tf.keras.Model):
    """
    DenseNet encoder for backward mapping coordinate prediction.
    
    Args:
        coordconv (bool): Whether to use coordinate convolution (default: True)
        input_channels (int): Number of input channels (default: 6)
    """
```

**Methods:**

```python
def __init__(self, coordconv=True, input_channels=6):
    """Initialize the DenseNet encoder."""

def call(self, inputs, training=None):
    """
    Forward pass through the encoder.
    
    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels)
        training (bool): Whether in training mode
        
    Returns:
        list: List of feature maps at different scales
    """
```

### DenseNet Decoder

#### `models.densenet_tf.waspDenseDecoder128`

```python
class waspDenseDecoder128(tf.keras.Model):
    """
    DenseNet decoder for backward mapping coordinate prediction.
    
    Args:
        coordconv (bool): Whether to use coordinate convolution (default: True)
        output_channels (int): Number of output channels (default: 2)
    """
```

### Dense Block Components

#### `models.densenet_tf.DenseBlockEncoder`

```python
class DenseBlockEncoder(tf.keras.layers.Layer):
    """
    Dense block for encoder with growth rate and bottleneck layers.
    
    Args:
        num_layers (int): Number of dense layers
        growth_rate (int): Growth rate for dense connections (default: 32)
        bn_size (int): Bottleneck size multiplier (default: 4)
        drop_rate (float): Dropout rate (default: 0.0)
    """
```

#### `models.densenet_tf.DenseTransitionBlockEncoder`

```python
class DenseTransitionBlockEncoder(tf.keras.layers.Layer):
    """
    Transition block for encoder with compression and downsampling.
    
    Args:
        num_output_features (int): Number of output features
        drop_rate (float): Dropout rate (default: 0.0)
    """
```

### Model Factory

#### `models.model_factory.get_model`

```python
def get_model(model_type, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type (str): Type of model ('unet', 'densenet_encoder', 'densenet_decoder')
        **kwargs: Model-specific arguments
        
    Returns:
        tf.keras.Model: Initialized model
        
    Raises:
        ValueError: If model_type is not supported
    """
```

**Example Usage:**

```python
from models.model_factory import get_model

# Create UNet model
unet = get_model('unet', input_nc=3, output_nc=3, num_downs=7)

# Create DenseNet encoder
encoder = get_model('densenet_encoder', coordconv=True, input_channels=6)

# Create DenseNet decoder
decoder = get_model('densenet_decoder', coordconv=True, output_channels=2)
```

## Data Loaders

### World Coordinate Data Loader

#### `loaders.doc3d_wc_loader.Doc3DWCLoader`

```python
class Doc3DWCLoader:
    """
    Data loader for world coordinate training data.
    
    Args:
        root_path (str): Path to dataset root directory
        split (str): Dataset split ('train', 'val', 'test') (default: 'train')
        img_size (tuple): Target image size (height, width) (default: (256, 256))
        augmentations (bool): Whether to apply data augmentations (default: True)
        normalize (bool): Whether to normalize images (default: True)
    """
```

**Methods:**

```python
def __init__(self, root_path, split='train', img_size=(256, 256), 
             augmentations=True, normalize=True):
    """Initialize the data loader."""

def __len__(self):
    """Return the number of samples in the dataset."""

def __getitem__(self, index):
    """
    Get a single sample from the dataset.
    
    Args:
        index (int): Sample index
        
    Returns:
        tuple: (image, world_coordinates)
            - image (tf.Tensor): RGB image tensor of shape (H, W, 3)
            - world_coordinates (tf.Tensor): World coordinate tensor of shape (H, W, 3)
    """

def get_tf_dataset(self, batch_size=8, shuffle=True, num_parallel_calls=tf.data.AUTOTUNE):
    """
    Create a tf.data.Dataset for efficient data loading.
    
    Args:
        batch_size (int): Batch size (default: 8)
        shuffle (bool): Whether to shuffle the dataset (default: True)
        num_parallel_calls (int): Number of parallel calls for map operations
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
```

**Example Usage:**

```python
from loaders.doc3d_wc_loader import Doc3DWCLoader

# Create data loader
loader = Doc3DWCLoader(
    root_path='data/doc3d_100',
    split='train',
    img_size=(256, 256),
    augmentations=True
)

# Get tf.data.Dataset
dataset = loader.get_tf_dataset(batch_size=8, shuffle=True)

# Iterate through batches
for batch_images, batch_coords in dataset.take(1):
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch coordinates shape: {batch_coords.shape}")
```

### Backward Mapping Data Loader

#### `loaders.doc3d_bm_loader.Doc3DBMLoader`

```python
class Doc3DBMLoader:
    """
    Data loader for backward mapping training data.
    
    Args:
        root_path (str): Path to dataset root directory
        split (str): Dataset split ('train', 'val', 'test') (default: 'train')
        img_size (tuple): Target image size (height, width) (default: (128, 128))
        tight_crop (bool): Whether to apply tight cropping (default: True)
        normalize (bool): Whether to normalize inputs (default: True)
    """
```

**Methods:**

```python
def __init__(self, root_path, split='train', img_size=(128, 128), 
             tight_crop=True, normalize=True):
    """Initialize the data loader."""

def __len__(self):
    """Return the number of samples in the dataset."""

def __getitem__(self, index):
    """
    Get a single sample from the dataset.
    
    Args:
        index (int): Sample index
        
    Returns:
        tuple: (input_data, backward_mapping)
            - input_data (tf.Tensor): Concatenated albedo and world coordinates (H, W, 6)
            - backward_mapping (tf.Tensor): Backward mapping coordinates (H, W, 2)
    """

def apply_tight_crop(self, albedo, world_coords, bm_coords):
    """
    Apply tight cropping to remove background regions.
    
    Args:
        albedo (tf.Tensor): Albedo image
        world_coords (tf.Tensor): World coordinates
        bm_coords (tf.Tensor): Backward mapping coordinates
        
    Returns:
        tuple: Cropped (albedo, world_coords, bm_coords)
    """
```

### Data Augmentation

#### `loaders.augmentations.get_augmentation_pipeline`

```python
def get_augmentation_pipeline(img_size=(256, 256), training=True):
    """
    Get data augmentation pipeline.
    
    Args:
        img_size (tuple): Target image size (height, width)
        training (bool): Whether for training (applies augmentations) or validation
        
    Returns:
        callable: Augmentation function
    """
```

**Available Augmentations:**

- Random horizontal flip
- Random rotation (±15 degrees)
- Random brightness adjustment (±0.1)
- Random contrast adjustment (±0.1)
- Random saturation adjustment (±0.1)
- Gaussian noise addition

## Loss Functions

### Gradient Loss

#### `losses.grad_loss.GradientLoss`

```python
class GradientLoss(tf.keras.losses.Loss):
    """
    Gradient loss using Sobel filters for edge detection.
    
    Args:
        channels (int): Number of input channels (default: 3)
        device (str): Device to run on (default: 'GPU:0')
        reduction (str): Type of reduction to apply (default: 'auto')
        name (str): Name of the loss (default: 'gradient_loss')
    """
```

**Methods:**

```python
def __init__(self, channels=3, device='GPU:0', reduction='auto', name='gradient_loss'):
    """Initialize the gradient loss."""

def call(self, y_true, y_pred):
    """
    Compute gradient loss between predictions and targets.
    
    Args:
        y_true (tf.Tensor): Ground truth tensor
        y_pred (tf.Tensor): Predicted tensor
        
    Returns:
        tf.Tensor: Gradient loss value
    """

def get_sobel_filters(self, channels):
    """
    Create Sobel filters for gradient computation.
    
    Args:
        channels (int): Number of channels
        
    Returns:
        tuple: (sobel_x, sobel_y) filter tensors
    """
```

**Example Usage:**

```python
from losses.grad_loss import GradientLoss

# Create gradient loss
grad_loss = GradientLoss(channels=3)

# Compute loss
predictions = tf.random.normal((8, 256, 256, 3))
targets = tf.random.normal((8, 256, 256, 3))
loss_value = grad_loss(targets, predictions)
print(f"Gradient loss: {loss_value}")
```

### Reconstruction Loss

#### `losses.recon_loss.ReconstructionLoss`

```python
class ReconstructionLoss(tf.keras.losses.Loss):
    """
    Reconstruction loss using grid sampling for image unwarping.
    
    Args:
        interpolation (str): Interpolation method ('bilinear', 'nearest') (default: 'bilinear')
        padding_mode (str): Padding mode ('zeros', 'border') (default: 'border')
        reduction (str): Type of reduction to apply (default: 'auto')
        name (str): Name of the loss (default: 'reconstruction_loss')
    """
```

**Methods:**

```python
def __init__(self, interpolation='bilinear', padding_mode='border', 
             reduction='auto', name='reconstruction_loss'):
    """Initialize the reconstruction loss."""

def call(self, bm_coords, wc_coords, albedo, gt_albedo):
    """
    Compute reconstruction loss by unwarping images.
    
    Args:
        bm_coords (tf.Tensor): Backward mapping coordinates
        wc_coords (tf.Tensor): World coordinates
        albedo (tf.Tensor): Input albedo image
        gt_albedo (tf.Tensor): Ground truth albedo image
        
    Returns:
        tf.Tensor: Reconstruction loss value
    """

def grid_sample(self, input_tensor, grid, interpolation='bilinear', padding_mode='border'):
    """
    Sample input tensor using grid coordinates.
    
    Args:
        input_tensor (tf.Tensor): Input tensor to sample from
        grid (tf.Tensor): Sampling grid coordinates
        interpolation (str): Interpolation method
        padding_mode (str): Padding mode for out-of-bounds coordinates
        
    Returns:
        tf.Tensor: Sampled tensor
    """
```

### SSIM Loss

#### `losses.ssim_loss.SSIMLoss`

```python
class SSIMLoss(tf.keras.losses.Loss):
    """
    Structural Similarity Index (SSIM) loss for perceptual quality.
    
    Args:
        window_size (int): Size of the sliding window (default: 11)
        k1 (float): Algorithm parameter (default: 0.01)
        k2 (float): Algorithm parameter (default: 0.03)
        reduction (str): Type of reduction to apply (default: 'auto')
        name (str): Name of the loss (default: 'ssim_loss')
    """
```

**Methods:**

```python
def __init__(self, window_size=11, k1=0.01, k2=0.03, reduction='auto', name='ssim_loss'):
    """Initialize the SSIM loss."""

def call(self, y_true, y_pred):
    """
    Compute SSIM loss between predictions and targets.
    
    Args:
        y_true (tf.Tensor): Ground truth tensor
        y_pred (tf.Tensor): Predicted tensor
        
    Returns:
        tf.Tensor: SSIM loss value (1 - SSIM)
    """
```

## Training Utilities

### Training Manager

#### `training.utils.TrainingManager`

```python
class TrainingManager:
    """
    Manages training process including checkpointing, logging, and validation.
    
    Args:
        model (tf.keras.Model): Model to train
        optimizer (tf.keras.optimizers.Optimizer): Optimizer
        loss_fn (tf.keras.losses.Loss): Loss function
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset (optional)
        checkpoint_dir (str): Directory to save checkpoints (default: 'checkpoints')
        log_dir (str): Directory to save logs (default: 'logs')
    """
```

**Methods:**

```python
def __init__(self, model, optimizer, loss_fn, train_dataset, val_dataset=None,
             checkpoint_dir='checkpoints', log_dir='logs'):
    """Initialize the training manager."""

def train_epoch(self):
    """
    Train for one epoch.
    
    Returns:
        dict: Training metrics for the epoch
    """

def validate_epoch(self):
    """
    Validate for one epoch.
    
    Returns:
        dict: Validation metrics for the epoch
    """

def save_checkpoint(self, epoch, metrics, is_best=False):
    """
    Save model checkpoint.
    
    Args:
        epoch (int): Current epoch number
        metrics (dict): Current metrics
        is_best (bool): Whether this is the best model so far
    """

def load_checkpoint(self, checkpoint_path):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        dict: Loaded checkpoint information
    """

def train(self, num_epochs, save_freq=5, val_freq=1):
    """
    Full training loop.
    
    Args:
        num_epochs (int): Number of epochs to train
        save_freq (int): Frequency of checkpoint saving (epochs)
        val_freq (int): Frequency of validation (epochs)
        
    Returns:
        dict: Training history
    """
```

### Learning Rate Scheduler

#### `training.utils.ReduceLROnPlateau`

```python
class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Args:
        optimizer (tf.keras.optimizers.Optimizer): Optimizer to modify
        monitor (str): Metric to monitor (default: 'val_loss')
        factor (float): Factor by which to reduce LR (default: 0.5)
        patience (int): Number of epochs with no improvement to wait (default: 10)
        min_lr (float): Minimum learning rate (default: 1e-7)
        verbose (bool): Whether to print messages (default: True)
    """
```

## Inference Pipeline

### DewarpNet Inference

#### `inference.infer.DewarpNetInference`

```python
class DewarpNetInference:
    """
    Complete inference pipeline for document unwarping.
    
    Args:
        wc_model_path (str): Path to world coordinate model checkpoint
        bm_model_path (str): Path to backward mapping model checkpoint
        device (str): Device to run inference on (default: 'auto')
        input_size (tuple): Input size for WC model (default: (256, 256))
        output_size (tuple): Processing size for BM model (default: (128, 128))
        batch_size (int): Batch size for inference (default: 1)
    """
```

**Methods:**

```python
def __init__(self, wc_model_path, bm_model_path, device='auto', 
             input_size=(256, 256), output_size=(128, 128), batch_size=1):
    """Initialize the inference pipeline."""

def load_models(self):
    """Load the trained models from checkpoints."""

def preprocess_image(self, image):
    """
    Preprocess input image for inference.
    
    Args:
        image (PIL.Image or np.ndarray): Input image
        
    Returns:
        tf.Tensor: Preprocessed image tensor
    """

def postprocess_output(self, output):
    """
    Postprocess model output to final image.
    
    Args:
        output (tf.Tensor): Model output tensor
        
    Returns:
        PIL.Image: Final unwarped image
    """

def process_image(self, image_path, return_intermediate=False):
    """
    Process a single image through the complete pipeline.
    
    Args:
        image_path (str): Path to input image
        return_intermediate (bool): Whether to return intermediate results
        
    Returns:
        PIL.Image or tuple: Unwarped image, optionally with intermediate results
    """

def process_batch(self, image_paths):
    """
    Process a batch of images.
    
    Args:
        image_paths (list): List of image paths
        
    Returns:
        list: List of unwarped PIL Images
    """

def process_directory(self, input_dir, output_dir, file_pattern='*.png,*.jpg,*.jpeg',
                     save_intermediate=False, visualize=False):
    """
    Process all images in a directory.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        file_pattern (str): File patterns to match
        save_intermediate (bool): Whether to save intermediate results
        visualize (bool): Whether to create visualization images
    """

def save_image(self, image, output_path, quality=95):
    """
    Save processed image to file.
    
    Args:
        image (PIL.Image): Image to save
        output_path (str): Output file path
        quality (int): JPEG quality (1-100)
    """
```

**Example Usage:**

```python
from inference.infer import DewarpNetInference

# Initialize inference pipeline
dewarper = DewarpNetInference(
    wc_model_path='checkpoints/wc_best.pth',
    bm_model_path='checkpoints/bm_best.pth',
    device='cuda:0',
    batch_size=4
)

# Process single image
unwarped = dewarper.process_image('input.png')
dewarper.save_image(unwarped, 'output.png')

# Process directory
dewarper.process_directory(
    input_dir='input_images/',
    output_dir='output_images/',
    save_intermediate=True,
    visualize=True
)
```

## Utility Functions

### GPU Utilities

#### `utils.gpu_utils.detect_gpu`

```python
def detect_gpu():
    """
    Detect available GPU devices.
    
    Returns:
        list: List of available GPU device names
    """
```

#### `utils.gpu_utils.get_gpu_memory_info`

```python
def get_gpu_memory_info(device_id=0):
    """
    Get GPU memory information.
    
    Args:
        device_id (int): GPU device ID
        
    Returns:
        dict: Memory information including total, used, and free memory
    """
```

#### `utils.gpu_utils.set_memory_growth`

```python
def set_memory_growth(enable=True):
    """
    Enable or disable GPU memory growth.
    
    Args:
        enable (bool): Whether to enable memory growth
    """
```

### Visualization Utilities

#### `utils.visualization.plot_training_curves`

```python
def plot_training_curves(log_dir, metrics=['loss', 'val_loss'], save_path=None):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_dir (str): Directory containing TensorBoard logs
        metrics (list): List of metrics to plot
        save_path (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
```

#### `utils.visualization.visualize_unwarping_process`

```python
def visualize_unwarping_process(original_image, world_coords, backward_mapping, 
                               unwarped_image, save_path=None):
    """
    Create visualization of the unwarping process.
    
    Args:
        original_image (PIL.Image): Original input image
        world_coords (np.ndarray): Predicted world coordinates
        backward_mapping (np.ndarray): Predicted backward mapping
        unwarped_image (PIL.Image): Final unwarped image
        save_path (str): Path to save visualization (optional)
        
    Returns:
        PIL.Image: Visualization image
    """
```

### Model Utilities

#### `utils.model_utils.count_parameters`

```python
def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (tf.keras.Model): Model to analyze
        
    Returns:
        int: Number of trainable parameters
    """
```

#### `utils.model_utils.compare_models`

```python
def compare_models(model1, model2):
    """
    Compare two models for architectural equivalence.
    
    Args:
        model1 (tf.keras.Model): First model
        model2 (tf.keras.Model): Second model
        
    Returns:
        dict: Comparison results including parameter counts and layer differences
    """
```

### Checkpoint Utilities

#### `utils.checkpoint_utils.convert_pytorch_to_tensorflow`

```python
def convert_pytorch_to_tensorflow(pytorch_checkpoint_path, tensorflow_model, 
                                 output_path):
    """
    Convert PyTorch checkpoint to TensorFlow format.
    
    Args:
        pytorch_checkpoint_path (str): Path to PyTorch checkpoint
        tensorflow_model (tf.keras.Model): TensorFlow model to load weights into
        output_path (str): Path to save converted checkpoint
    """
```

#### `utils.checkpoint_utils.validate_checkpoint`

```python
def validate_checkpoint(checkpoint_path, model=None):
    """
    Validate checkpoint file integrity and compatibility.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (tf.keras.Model): Model to check compatibility with (optional)
        
    Returns:
        dict: Validation results
    """
```

This API reference provides comprehensive documentation for all major components of the TensorFlow DewarpNet implementation. Each class and function includes detailed parameter descriptions, return values, and usage examples.