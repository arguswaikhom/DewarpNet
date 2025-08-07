# TensorFlow DewarpNet Loss Functions

This directory contains the TensorFlow implementation of all loss functions used in the DewarpNet document unwarping system.

## Overview

The loss functions are designed to match the PyTorch implementation exactly, ensuring equivalent training behavior and performance. The implementation includes:

1. **Gradient Loss** - Uses Sobel filters for edge-based loss computation
2. **Reconstruction Loss** - Combines MSE and SSIM losses for image quality assessment
3. **SSIM Loss** - Structural Similarity Index for perceptual quality measurement
4. **Loss Factory** - Factory pattern for creating and managing loss functions
5. **Combined Loss** - Utility for combining multiple loss components with weights

## Components

### Core Loss Functions

#### GradLoss (`grad_loss.py`)
- Implements gradient-based loss using Sobel edge detection
- Supports configurable window sizes (3, 5, 7)
- Computes L1 loss between predicted and ground truth gradients
- Used for world coordinate network training

```python
from losses import GradLoss

grad_loss = GradLoss(window_size=3)
loss_value = grad_loss(y_true, y_pred)
```

#### ReconLoss (`recon_loss.py`)
- Implements reconstruction loss with image unwarping
- Combines MSE loss and optional SSIM loss
- Uses bilinear sampling for grid-based image transformation
- Used for backward mapping network training

```python
from losses import ReconLoss

recon_loss = ReconLoss(use_ssim=True, ssim_weight=1.0, mse_weight=1.0)
result = recon_loss(inputs, y_true, y_pred)
# Returns: {'total_loss', 'mse_loss', 'ssim_loss', 'uworg', 'uwpred'}
```

#### SSIMLoss (`ssim_loss.py`)
- Implements Structural Similarity Index loss
- Uses Gaussian windows for local similarity computation
- Configurable window size and averaging options
- Can be used standalone or as part of reconstruction loss

```python
from losses import SSIMLoss

ssim_loss = SSIMLoss(window_size=11)
loss_value = ssim_loss(y_true, y_pred)  # Returns 1 - SSIM
```

### Factory and Utilities

#### LossFactory (`loss_factory.py`)
- Factory pattern for creating loss functions
- Supports caching and configuration management
- Provides aliases for different loss types

```python
from losses import LossFactory

factory = LossFactory()
grad_loss = factory.create_loss('grad_loss', window_size=5)
recon_loss = factory.create_loss('recon_loss', use_ssim=False)
```

#### CombinedLoss
- Combines multiple loss functions with configurable weights
- Supports complex training scenarios with multiple objectives
- Returns detailed breakdown of loss components

```python
from losses import CombinedLoss

loss_configs = {
    'gradient': {'type': 'grad_loss', 'config': {'window_size': 3}},
    'reconstruction': {'type': 'recon_loss', 'config': {'use_ssim': True}}
}
weights = {'gradient': 1.0, 'reconstruction': 0.5}

combined_loss = CombinedLoss(loss_configs, weights)
result = combined_loss(*args, **kwargs)
```

#### LossLogger
- Utility for logging and monitoring loss values during training
- Supports configurable logging frequency
- Provides loss history and averaging capabilities

```python
from losses import LossLogger

logger = LossLogger(log_frequency=10)
logger.log_losses({'total_loss': loss_value}, step=epoch)
averages = logger.get_average_losses(last_n_steps=100)
```

### Convenience Functions

#### Pre-configured Loss Functions
```python
from losses import (
    create_dewarpnet_loss,
    create_world_coordinate_loss,
    create_backward_mapping_loss
)

# World coordinate training
wc_loss = create_world_coordinate_loss(window_size=3)

# Backward mapping training
bm_loss = create_backward_mapping_loss(grad_weight=1.0, recon_weight=0.5)

# Combined DewarpNet loss
dewarp_loss = create_dewarpnet_loss(grad_weight=1.0, recon_weight=1.0, ssim_weight=1.0)
```

## Usage Examples

### World Coordinate Training
```python
from losses import GradLoss

# Create gradient loss for world coordinate training
loss_fn = GradLoss(window_size=3, padding='SAME')

# During training
loss_value = loss_fn(ground_truth_coords, predicted_coords)
```

### Backward Mapping Training
```python
from losses import UnwarpLoss

# Create unwarp loss matching PyTorch implementation
loss_fn = UnwarpLoss()

# During training (inputs contain RGB + world coordinates)
mse_loss, ssim_loss, uworg, uwpred = loss_fn(inputs, predicted_mapping, ground_truth_mapping)
```

### Combined Training Pipeline
```python
from losses import create_dewarpnet_loss, LossLogger

# Create combined loss with custom weights
combined_loss = create_dewarpnet_loss(
    grad_weight=1.0,
    recon_weight=0.5,
    ssim_weight=1.0
)

# Create logger for monitoring
logger = LossLogger(log_frequency=10)

# Training loop
for epoch in range(num_epochs):
    # ... get batch data ...
    
    # Compute loss
    loss_result = combined_loss(inputs, y_true, y_pred)
    
    # Log losses
    logger.log_losses(loss_result, step=epoch)
    
    # Backpropagation
    # ... optimizer step ...
```

## Testing

All loss functions include comprehensive unit tests:

```bash
# Run all loss function tests
python tensorflow/tests/test_all_losses.py

# Run individual test suites
python tensorflow/tests/test_grad_loss.py
python tensorflow/tests/test_recon_loss.py
python tensorflow/tests/test_loss_factory.py
```

## Key Features

1. **PyTorch Compatibility** - Exact replication of PyTorch loss behavior
2. **GPU Acceleration** - Optimized for TensorFlow GPU execution
3. **Flexible Configuration** - Configurable parameters for different use cases
4. **Comprehensive Testing** - Full test coverage with validation against PyTorch
5. **Factory Pattern** - Easy creation and management of loss functions
6. **Logging Support** - Built-in utilities for training monitoring
7. **Type Safety** - Proper type hints and documentation

## Performance Considerations

- All loss functions are optimized for TensorFlow's eager execution
- GPU memory usage is minimized through efficient tensor operations
- Bilinear sampling uses optimized TensorFlow operations
- Loss computation is vectorized for batch processing

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **4.1**: Gradient loss with Sobel filters and multi-channel support
- **4.2**: Reconstruction loss with grid sampling and SSIM computation
- **4.3**: Loss function factory with selection, weighting, and logging utilities

All components have been tested and validated to ensure compatibility with the existing PyTorch implementation.