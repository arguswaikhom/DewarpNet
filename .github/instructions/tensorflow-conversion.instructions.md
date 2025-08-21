---
applyTo: '**'
---

# TensorFlow Conversion Tasks

## Task 1: Model Architecture Conversion

### 1.1 Convert UNet Generator (`models/unetnc.py` → `models_tf/unetnc_tf.py`)
- Convert `UnetGenerator` class to TensorFlow/Keras model
- Preserve exact architecture: 7 downsampling layers, skip connections
- Key conversions:
  - `nn.Conv2d` → `tf.keras.layers.Conv2D`
  - `nn.ConvTranspose2d` → `tf.keras.layers.Conv2DTranspose`
  - `nn.BatchNorm2d` → `tf.keras.layers.BatchNormalization`
  - `nn.LeakyReLU(0.2)` → `tf.keras.layers.LeakyReLU(alpha=0.2)`
  - `nn.ReLU` → `tf.keras.layers.ReLU`
  - `nn.Tanh` → `tf.keras.layers.Activation('tanh')`
- Implement skip connections using `tf.keras.layers.Concatenate`
- Maintain exact filter counts: ngf=64, progression (64→128→256→512→512...)
- Preserve kernel_size=4, stride=2, padding=1 for all conv layers

### 1.2 Convert DenseNet (`models/densenetccnl.py` → `models_tf/densenetccnl_tf.py`)
- Convert `dnetccnl` class maintaining dense block structure
- Key components to convert:
  - `DenseBlockEncoder/Decoder` with residual connections
  - `DenseTransitionBlockEncoder/Decoder` with pooling/upsampling
  - `add_coordConv_channels()` function for coordinate convolution
  - `waspDenseEncoder128` and `waspDenseDecoder128`
- Preserve exact architecture:
  - Input: 128×128 with coordinate channels (nc+2)
  - Dense blocks: 6, 12, 16, 16, 16 convolutions per stage
  - Filter progression: ndf→2ndf→4ndf→8ndf→8ndf
  - Output: 2-channel backward mapping
- Convert activation functions and dropout layers identically

### 1.3 Model Factory (`models_tf/__init__.py`)
- Create TF equivalent of `get_model()` function
- Support architectures: 'unetnc_tf', 'dnetccnl_tf'
- Match exact parameter initialization and model instantiation

## Task 2: Loss Function Conversion

### 2.1 Gradient Loss (`grad_loss_tf.py`)
- Convert `Gradloss` class to TensorFlow
- Preserve exact Sobel kernel generation with multipliers:
  - Window size 3: mult=2
  - Window size 5: mult=20  
  - Window size 7: mult=780
- Convert convolution operations to `tf.nn.conv2d`
- Maintain L1 loss computation: `tf.reduce_mean(tf.abs(pred_grad - label_grad))`

### 2.2 Reconstruction Loss (`recon_lossc_tf.py`)
- Convert `Unwarploss` class preserving:
  - `unwarp()` function using `tf.nn.grid_sample` equivalent
  - MSE loss computation
  - SSIM loss integration
- Key conversions:
  - `F.grid_sample` → `tfa.image.resampler` or custom grid sampling
  - `F.upsample` → `tf.image.resize`
  - Coordinate normalization and tensor manipulations

### 2.3 SSIM Loss (`pytorch_ssim_tf/`)
- Convert SSIM implementation to TensorFlow
- Preserve exact Gaussian window generation
- Match convolution operations and statistical computations
- Maintain window_size=11, channels=3 defaults

## Task 3: Data Pipeline Conversion

### 3.1 Convert WC Data Loader (`loaders_tf/doc3dwc_loader_tf.py`)
- Convert `doc3dwcLoader` to `tf.data.Dataset`
- Preserve exact preprocessing:
  - Image resizing to 256×256
  - Label normalization with exact bounds: xmx=1.2539363, xmn=-1.2442188, etc.
  - BGR conversion and normalization (img/255.0)
  - NHWC→NCHW transpose operations
- Maintain augmentation pipeline from `augmentationsk.py`
- Support both train/val splits with identical file loading

### 3.2 Convert BM Data Loader (`loaders_tf/doc3dbmnoimgc_loader_tf.py`)
- Convert backward mapping loader maintaining:
  - 128×128 input size for coordinate data
  - Exact data path structure and file naming
  - EXR file reading for coordinate ground truth
  - PNG image loading and preprocessing

### 3.3 Augmentation Pipeline (`loaders_tf/augmentationsk_tf.py`)
- Convert augmentation functions to TensorFlow operations
- Preserve `data_aug()` and `tight_crop()` functionality
- Maintain exact random transformations and texture blending

## Task 4: Training Loop Conversion

### 4.1 WC Training (`trainwc_tf.py`)
- Convert training loop preserving:
  - Adam optimizer: lr=args.l_rate, weight_decay=5e-4, amsgrad=True
  - ReduceLROnPlateau scheduler: factor=0.5, patience=5
  - Loss combination: L1 + gradient loss
  - Hardtanh(0,1) activation for world coordinates
  - Exact experiment naming: 'htan_doc3d_l1grad_bghsaugk_scratch'
- Maintain TensorBoard logging with identical metrics
- Preserve checkpoint saving every 10 epochs and best model tracking

### 4.2 BM Training (`trainbm_tf.py`)
- Convert maintaining:
  - Adam optimizer: lr=0.0001, weight_decay=5e-4, amsgrad=True
  - ReduceLROnPlateau scheduler: factor=0.5, patience=3
  - Loss combination: (10.0*L1) + (0.5*reconstruction_loss)
  - Input handling: images[:,3:,:,:] for coordinate channels
  - Experiment naming: 'dnetccnl_htan_swat3dmini1kbm_l1_noaug_scratch'

### 4.3 Training Utilities (`utils_tf.py`)
- Convert utility functions:
  - `convert_state_dict()` → TF checkpoint handling
  - `show_wc_tnsboard()` → TF summary writing
  - Learning rate extraction and logging
  - Model saving/loading with proper state management

## Task 5: Inference Pipeline Conversion

### 5.1 Convert Inference (`infer_tf.py`)
- Convert end-to-end inference preserving:
  - Image preprocessing: resize to 256×256, BGR conversion, normalization
  - WC model prediction with Hardtanh(0,1) activation
  - Interpolation to 128×128 for BM model input
  - Grid sampling for final unwarping
  - cv2.blur smoothing operations
- Maintain command-line interface compatibility
- Preserve visualization options with matplotlib

### 5.2 Unwarp Function
- Convert `unwarp()` function maintaining:
  - Bilinear interpolation for backward mapping resize
  - Grid sample operations for final image warping
  - Exact coordinate transformations and tensor operations

## Task 6: Integration & Compatibility

### 6.1 Argument Parsing & Configuration
- Maintain identical command-line arguments across all scripts
- Preserve default values, help strings, and argument validation
- Ensure compatibility with existing data paths and model configurations

### 6.2 Logging & Monitoring
- Convert TensorBoard logging to TF format while preserving:
  - Loss tracking and visualization
  - Image/prediction displays
  - Learning rate monitoring
  - Experiment naming conventions

### 6.3 Model Checkpointing
- Implement TF equivalent of PyTorch checkpoint system:
  - Save model state, optimizer state, epoch information
  - Support resume functionality from checkpoints
  - Maintain best model tracking based on validation metrics

## Task 7: Validation & Testing

### 7.1 Numerical Equivalence Testing
- Create validation scripts comparing PyTorch vs TensorFlow:
  - Model output comparisons on identical inputs
  - Loss function value matching
  - Gradient computation verification
  - Training step-by-step validation

### 7.2 Performance Verification
- Ensure TF version achieves identical training/validation metrics
- Verify inference pipeline produces equivalent results
- Test with provided evaluation images in `eval/inp/`

## Key Implementation Notes

- **Exact Architecture Preservation**: All layer counts, filter sizes, and connections must match exactly
- **Loss Function Precision**: Custom loss implementations must produce identical values
- **Training Dynamics**: Optimizer settings, learning rate schedules, and batch processing must be identical
- **Data Pipeline**: Preprocessing, augmentations, and data loading must maintain exact same transformations
- **Coordinate Systems**: Pay special attention to tensor dimension orders (NCHW vs NHWC) and coordinate conventions
- **Activation Functions**: Preserve exact activation choices and parameters (Hardtanh bounds, LeakyReLU alpha values)
- **Skip Connections**: UNet skip connections and DenseNet residual paths must maintain identical connectivity
- **Testing input/output**: Never use generated input/output data. Always use real data for any testing.