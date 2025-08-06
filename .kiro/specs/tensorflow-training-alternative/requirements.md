# Requirements Document

## Introduction

This feature involves creating a TensorFlow alternative to the existing PyTorch training scripts for the DewarpNet model. The project currently has two PyTorch training scripts: `trainwc.py` for world coordinate regression and `trainbm.py` for backward mapping regression. The goal is to replicate the exact same training behavior, model architectures, loss functions, and data processing pipeline in TensorFlow, with the ability to train on a subset (100 samples) first for validation, then scale to the full dataset.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to train the DewarpNet model using TensorFlow instead of PyTorch, so that I can leverage TensorFlow's ecosystem and deployment capabilities while maintaining identical training behavior.

#### Acceptance Criteria

1. WHEN the TensorFlow training script is executed THEN the system SHALL produce identical model architectures to the PyTorch versions (dnetccnl and unetnc)
2. WHEN training is performed THEN the system SHALL use the same loss functions (L1, MSE, gradient loss, reconstruction loss, SSIM) as the PyTorch implementation
3. WHEN data is loaded THEN the system SHALL apply identical preprocessing, normalization, and augmentation as the PyTorch data loaders
4. WHEN training progresses THEN the system SHALL log identical metrics and save checkpoints in a compatible format

### Requirement 2

**User Story:** As a researcher, I want to validate the TensorFlow implementation with a subset of data first, so that I can ensure correctness before committing to full dataset training.

#### Acceptance Criteria

1. WHEN the training script is configured for subset training THEN the system SHALL train on exactly 100 samples from the dataset
2. WHEN subset training completes successfully THEN the system SHALL provide clear validation that the implementation is working correctly
3. WHEN switching from subset to full dataset THEN the system SHALL require minimal configuration changes
4. WHEN training on subset data THEN the system SHALL maintain all training features (tensorboard logging, checkpointing, validation)

### Requirement 3

**User Story:** As a researcher, I want the TensorFlow training to support both world coordinate regression and backward mapping training modes, so that I can train both components of the DewarpNet pipeline.

#### Acceptance Criteria

1. WHEN training world coordinate regression THEN the system SHALL use the doc3dwc data loader equivalent and unetnc model architecture
2. WHEN training backward mapping THEN the system SHALL use the doc3dbmnic data loader equivalent and dnetccnl model architecture
3. WHEN either training mode is selected THEN the system SHALL apply the correct loss functions and optimization parameters
4. WHEN training completes THEN the system SHALL save models in a format compatible with the inference pipeline

### Requirement 4

**User Story:** As a researcher, I want the TensorFlow implementation to maintain identical hyperparameters and training configurations, so that results are directly comparable to the PyTorch version.

#### Acceptance Criteria

1. WHEN training is initialized THEN the system SHALL use identical learning rates, batch sizes, and epoch counts as the PyTorch scripts
2. WHEN optimization is performed THEN the system SHALL use Adam optimizer with identical parameters (lr=1e-5, weight_decay=5e-4, amsgrad=True)
3. WHEN learning rate scheduling is applied THEN the system SHALL use ReduceLROnPlateau with identical parameters (factor=0.5, patience=3/5)
4. WHEN model checkpointing occurs THEN the system SHALL save best models based on validation MSE and periodic checkpoints every 10 epochs

### Requirement 5

**User Story:** As a researcher, I want clear data organization guidelines and folder structure, so that I can properly set up the doc3d dataset for training both world coordinate and backward mapping models.

#### Acceptance Criteria

1. WHEN setting up the dataset THEN the system SHALL provide clear documentation of the required folder structure for doc3d dataset
2. WHEN organizing training data THEN the system SHALL specify the exact file formats and naming conventions required (PNG for images, EXR for world coordinates, MAT for backward mapping)
3. WHEN preparing data splits THEN the system SHALL define the format and location of train.txt and val.txt files containing sample identifiers
4. WHEN setting up for subset training THEN the system SHALL provide instructions for creating a 100-sample subset while maintaining the same folder structure
5. WHEN configuring data paths THEN the system SHALL document all required directories: img/, wc/, bm/, recon/, and texture files for augmentation

### Requirement 6

**User Story:** As a researcher, I want to convert the trained TensorFlow models to TensorFlow Lite format with different optimization levels, so that I can deploy the models on mobile and edge devices with varying performance requirements.

#### Acceptance Criteria

1. WHEN model training completes THEN the system SHALL provide functionality to convert the trained model to TensorFlow Lite format
2. WHEN creating the unoptimized TFLite model THEN the system SHALL preserve maximum quality with no quantization or optimization applied
3. WHEN creating the first optimized TFLite model THEN the system SHALL apply dynamic range quantization to reduce model size while maintaining reasonable quality
4. WHEN creating the second optimized TFLite model THEN the system SHALL apply full integer quantization for maximum size reduction and fastest inference
5. WHEN generating TFLite models THEN the system SHALL validate that each model produces functionally equivalent outputs to the original TensorFlow model
6. WHEN TFLite conversion completes THEN the system SHALL provide model size comparisons and performance benchmarks for each optimization level

### Requirement 7

**User Story:** As a researcher, I want comprehensive logging and visualization capabilities, so that I can monitor training progress and compare with PyTorch results.

#### Acceptance Criteria

1. WHEN training progresses THEN the system SHALL log training and validation losses to both console and file
2. WHEN tensorboard logging is enabled THEN the system SHALL visualize loss curves, learning rates, and sample predictions
3. WHEN training completes THEN the system SHALL generate experiment logs with identical naming conventions
4. WHEN validation is performed THEN the system SHALL display unwarp visualizations and world coordinate predictions similar to PyTorch implementation