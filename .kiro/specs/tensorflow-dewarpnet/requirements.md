# Requirements Document

## Introduction

This feature involves creating a complete TensorFlow implementation of the DewarpNet document unwarping system, which currently exists only in PyTorch. DewarpNet is a two-stage neural network system that performs document unwarping: first predicting world coordinates from RGB images, then performing backward mapping to unwarp the document. The TensorFlow implementation must replicate the exact training logic, model architectures, data loading, loss functions, and inference pipeline to achieve equivalent performance to the PyTorch version.

## Requirements

### Requirement 1

**User Story:** As a machine learning researcher, I want a TensorFlow implementation of DewarpNet training, so that I can leverage TensorFlow's ecosystem and potentially achieve better performance on my hardware setup.

#### Acceptance Criteria

1. WHEN the system is initialized THEN it SHALL create a dedicated tensorflow directory structure with all necessary components
2. WHEN training is started THEN the system SHALL use the same dataset structure as the PyTorch version (/home/argus/Workspace/dataset)
3. WHEN models are trained THEN they SHALL achieve comparable performance metrics to the PyTorch implementation
4. WHEN inference is performed THEN the TensorFlow models SHALL produce visually equivalent results to PyTorch models

### Requirement 2

**User Story:** As a developer, I want the TensorFlow implementation to have identical model architectures, so that the learned representations are equivalent between frameworks.

#### Acceptance Criteria

1. WHEN the UNet model is implemented THEN it SHALL have the same architecture as the PyTorch UnetGenerator with 7 downsampling layers
2. WHEN the DenseNet model is implemented THEN it SHALL replicate the exact dnetccnl architecture with coordconv channels
3. WHEN models are initialized THEN they SHALL have equivalent parameter counts to their PyTorch counterparts
4. WHEN forward passes are executed THEN the output shapes SHALL match the PyTorch implementation exactly

### Requirement 3

**User Story:** As a data scientist, I want the TensorFlow training pipeline to use the same data loading and augmentation strategies, so that training conditions are identical.

#### Acceptance Criteria

1. WHEN data is loaded THEN the system SHALL use the same normalization values and preprocessing steps as PyTorch
2. WHEN augmentations are applied THEN they SHALL replicate the exact augmentation pipeline from the PyTorch loaders
3. WHEN world coordinate data is processed THEN it SHALL use identical coordinate normalization ranges
4. WHEN backward mapping data is loaded THEN it SHALL apply the same tight cropping and coordinate transformations

### Requirement 4

**User Story:** As a researcher, I want the TensorFlow implementation to use equivalent loss functions and training procedures, so that convergence behavior matches the original implementation.

#### Acceptance Criteria

1. WHEN gradient loss is computed THEN it SHALL use the same Sobel filter implementation and weighting
2. WHEN reconstruction loss is calculated THEN it SHALL perform identical grid sampling and SSIM computation
3. WHEN training loops execute THEN they SHALL use the same optimizer settings, learning rate scheduling, and batch processing
4. WHEN validation is performed THEN it SHALL compute identical metrics and model checkpointing logic

### Requirement 5

**User Story:** As a system administrator, I want proper environment setup and dependency management, so that the TensorFlow implementation can be deployed reliably.

#### Acceptance Criteria

1. WHEN the environment is set up THEN it SHALL create a dedicated conda environment with TensorFlow GPU support
2. WHEN dependencies are installed THEN they SHALL be documented in a tensorflow-specific requirements file
3. WHEN GPU is available THEN the system SHALL automatically detect and utilize NVIDIA GPU acceleration
4. WHEN GPU is not detected THEN the system SHALL provide clear error messages and troubleshooting guidance

### Requirement 6

**User Story:** As a user, I want to test the implementation on smaller datasets first, so that I can validate correctness before full-scale training.

#### Acceptance Criteria

1. WHEN testing is initiated THEN the system SHALL default to using doc3d_100 dataset for initial validation
2. WHEN small dataset training completes THEN the system SHALL provide options to scale up to doc3d_1000 or full doc3d
3. WHEN dataset paths are configured THEN the system SHALL create symbolic links without modifying original data
4. WHEN training progress is monitored THEN the system SHALL display comprehensive logging and progress information

### Requirement 7

**User Story:** As an end user, I want to process images with the trained TensorFlow models, so that I can unwarp documents using the new implementation.

#### Acceptance Criteria

1. WHEN inference is performed THEN the system SHALL load both world coordinate and backward mapping TensorFlow models
2. WHEN images are processed THEN they SHALL be loaded from /home/argus/Workspace/dataset/input_crop directory
3. WHEN unwarping is applied THEN the output SHALL be visually equivalent to PyTorch model results
4. WHEN processing is complete THEN unwarped images SHALL be saved with appropriate file naming and format