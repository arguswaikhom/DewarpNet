# TensorFlow DewarpNet Environment Setup Guide

This guide provides comprehensive instructions for setting up the TensorFlow DewarpNet environment, including troubleshooting common issues.

## Quick Start

For most users, the automated setup script will handle everything:

```bash
# From the project root directory
python tensorflow/setup_env.py
```

This will:
1. Create a conda environment named `dewarpnet_tf_kiro_full`
2. Install all required dependencies
3. Set up dataset symbolic links
4. Validate the installation

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8 or higher (3.9 recommended)
- **Memory**: 16GB RAM recommended (8GB minimum)
- **Storage**: 100GB+ free space for datasets and models
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Required Software

1. **Conda/Miniconda**: Package and environment manager
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - Verify installation: `conda --version`

2. **NVIDIA Drivers** (for GPU support):
   - Linux: `sudo apt install nvidia-driver-470` (or latest)
   - Windows: Download from https://www.nvidia.com/drivers
   - Verify: `nvidia-smi`

3. **CUDA Toolkit** (handled automatically by conda):
   - Will be installed automatically via conda
   - Compatible versions: CUDA 11.2+ with cuDNN 8.1+

## Detailed Setup Instructions

### Step 1: Environment Creation

#### Automatic Setup (Recommended)
```bash
python tensorflow/setup_env.py
```

#### Manual Setup
```bash
# Create conda environment
conda create -n dewarpnet_tf_kiro_full python=3.9 -y

# Activate environment
conda activate dewarpnet_tf_kiro_full

# Install Python dependencies
pip install -r tensorflow/requirements_tf.txt

# Install CUDA support
conda install cudatoolkit cudnn -c conda-forge -y
```

### Step 2: Dataset Configuration

The setup script will automatically create symbolic links to your dataset. If you need to do this manually:

```bash
# Create data directory
mkdir -p tensorflow/data

# Create symbolic links (adjust paths as needed)
ln -s /home/argus/Workspace/dataset/doc3d tensorflow/data/doc3d
ln -s /home/argus/Workspace/dataset/doc3d_100 tensorflow/data/doc3d_100
ln -s /home/argus/Workspace/dataset/doc3d_1000 tensorflow/data/doc3d_1000
ln -s /home/argus/Workspace/dataset/input_crop tensorflow/data/input_crop
```

### Step 3: Validation

Validate your setup with the comprehensive validation script:

```bash
# Activate environment
conda activate dewarpnet_tf_kiro_full

# Run validation
python tensorflow/utils/env_validator.py
```

This will check:
- System requirements
- Conda installation
- Environment configuration
- TensorFlow installation
- GPU detection
- Dataset availability
- Disk space

### Step 4: GPU Testing

Test GPU functionality:

```bash
# Basic GPU test
python tensorflow/utils/gpu_utils.py

# Comprehensive GPU validation
python tensorflow/utils/gpu_utils.py --benchmark-only
```

## Environment Configuration Options

### Custom Environment Name

```bash
python tensorflow/setup_env.py --env-name my_custom_env
```

### Custom Dataset Path

```bash
python tensorflow/setup_env.py --dataset-path /path/to/my/dataset
```

### Skip Conda Environment Creation

```bash
python tensorflow/setup_env.py --skip-conda
```

### Different Python Version

```bash
python tensorflow/setup_env.py --python-version 3.10
```

## Troubleshooting

### Interactive Troubleshooting

For interactive problem diagnosis:

```bash
python tensorflow/utils/troubleshoot.py
```

### Automated Fixes

Run automated fixes for common issues:

```bash
python tensorflow/utils/troubleshoot.py --auto-fix
```

### Specific Issue Fixes

Target specific problems:

```bash
# Conda issues
python tensorflow/utils/troubleshoot.py --issue conda

# Environment issues
python tensorflow/utils/troubleshoot.py --issue env

# TensorFlow issues
python tensorflow/utils/troubleshoot.py --issue tensorflow

# GPU issues
python tensorflow/utils/troubleshoot.py --issue gpu
```

## Common Issues and Solutions

### 1. Conda Not Found

**Problem**: `conda: command not found`

**Solutions**:
- Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Add to PATH: `export PATH="$HOME/miniconda3/bin:$PATH"`
- Restart terminal

### 2. TensorFlow Import Error

**Problem**: `ImportError: No module named 'tensorflow'`

**Solutions**:
```bash
# Ensure correct environment is activated
conda activate dewarpnet_tf_kiro_full

# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow>=2.10.0

# Clear pip cache
pip cache purge
```

### 3. GPU Not Detected

**Problem**: TensorFlow cannot see GPU

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA support
conda install cudatoolkit cudnn -c conda-forge

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 4. CUDA Version Mismatch

**Problem**: CUDA version incompatibility

**Solutions**:
```bash
# Use conda to manage CUDA versions
conda install cudatoolkit=11.8 cudnn=8.6 -c conda-forge

# Or use tensorflow with bundled CUDA
pip install tensorflow[and-cuda]
```

### 5. Memory Errors

**Problem**: Out of memory during training

**Solutions**:
- Reduce batch size in training scripts
- Enable GPU memory growth
- Use mixed precision training
- Clear GPU memory between runs

### 6. Dataset Not Found

**Problem**: Dataset files not accessible

**Solutions**:
```bash
# Check dataset path
ls -la /home/argus/Workspace/dataset

# Recreate symbolic links
python tensorflow/setup_env.py --dataset-path /correct/path/to/dataset

# Fix permissions
sudo chown -R $USER:$USER /path/to/dataset
```

### 7. Permission Errors

**Problem**: Permission denied errors

**Solutions**:
```bash
# Fix conda permissions
sudo chown -R $USER:$USER ~/miniconda3

# Fix project permissions
sudo chown -R $USER:$USER /path/to/project

# Fix dataset permissions
chmod -R 755 /path/to/dataset
```

## Environment Testing

### Basic Functionality Test

```bash
python tensorflow/utils/test_setup.py
```

This tests:
- Directory structure
- Dataset links
- TensorFlow imports
- Basic operations
- Model creation
- Data pipeline

### Performance Benchmarking

```bash
python tensorflow/utils/gpu_utils.py --benchmark-only
```

### Integration Testing

```bash
# Run all tests
python tensorflow/tests/run_all_tests.py

# Run integration tests only
python tensorflow/tests/run_integration_tests.py
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# Enable GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Optimize CPU performance
export TF_CPP_MIN_LOG_LEVEL=2

# Enable mixed precision
export TF_ENABLE_AUTO_MIXED_PRECISION=1
```

Add to your `~/.bashrc` or `~/.zshrc` for persistence.

## Dataset Configuration

### Dataset Structure

Expected dataset structure:
```
/home/argus/Workspace/dataset/
├── doc3d/                 # Full Doc3D dataset
├── doc3d_100/            # 100 sample subset
├── doc3d_1000/           # 1000 sample subset
└── input_crop/           # Input crop images
```

### Dataset Size Selection

- **doc3d_100**: Quick testing and validation (~1GB)
- **doc3d_1000**: Medium-scale experiments (~10GB)
- **doc3d**: Full dataset for production training (~100GB)

### Creating Custom Dataset Subsets

```bash
# Create a custom subset
python -c "
import os
import shutil
from pathlib import Path

# Create doc3d_50 subset
src = Path('/home/argus/Workspace/dataset/doc3d')
dst = Path('/home/argus/Workspace/dataset/doc3d_50')
dst.mkdir(exist_ok=True)

# Copy first 50 samples
for i, file_path in enumerate(src.rglob('*')):
    if i >= 50:
        break
    if file_path.is_file():
        rel_path = file_path.relative_to(src)
        dst_path = dst / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dst_path)
"
```

## Performance Optimization

### GPU Memory Management

```python
import tensorflow as tf

# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Mixed Precision Training

```python
# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Data Pipeline Optimization

```python
# Optimize data loading
dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
```

## Maintenance

### Updating Dependencies

```bash
# Update all packages
conda activate dewarpnet_tf_kiro_full
pip install --upgrade -r tensorflow/requirements_tf.txt

# Update specific packages
pip install --upgrade tensorflow
conda update cudatoolkit cudnn
```

### Environment Cleanup

```bash
# Clean conda cache
conda clean --all

# Clean pip cache
pip cache purge

# Remove unused environments
conda env remove -n old_environment_name
```

### Backup and Restore

```bash
# Export environment
conda env export -n dewarpnet_tf_kiro_full > environment.yml

# Restore environment
conda env create -f environment.yml
```

## Getting Help

1. **Validation Issues**: Run `python tensorflow/utils/env_validator.py`
2. **Interactive Help**: Run `python tensorflow/utils/troubleshoot.py`
3. **Automated Fixes**: Run `python tensorflow/utils/troubleshoot.py --auto-fix`
4. **GPU Issues**: Run `python tensorflow/utils/gpu_utils.py`
5. **Setup Testing**: Run `python tensorflow/utils/test_setup.py`

For persistent issues, check the validation output and error logs for specific guidance.