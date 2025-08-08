# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the TensorFlow DewarpNet implementation.

## Quick Diagnostics

### System Health Check

```bash
# Run comprehensive system check
python tensorflow/utils/test_setup.py

# Check GPU availability
python tensorflow/utils/gpu_utils.py

# Validate environment
conda list | grep tensorflow
```

### Common Quick Fixes

```bash
# Restart conda environment
conda deactivate
conda activate dewarpnet_tf

# Clear TensorFlow cache
python -c "import tensorflow as tf; tf.keras.backend.clear_session()"

# Reset GPU memory
nvidia-smi --gpu-reset
```

## Installation and Setup Issues

### 1. Environment Setup Failures

#### Problem: Conda environment creation fails
```
CondaError: Could not create environment
```

**Solutions:**
```bash
# Update conda
conda update conda

# Clean conda cache
conda clean --all

# Use mamba instead (faster)
conda install mamba -n base -c conda-forge
mamba env create -f tensorflow/environment.yml

# Manual environment creation
conda create -n dewarpnet_tf python=3.9
conda activate dewarpnet_tf
pip install -r tensorflow/requirements_tf.txt
```

#### Problem: TensorFlow GPU not detected
```
Could not load dynamic library 'libcudart.so.11.0'
```

**Solutions:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall CUDA-compatible TensorFlow
pip uninstall tensorflow
pip install tensorflow-gpu==2.10.0

# Install CUDA toolkit via conda
conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge

# Verify installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Problem: Dataset symbolic links not created
```
FileNotFoundError: Dataset directory not found
```

**Solutions:**
```bash
# Check dataset path exists
ls -la /home/argus/Workspace/dataset/

# Manually create symbolic links
cd tensorflow/data/
ln -sf /home/argus/Workspace/dataset/doc3d ./doc3d
ln -sf /home/argus/Workspace/dataset/doc3d_100 ./doc3d_100
ln -sf /home/argus/Workspace/dataset/doc3d_1000 ./doc3d_1000
ln -sf /home/argus/Workspace/dataset/input_crop ./input_crop

# Re-run setup with custom path
python tensorflow/setup_env.py --dataset-path /your/custom/path
```

### 2. Dependency Issues

#### Problem: Import errors
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions:**
```bash
# Verify environment activation
conda info --envs
conda activate dewarpnet_tf

# Reinstall dependencies
pip install -r tensorflow/requirements_tf.txt

# Check Python path
python -c "import sys; print(sys.path)"

# Install missing packages individually
pip install tensorflow-gpu==2.10.0
pip install opencv-python==4.5.5.64
pip install pillow==8.4.0
```

#### Problem: Version conflicts
```
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions:**
```bash
# Create fresh environment
conda deactivate
conda env remove -n dewarpnet_tf
conda create -n dewarpnet_tf python=3.9

# Install with specific versions
pip install tensorflow-gpu==2.10.0 --no-deps
pip install numpy==1.21.6
pip install opencv-python==4.5.5.64

# Use conda for problematic packages
conda install numpy opencv pillow matplotlib -c conda-forge
```

## Training Issues

### 1. Memory Issues

#### Problem: GPU out of memory
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**
```bash
# Reduce batch size
python training/train_wc.py --batch_size 2

# Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Use gradient checkpointing
python training/train_wc.py --gradient_checkpointing

# Monitor memory usage
watch -n 1 nvidia-smi
```

#### Problem: System RAM exhaustion
```
MemoryError: Unable to allocate array
```

**Solutions:**
```bash
# Reduce number of data loading workers
python training/train_wc.py --num_workers 2

# Use smaller dataset for testing
python training/train_wc.py --dataset doc3d_100

# Enable data streaming
python training/train_wc.py --stream_data

# Monitor system memory
htop
```

### 2. Training Performance Issues

#### Problem: Very slow training
```
Training speed: 0.1 it/s (expected: 2-3 it/s)
```

**Solutions:**
```bash
# Check GPU utilization
nvidia-smi

# Increase batch size
python training/train_wc.py --batch_size 8

# Enable mixed precision
python training/train_wc.py --mixed_precision

# Increase data loading workers
python training/train_wc.py --num_workers 8

# Profile training
python training/train_wc.py --profile
```

#### Problem: Loss not decreasing
```
Training loss stuck at initial value
```

**Solutions:**
```bash
# Check learning rate
python training/train_wc.py --learning_rate 0.001

# Verify data loading
python training/train_wc.py --test_data_only

# Check model initialization
python training/train_wc.py --validate_model_only

# Enable debug mode
python training/train_wc.py --debug

# Try different optimizer
python training/train_wc.py --optimizer sgd
```

### 3. Data Loading Issues

#### Problem: Data loader crashes
```
RuntimeError: DataLoader worker (pid X) is killed by signal
```

**Solutions:**
```bash
# Reduce number of workers
python training/train_wc.py --num_workers 1

# Check data integrity
python utils/validate_dataset.py --dataset doc3d_100

# Use single-threaded loading
python training/train_wc.py --num_workers 0

# Check system limits
ulimit -n
# Increase if needed: ulimit -n 4096
```

#### Problem: Corrupted data files
```
Error loading image: Truncated or corrupted file
```

**Solutions:**
```bash
# Validate dataset integrity
python utils/validate_dataset.py --dataset doc3d_100 --fix_corrupted

# Skip corrupted files
python training/train_wc.py --skip_corrupted

# Re-download dataset
# (Follow dataset preparation instructions)

# Check disk space
df -h
```

## Model and Inference Issues

### 1. Model Loading Issues

#### Problem: Checkpoint loading fails
```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**
```bash
# Check checkpoint file
python utils/validate_checkpoint.py --model_path checkpoints/wc_best.pth

# Use strict=False loading
python inference/infer.py --strict_loading false

# Convert checkpoint format
python utils/convert_checkpoint.py --input old_checkpoint.pth --output new_checkpoint.pth

# Check model architecture compatibility
python utils/compare_models.py --checkpoint checkpoints/wc_best.pth
```

#### Problem: Model architecture mismatch
```
RuntimeError: size mismatch for layer.weight
```

**Solutions:**
```bash
# Check model configuration
python utils/inspect_checkpoint.py --model_path checkpoints/wc_best.pth

# Use correct model architecture
python inference/infer.py --model_type unet_v2

# Rebuild model with correct parameters
python utils/rebuild_checkpoint.py --input checkpoints/wc_best.pth
```

### 2. Inference Quality Issues

#### Problem: Poor unwarping quality
```
Output images are blurry or distorted
```

**Solutions:**
```bash
# Check input image quality
python utils/analyze_image.py --image input.png

# Use higher resolution
python inference/infer.py --input_size 512

# Enable enhanced postprocessing
python inference/infer.py --postprocessing enhanced

# Check model performance
python utils/benchmark_model.py --model_path checkpoints/wc_best.pth

# Verify training convergence
python utils/plot_training_curves.py --log_dir logs/
```

#### Problem: Inference crashes
```
Segmentation fault during inference
```

**Solutions:**
```bash
# Use CPU inference
python inference/infer.py --device cpu

# Reduce batch size
python inference/infer.py --batch_size 1

# Check GPU memory
nvidia-smi

# Update GPU drivers
sudo apt update && sudo apt install nvidia-driver-470

# Use debug mode
python inference/infer.py --debug
```

## Performance Issues

### 1. GPU Utilization Problems

#### Problem: Low GPU utilization
```
GPU utilization: 20% (expected: 80-90%)
```

**Solutions:**
```bash
# Increase batch size
python training/train_wc.py --batch_size 16

# Check data loading bottleneck
python training/train_wc.py --profile_data_loading

# Increase number of workers
python training/train_wc.py --num_workers 8

# Enable prefetching
python training/train_wc.py --prefetch_factor 4

# Check CPU bottleneck
htop
```

#### Problem: GPU memory not fully utilized
```
GPU memory usage: 2GB/12GB
```

**Solutions:**
```bash
# Increase batch size gradually
python training/train_wc.py --batch_size 8
python training/train_wc.py --batch_size 12
python training/train_wc.py --batch_size 16

# Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=false

# Use larger model if available
python training/train_wc.py --model_size large
```

### 2. Training Speed Issues

#### Problem: Training slower than expected
```
Expected: 2-3 it/s, Actual: 0.5 it/s
```

**Solutions:**
```bash
# Profile training loop
python training/train_wc.py --profile

# Check I/O bottleneck
iotop

# Use SSD for dataset storage
# Move dataset to faster storage

# Enable mixed precision
python training/train_wc.py --mixed_precision

# Optimize data pipeline
python training/train_wc.py --optimize_data_pipeline
```

## Environment-Specific Issues

### 1. Windows-Specific Issues

#### Problem: Path separator issues
```
FileNotFoundError: [Errno 2] No such file or directory: 'data\doc3d\img\001.png'
```

**Solutions:**
```bash
# Use forward slashes in paths
python training/train_wc.py --dataset_path data/doc3d

# Set environment variable
set PYTHONPATH=%PYTHONPATH%;tensorflow

# Use raw strings in Python
# path = r"C:\path\to\dataset"
```

#### Problem: Long path issues
```
OSError: [Errno 36] File name too long
```

**Solutions:**
```bash
# Enable long paths in Windows
# Run as administrator:
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Use shorter paths
# Move dataset closer to root directory
```

### 2. Linux-Specific Issues

#### Problem: Permission denied errors
```
PermissionError: [Errno 13] Permission denied: '/home/argus/Workspace/dataset'
```

**Solutions:**
```bash
# Fix permissions
sudo chown -R $USER:$USER /home/argus/Workspace/dataset
chmod -R 755 /home/argus/Workspace/dataset

# Use sudo for setup (not recommended)
sudo python tensorflow/setup_env.py

# Check user groups
groups $USER
```

#### Problem: Library loading issues
```
ImportError: libcudnn.so.8: cannot open shared object file
```

**Solutions:**
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Install via package manager
sudo apt install libcudnn8 libcudnn8-dev

# Use conda installation
conda install cudnn -c conda-forge

# Check library location
find /usr -name "libcudnn*" 2>/dev/null
```

## Debugging Tools and Techniques

### 1. Logging and Monitoring

#### Enable Debug Logging
```bash
# Set TensorFlow logging level
export TF_CPP_MIN_LOG_LEVEL=0

# Enable debug mode in scripts
python training/train_wc.py --debug --verbose

# Use Python logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Your code here
"
```

#### Monitor System Resources
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# System monitoring
htop
iotop
nethogs

# Disk usage
df -h
du -sh tensorflow/

# Memory usage
free -h
```

### 2. Profiling Tools

#### TensorFlow Profiler
```python
import tensorflow as tf

# Enable profiling
tf.profiler.experimental.start('logs/profiler')

# Your training code here

tf.profiler.experimental.stop()

# View in TensorBoard
# tensorboard --logdir logs/profiler
```

#### Python Profiling
```bash
# Profile training script
python -m cProfile -o profile_output.prof training/train_wc.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(20)
"
```

### 3. Testing and Validation

#### Unit Testing
```bash
# Run all tests
python -m pytest tensorflow/tests/ -v

# Run specific test category
python -m pytest tensorflow/tests/unit/ -v
python -m pytest tensorflow/tests/integration/ -v

# Run with coverage
python -m pytest tensorflow/tests/ --cov=tensorflow --cov-report=html
```

#### Model Validation
```bash
# Compare with PyTorch implementation
python utils/compare_with_pytorch.py

# Validate model outputs
python utils/validate_model_outputs.py --model_path checkpoints/wc_best.pth

# Check numerical stability
python utils/check_numerical_stability.py
```

## Getting Help

### 1. Information to Collect

When reporting issues, please provide:

```bash
# System information
python tensorflow/utils/collect_system_info.py

# Environment information
conda list > environment_info.txt
pip list > pip_info.txt

# GPU information
nvidia-smi > gpu_info.txt

# Error logs
# Copy full error traceback
# Include relevant log files from logs/ directory
```

### 2. Useful Commands for Debugging

```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"

# Check CUDA compatibility
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda()); print(tf.test.is_gpu_available())"

# Memory usage
python -c "
import psutil
import GPUtil
print(f'RAM: {psutil.virtual_memory().percent}%')
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f'GPU {gpu.id}: {gpu.memoryUtil*100:.1f}%')
"

# Disk space
df -h

# Process information
ps aux | grep python
```

### 3. Common Solutions Summary

| Issue Type | Quick Fix | Detailed Solution |
|------------|-----------|-------------------|
| GPU OOM | Reduce batch size | See Memory Issues section |
| Slow training | Increase batch size | See Performance Issues section |
| Import errors | Reinstall dependencies | See Dependency Issues section |
| Data loading | Reduce num_workers | See Data Loading Issues section |
| Poor quality | Check model/data | See Inference Quality section |
| Setup fails | Clean environment | See Installation Issues section |

### 4. Prevention Tips

1. **Regular Maintenance**
   ```bash
   # Clean conda cache monthly
   conda clean --all
   
   # Update packages regularly
   conda update --all
   
   # Monitor disk space
   df -h
   ```

2. **Best Practices**
   - Always use virtual environments
   - Pin dependency versions in production
   - Monitor system resources during training
   - Keep regular backups of working configurations
   - Test on small datasets before full training

3. **Documentation**
   - Keep notes of working configurations
   - Document any custom modifications
   - Save successful command combinations
   - Record performance benchmarks

This troubleshooting guide should help you resolve most common issues. If you encounter problems not covered here, please collect the diagnostic information mentioned above and seek help from the community or maintainers.