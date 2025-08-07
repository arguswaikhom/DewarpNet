# TensorFlow DewarpNet Implementation

This directory contains the TensorFlow implementation of DewarpNet, a two-stage neural network system for document unwarping.

## Quick Start

### 1. Environment Setup

```bash
# Run the setup script
python tensorflow/setup_env.py

# Activate the conda environment
conda activate dewarpnet_tf

# Validate the setup
python tensorflow/utils/gpu_utils.py
python tensorflow/utils/test_setup.py
```

### 2. Directory Structure

```
tensorflow/
├── models/          # Neural network architectures
├── loaders/         # Data loading utilities  
├── losses/          # Loss function implementations
├── training/        # Training scripts and utilities
├── inference/       # Inference pipeline
├── utils/           # Utility functions
├── data/            # Symbolic links to datasets
├── requirements_tf.txt
├── setup_env.py
└── README.md
```

### 3. Dataset Setup

The setup script automatically creates symbolic links to:
- `/home/argus/Workspace/dataset/doc3d` → `tensorflow/data/doc3d`
- `/home/argus/Workspace/dataset/doc3d_100` → `tensorflow/data/doc3d_100`
- `/home/argus/Workspace/dataset/doc3d_1000` → `tensorflow/data/doc3d_1000`
- `/home/argus/Workspace/dataset/input_crop` → `tensorflow/data/input_crop`

### 4. GPU Requirements

- NVIDIA GPU with CUDA support
- CUDA 11.2+ and cuDNN 8.1+
- TensorFlow GPU 2.10+

### 5. Testing the Setup

```bash
# Test GPU detection and performance
python tensorflow/utils/gpu_utils.py

# Test complete setup
python tensorflow/utils/test_setup.py

# Benchmark GPU performance only
python tensorflow/utils/gpu_utils.py --benchmark-only
```

## Environment Setup Options

### Custom Environment Name
```bash
python tensorflow/setup_env.py --env-name my_dewarpnet_env
```

### Custom Dataset Path
```bash
python tensorflow/setup_env.py --dataset-path /path/to/your/dataset
```

### Skip Conda Environment Creation
```bash
python tensorflow/setup_env.py --skip-conda
```

## Troubleshooting

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Reinstall TensorFlow GPU: `pip install tensorflow-gpu>=2.10.0`

### Import Errors
1. Activate conda environment: `conda activate dewarpnet_tf`
2. Reinstall dependencies: `pip install -r tensorflow/requirements_tf.txt`

### Dataset Links Missing
1. Verify dataset path exists: `/home/argus/Workspace/dataset`
2. Re-run setup: `python tensorflow/setup_env.py`

## Next Steps

After successful setup:
1. Implement model architectures (Task 2)
2. Create data loaders (Task 3)  
3. Implement loss functions (Task 4)
4. Set up training pipeline (Task 5)
5. Create inference pipeline (Task 6)

## Performance Expectations

The TensorFlow implementation should achieve:
- Training speed comparable to PyTorch version
- Identical model architectures and parameter counts
- Equivalent loss convergence and final performance
- Compatible checkpoint formats for model sharing