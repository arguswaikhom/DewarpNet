# Training Guide

This guide provides detailed instructions for training the TensorFlow DewarpNet models.

## Overview

DewarpNet training consists of two sequential stages:

1. **World Coordinate Network (WC) Training**: Learns to predict 3D world coordinates from RGB images
2. **Backward Mapping Network (BM) Training**: Learns to predict 2D mapping coordinates for unwarping

## Prerequisites

- Completed environment setup (see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md))
- Dataset properly linked in `tensorflow/data/`
- GPU with at least 8GB VRAM

## Stage 1: World Coordinate Network Training

### Basic Training

```bash
# Navigate to tensorflow directory
cd tensorflow

# Train on doc3d_100 dataset (recommended for testing)
python training/train_wc.py --dataset doc3d_100 --epochs 50

# Train on full dataset
python training/train_wc.py --dataset doc3d --epochs 200
```

### Advanced Training Options

```bash
# Custom batch size and learning rate
python training/train_wc.py \
    --dataset doc3d_100 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --epochs 50

# Resume from checkpoint
python training/train_wc.py \
    --dataset doc3d_100 \
    --resume_from checkpoints/wc_epoch_25.pth \
    --epochs 50

# Enable mixed precision training
python training/train_wc.py \
    --dataset doc3d_100 \
    --mixed_precision \
    --epochs 50

# Custom output directory
python training/train_wc.py \
    --dataset doc3d_100 \
    --output_dir custom_checkpoints/ \
    --epochs 50
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | doc3d_100 | Dataset to use (doc3d_100, doc3d_1000, doc3d) |
| `--batch_size` | 8 | Training batch size |
| `--learning_rate` | 0.0001 | Initial learning rate |
| `--epochs` | 50 | Number of training epochs |
| `--img_size` | 256 | Input image size (256x256) |
| `--num_workers` | 4 | Data loader workers |
| `--save_freq` | 5 | Checkpoint save frequency (epochs) |
| `--val_freq` | 1 | Validation frequency (epochs) |
| `--mixed_precision` | False | Enable mixed precision training |
| `--resume_from` | None | Path to checkpoint to resume from |
| `--output_dir` | checkpoints/ | Output directory for checkpoints |

### Monitoring Training

#### TensorBoard
```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir logs/wc_training

# View at http://localhost:6006
```

#### Training Logs
```bash
# View real-time logs
tail -f logs/wc_training/training.log

# View specific metrics
grep "Validation Loss" logs/wc_training/training.log
```

### Expected Training Behavior

#### Loss Curves
- **Training Loss**: Should decrease from ~0.1 to ~0.02-0.03
- **Validation Loss**: Should follow training loss with slight gap
- **Gradient Loss**: Should decrease from ~0.05 to ~0.01

#### Training Time
- **doc3d_100**: ~30 minutes (50 epochs, RTX 3080)
- **doc3d_1000**: ~5 hours (50 epochs, RTX 3080)
- **doc3d**: ~50 hours (200 epochs, RTX 3080)

#### Memory Usage
- **Batch Size 8**: ~6-7GB VRAM
- **Batch Size 4**: ~4-5GB VRAM
- **Mixed Precision**: ~20% memory reduction

## Stage 2: Backward Mapping Network Training

### Basic Training

```bash
# Train BM network (requires trained WC model)
python training/train_bm.py \
    --wc_model_path checkpoints/wc_best.pth \
    --dataset doc3d_100 \
    --epochs 100

# Train on full dataset
python training/train_bm.py \
    --wc_model_path checkpoints/wc_best.pth \
    --dataset doc3d \
    --epochs 300
```

### Advanced Training Options

```bash
# Custom loss weights
python training/train_bm.py \
    --wc_model_path checkpoints/wc_best.pth \
    --dataset doc3d_100 \
    --grad_loss_weight 20 \
    --recon_loss_weight 1 \
    --ssim_loss_weight 0.1 \
    --epochs 100

# Custom batch size and learning rate
python training/train_bm.py \
    --wc_model_path checkpoints/wc_best.pth \
    --dataset doc3d_100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --epochs 100

# Enable visualization during training
python training/train_bm.py \
    --wc_model_path checkpoints/wc_best.pth \
    --dataset doc3d_100 \
    --visualize_freq 10 \
    --epochs 100
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wc_model_path` | Required | Path to trained WC model |
| `--dataset` | doc3d_100 | Dataset to use |
| `--batch_size` | 16 | Training batch size |
| `--learning_rate` | 0.0001 | Initial learning rate |
| `--epochs` | 100 | Number of training epochs |
| `--img_size` | 128 | Input image size (128x128) |
| `--grad_loss_weight` | 20 | Gradient loss weight |
| `--recon_loss_weight` | 1 | Reconstruction loss weight |
| `--ssim_loss_weight` | 0.1 | SSIM loss weight |
| `--visualize_freq` | 0 | Visualization frequency (0=disabled) |
| `--tight_crop` | True | Enable tight cropping |
| `--resume_from` | None | Path to checkpoint to resume from |
| `--output_dir` | checkpoints/ | Output directory for checkpoints |

### Expected Training Behavior

#### Loss Curves
- **Total Loss**: Should decrease from ~20 to ~1-2
- **Gradient Loss**: Should decrease from ~1.0 to ~0.05
- **Reconstruction Loss**: Should decrease from ~0.1 to ~0.01
- **SSIM Loss**: Should decrease from ~0.5 to ~0.1

#### Training Time
- **doc3d_100**: ~45 minutes (100 epochs, RTX 3080)
- **doc3d_1000**: ~7 hours (100 epochs, RTX 3080)
- **doc3d**: ~70 hours (300 epochs, RTX 3080)

## Training Best Practices

### 1. Dataset Selection Strategy

```bash
# Recommended training progression:
# 1. Start with doc3d_100 for quick validation
python training/train_wc.py --dataset doc3d_100 --epochs 50
python training/train_bm.py --wc_model_path checkpoints/wc_best.pth --dataset doc3d_100 --epochs 100

# 2. Scale to doc3d_1000 for better performance
python training/train_wc.py --dataset doc3d_1000 --epochs 100
python training/train_bm.py --wc_model_path checkpoints/wc_best.pth --dataset doc3d_1000 --epochs 200

# 3. Final training on full dataset
python training/train_wc.py --dataset doc3d --epochs 200
python training/train_bm.py --wc_model_path checkpoints/wc_best.pth --dataset doc3d --epochs 300
```

### 2. Hyperparameter Tuning

#### Learning Rate Scheduling
```bash
# The training scripts automatically use ReduceLROnPlateau
# Patience: 10 epochs
# Factor: 0.5
# Min LR: 1e-7
```

#### Batch Size Guidelines
- **8GB VRAM**: WC batch_size=4-6, BM batch_size=8-12
- **12GB VRAM**: WC batch_size=6-8, BM batch_size=12-16
- **16GB+ VRAM**: WC batch_size=8-12, BM batch_size=16-24

### 3. Monitoring and Debugging

#### Check GPU Utilization
```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi
```

#### Memory Optimization
```bash
# If running out of memory, reduce batch size
python training/train_wc.py --batch_size 4

# Enable gradient checkpointing (if implemented)
python training/train_wc.py --gradient_checkpointing

# Use mixed precision
python training/train_wc.py --mixed_precision
```

#### Debugging Training Issues
```bash
# Enable debug mode for detailed logging
python training/train_wc.py --debug

# Test data loading without training
python training/train_wc.py --test_data_only

# Validate model architecture
python training/train_wc.py --validate_model_only
```

### 4. Checkpoint Management

#### Automatic Checkpointing
- Best model (lowest validation loss): `checkpoints/{model}_best.pth`
- Latest model: `checkpoints/{model}_latest.pth`
- Epoch checkpoints: `checkpoints/{model}_epoch_{N}.pth`

#### Manual Checkpoint Operations
```bash
# List available checkpoints
ls -la checkpoints/

# Resume from specific checkpoint
python training/train_wc.py --resume_from checkpoints/wc_epoch_25.pth

# Convert checkpoint format (if needed)
python utils/convert_checkpoint.py --input checkpoints/wc_best.pth --output checkpoints/wc_best_converted.pth
```

## Validation and Testing

### During Training Validation
```bash
# Validation runs automatically every epoch
# Check validation metrics in logs or TensorBoard
```

### Post-Training Validation
```bash
# Test trained WC model
python tests/integration/test_wc_model.py --model_path checkpoints/wc_best.pth

# Test trained BM model
python tests/integration/test_bm_model.py --model_path checkpoints/bm_best.pth

# Full pipeline test
python tests/integration/test_full_pipeline.py \
    --wc_model checkpoints/wc_best.pth \
    --bm_model checkpoints/bm_best.pth
```

### Performance Benchmarking
```bash
# Benchmark training speed
python tests/performance/benchmark_training.py

# Compare with PyTorch implementation
python tests/performance/compare_frameworks.py
```

## Troubleshooting Training Issues

### Common Issues and Solutions

#### 1. Out of Memory Errors
```bash
# Reduce batch size
python training/train_wc.py --batch_size 2

# Enable mixed precision
python training/train_wc.py --mixed_precision

# Reduce image size (not recommended)
python training/train_wc.py --img_size 128
```

#### 2. Slow Training Speed
```bash
# Increase number of data loading workers
python training/train_wc.py --num_workers 8

# Enable mixed precision
python training/train_wc.py --mixed_precision

# Check GPU utilization
nvidia-smi
```

#### 3. Loss Not Decreasing
```bash
# Check learning rate
python training/train_wc.py --learning_rate 0.001

# Verify data loading
python training/train_wc.py --test_data_only

# Check model architecture
python training/train_wc.py --validate_model_only
```

#### 4. Validation Loss Increasing (Overfitting)
```bash
# Reduce learning rate
python training/train_wc.py --learning_rate 0.00005

# Add regularization (if implemented)
python training/train_wc.py --weight_decay 0.0001

# Use smaller dataset for testing
python training/train_wc.py --dataset doc3d_100
```

### Getting Help

1. Check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
2. Review training logs in `logs/` directory
3. Use debug mode: `--debug` flag
4. Check GPU memory: `nvidia-smi`
5. Validate environment: `python utils/test_setup.py`

## Advanced Training Techniques

### 1. Transfer Learning
```bash
# Fine-tune from pre-trained model
python training/train_wc.py \
    --pretrained_path pretrained/wc_pretrained.pth \
    --learning_rate 0.00001 \
    --epochs 20
```

### 2. Multi-GPU Training (if implemented)
```bash
# Train on multiple GPUs
python training/train_wc.py --multi_gpu --gpus 0,1,2,3
```

### 3. Custom Loss Weights
```bash
# Experiment with different loss combinations
python training/train_bm.py \
    --wc_model_path checkpoints/wc_best.pth \
    --grad_loss_weight 10 \
    --recon_loss_weight 2 \
    --ssim_loss_weight 0.5
```

### 4. Data Augmentation Tuning
```bash
# Disable augmentations for debugging
python training/train_wc.py --no_augmentations

# Custom augmentation parameters (if implemented)
python training/train_wc.py --aug_rotation 15 --aug_scale 0.1
```

This comprehensive training guide should help you successfully train the TensorFlow DewarpNet models. Remember to start with smaller datasets for validation before scaling to full training.