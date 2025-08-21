# DewarpNet AI Coding Assistant Instructions

## Project Overview

DewarpNet is a two-stage deep learning pipeline for document image unwarping:
1. **Shape Network** (`trainwc.py`): Predicts 3D world coordinates from RGB input using UNet architecture (`unetnc`)
2. **Texture Mapping Network** (`trainbm.py`): Generates backward mapping from world coordinates using DenseNet (`dnetccnl`)

## Architecture & Data Flow

**Two-Phase Training Pipeline:**
- World Coordinate (WC) training: `RGB → 3D coordinates` (256x256 input)
- Backward Mapping (BM) training: `3D coordinates → 2D mapping` (128x128 input)
- Inference chains both models: `RGB → WC → BM → Unwarped Image`

**Key Models:**
- `models/unetnc.py`: UNet with 7 downsampling layers for world coordinate regression
- `models/densenetccnl.py`: DenseNet variant for backward mapping regression

## Training Workflow

**1. Shape Network Training:**
```bash
python trainwc.py --arch unetnc --data_path ./data/doc3d/ --batch_size 50 --tboard
```
- Uses L1 + gradient loss (`grad_loss.py`)
- Outputs saved to `checkpoints-wc/`
- Hardtanh(0,1) activation for world coordinates

**2. Texture Mapping Training:**
```bash
python trainbm.py --arch dnetccnl --img_rows 128 --img_cols 128 --batch_size 50 --l_rate 0.0001 --tboard
```
- Uses L1 + reconstruction loss (`recon_lossc.py`) 
- Outputs saved to `checkpoints-bm/`
- Includes SSIM loss for texture quality

## Custom Loss Functions

**Gradient Loss (`grad_loss.py`):**
- Custom Sobel-based gradient preservation
- Window sizes: 3, 5, 7 with specific multipliers (2, 20, 780)
- Applied to world coordinate prediction

**Reconstruction Loss (`recon_lossc.py`):**
- Unwarp-based loss using `F.grid_sample`
- Combines MSE + SSIM for texture quality
- Operates on full-resolution images

## Data Structure Conventions

**Doc3D Dataset Layout:**
```
data/doc3d/
├── img/     # Input RGB images
├── wc/      # World coordinate ground truth
├── bm/      # Backward mapping ground truth
├── train.txt # Training file list
└── val.txt   # Validation file list
```

**File Naming:** Format like `1/824_8-cp_Page_0503-7Ns0001`

## Development Patterns

**Model Loading:**
- Use `get_model()` from `models/__init__.py` with architecture names `unetnc`/`dnetccnl`
- DataParallel wrapper for multi-GPU training
- `convert_state_dict()` in `utils.py` handles DataParallel state dict conversion

**Data Loading:**
- Loaders: `doc3dwcLoader` (WC), `doc3dbmnoimgcLoader` (BM)
- Custom augmentations in `loaders/augmentationsk.py`
- Image size conventions: 256×256 for WC, 128×128 for BM

**Experiment Tracking:**
- TensorBoard integration with `--tboard` flag
- Custom experiment naming: `{activation}_{dataset}_{loss}_{augmentations}_{training_start}`
- Log files with loss breakdowns by epoch

## Inference Pipeline

**End-to-End Process (`infer.py`):**
1. Load RGB image → resize to 256×256
2. WC model predicts world coordinates
3. Interpolate WC output to 128×128 for BM model  
4. BM model predicts backward mapping
5. Apply grid sampling to unwarp original image

**Key Implementation:** Uses `F.grid_sample` with bilinear interpolation and cv2.blur smoothing

## Critical Dependencies

- PyTorch with CUDA support
- `scipy==1.1.0` (specific version for reproducible results)
- TensorBoard for visualization (`tensorboardX`)
- Custom SSIM implementation in `pytorch_ssim/`

## Debugging & Evaluation

**Common Issues:**
- Matlab SSIM version differences (2018b vs 2020a) affect reported metrics
- Use `scipy < 1.2.0` for exact paper reproduction
- DataParallel state dict requires conversion for single GPU inference

**Evaluation:** Uses DocUNet evaluation protocol with provided test images in `eval/inp/`