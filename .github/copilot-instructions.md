# DewarpNet AI Coding Instructions

## Project Overview
DewarpNet is a PyTorch implementation for single-image document unwarping using stacked 3D and 2D regression networks. The system consists of two-stage training: a Shape Network (World Coordinate regression) and a Texture Mapping Network (Backward Mapping).

## Architecture Components

### Two-Stage Pipeline
1. **Shape Network (WC)**: Regresses 3D world coordinates from RGB input using UNet architecture (`unetnc`)
2. **Texture Mapping Network (BM)**: Performs backward mapping from world coordinates to texture using DenseNet architecture (`dnetccnl`)

Key files: `trainwc.py` (stage 1), `trainbm.py` (stage 2), `infer.py` (end-to-end inference)

### Data Flow Pattern
```
RGB Image → Shape Network → World Coordinates → Texture Network → Backward Mapping → Unwarped Document
```

### Model Architecture Registry
- Models are registered in `models/__init__.py` via `get_model()` function
- Available architectures: `unetnc` (UNet for world coordinates), `dnetccnl` (DenseNet for backward mapping)
- Model loading follows DataParallel pattern with `convert_state_dict()` utility in `utils.py`

## Development Patterns

### Training Configuration
- **Shape Network**: Uses L1Loss + optional Gradient Loss (`grad_loss.py`), Hardtanh(0,1) activation
- **Texture Network**: Uses L1Loss + Reconstruction Loss (`recon_lossc.py`) + SSIM loss
- Both use Adam optimizer with ReduceLROnPlateau scheduler
- Experiment naming convention: `{activation}_{dataset}_{lossparams}_{augmentations}_{trainstart}`

### Data Loading Architecture
- Loaders in `loaders/` follow naming pattern: `doc3d{wc|bmnoimgc}_loader.py`
- World coordinate loader (`doc3dwc_loader.py`): Loads RGB + .exr world coordinates
- Backward mapping loader (`doc3dbmnoimgc_loader.py`): Loads albedo + world coordinates + .mat backward mapping
- Custom augmentations in `loaders/augmentationsk.py` with texture blending

### File Structure Conventions
- Models saved as `.pkl` files with naming: `{arch}_{epoch}_{val_mse}_{train_mse}_{experiment_name}_model.pkl`
- Checkpoints in `checkpoints-wc/` (world coordinates) and `checkpoints-bm/` (backward mapping)
- Data expects Doc3D dataset structure: `img/`, `wc/`, `bm/`, `alb/`, `norm/`, `recon/` folders

### Loss Functions
- `grad_loss.py`: Custom gradient-based loss using Sobel filters for edge-aware training
- `recon_lossc.py`: Reconstruction loss for texture mapping with SSIM components
- Loss combination patterns: `l1loss + (0.2*g_loss)` for WC, `(10.0*l1loss) + (0.5*rloss)` for BM

## Critical Developer Workflows

### Training Commands
```bash
# Stage 1 - World Coordinate Training
python trainwc.py --arch unetnc --data_path ./data/doc3d/ --batch_size 50 --tboard

# Stage 2 - Backward Mapping Training  
python trainbm.py --arch dnetccnl --img_rows 128 --img_cols 128 --n_epoch 250 --batch_size 50 --l_rate 0.0001 --tboard --data_path ./data/doc3d

# Inference
python infer.py --wc_model_path ./eval/models/unetnc_doc3d.pkl --bm_model_path ./eval/models/dnetccnl_doc3d.pkl --show
```

### Data Preprocessing
- World coordinates normalized to [0,1] using hardcoded bounds: `xmx=1.2539363, xmn=-1.2442188, ymx=1.2396319, ymn=-1.2289206, zmx=0.6436657, zmn=-0.67492497`
- Backward mapping normalized to [-1,1] range
- Images resized to 256x256 for WC training, 128x128 for BM training

### Debugging & Visualization  
- TensorBoard integration with `--tboard` flag
- Visualization utilities in `utils.py`: `show_wc_tnsboard()`, `show_unwarp_tnsboard()`
- Image grid display for training progress monitoring

## Integration Points

### External Dependencies
- OpenEXR files for world coordinate ground truth (`.exr` format)
- HDF5/MAT files for backward mapping ground truth (`.mat` format)  
- Doc3D dataset structure with specific folder organization
- Scipy version dependency: Use scipy < 1.2.0 for exact paper reproduction

### Model Compatibility
- Models expect DataParallel wrapping - use `convert_state_dict()` when loading for inference
- CPU/GPU compatibility handled in `infer.py` with device-aware model loading
- Hardtanh activation crucial for world coordinate output clamping

### Evaluation Integration
- OCR evaluation uses Tesseract v4.1.0 with PyTesseract v0.2.6
- Image metrics follow DocUNet evaluation protocol
- MATLAB SSIM version dependency affects reported scores (2018b vs 2020a)

## Project-Specific Conventions

- **Naming**: Use underscores for experiment names, hyphens for checkpoint directories
- **GPU Usage**: Multi-GPU training via DataParallel, single GPU inference
- **Image Formats**: PNG for RGB, EXR for world coordinates, MAT for backward mapping
- **Coordinate Systems**: World coordinates in camera space, backward mapping in texture space
- **Augmentation**: Background texture blending during WC training, crop-based augmentation for BM training
