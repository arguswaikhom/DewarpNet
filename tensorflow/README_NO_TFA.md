# DewarpNet TensorFlow - No TensorFlow Addons Required

**Important Update**: TensorFlow Addons has been deprecated as of May 2024. This implementation uses only native TensorFlow operations.

## Changes Made:

1. **Removed TensorFlow Addons dependency** - All TFA functions replaced with native TF operations
2. **Custom Grid Sampling** - Implemented bilinear grid sampling using native TensorFlow ops
3. **Native TensorFlow Only** - All functionality preserved using core TensorFlow

## Installation:

```bash
# Only TensorFlow is required now
pip install tensorflow
```

## Key Replacements:

- `tfa.image.resampler` → Custom `grid_sample_bilinear` function
- All other TFA imports removed
- Full backward compatibility maintained

## Files Updated:

- `tensorflow/recon_lossc_tf.py` - Custom grid sampling implementation
- `tensorflow/models_tf/densenetccnl_tf.py` - Removed TFA import
- `tensorflow/trainwc_tf.py` - Removed TFA import  
- `tensorflow/trainbm_tf.py` - Removed TFA import
- `tensorflow/DewarpNet_TensorFlow_Training.ipynb` - Updated dependencies

## Usage Remains Identical:

The training process and API remain exactly the same. Only the internal implementation changed to use native TensorFlow operations.

```python
# Training still works the same way
cd tensorflow
python trainwc_tf.py --arch unetnc_tf --data_path ../data/doc3d/ --batch_size 50 --tboard
python trainbm_tf.py --arch dnetccnl_tf --data_path ../data/doc3d/ --batch_size 50 --tboard
```

## Performance:

The custom grid sampling implementation provides identical results to TFA's resampler while using only native TensorFlow operations, ensuring long-term compatibility and maintainability.