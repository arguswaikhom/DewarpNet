"""
TensorFlow loaders for DewarpNet.
This package provides data loading functionality for world coordinate and backward mapping tasks.
"""

from .doc3d_wc_loader import Doc3DWCLoader, create_wc_dataset
from .doc3d_bm_loader import Doc3DBMLoader, create_bm_dataset
from .augmentations_tf import (
    tight_crop_tf, 
    tight_crop_d_tf, 
    color_jitter_tf, 
    data_aug_tf,
    tf_color_jitter,
    tf_random_crop_and_resize
)
from .data_pipeline import (
    TFDataPipeline,
    create_training_pipeline,
    benchmark_dataset_performance,
    validate_dataset_integrity
)

__all__ = [
    # World Coordinate Loader
    'Doc3DWCLoader',
    'create_wc_dataset',
    
    # Backward Mapping Loader
    'Doc3DBMLoader', 
    'create_bm_dataset',
    
    # Augmentations
    'tight_crop_tf',
    'tight_crop_d_tf',
    'color_jitter_tf',
    'data_aug_tf',
    'tf_color_jitter',
    'tf_random_crop_and_resize',
    
    # Data Pipeline
    'TFDataPipeline',
    'create_training_pipeline',
    'benchmark_dataset_performance',
    'validate_dataset_integrity'
]