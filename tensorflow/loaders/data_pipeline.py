"""
TensorFlow Data Pipeline for DewarpNet.
This module provides optimized tf.data.Dataset implementations for efficient data loading.
"""

import os
from typing import Tuple, Optional, Union, Callable, Dict, Any

import tensorflow as tf
import numpy as np

from .doc3d_wc_loader import Doc3DWCLoader
from .doc3d_bm_loader import Doc3DBMLoader


class TFDataPipeline:
    """
    TensorFlow data pipeline manager for DewarpNet training.
    
    This class provides optimized data loading with prefetching, parallel processing,
    and efficient batch handling for both world coordinate and backward mapping tasks.
    """
    
    def __init__(
        self,
        root: str,
        task_type: str = 'wc',  # 'wc' for world coordinates, 'bm' for backward mapping
        img_size: Union[int, Tuple[int, int]] = 512,
        altroot: Optional[str] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            root: Root directory path containing the dataset
            task_type: Type of task ('wc' or 'bm')
            img_size: Target image size
            altroot: Alternative root path (for backward mapping)
        """
        self.root = root
        self.task_type = task_type
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.altroot = altroot
        
        # Validate task type
        if task_type not in ['wc', 'bm']:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'wc' or 'bm'")
    
    def create_dataset(
        self,
        split: str = 'train',
        batch_size: int = 4,
        shuffle: bool = True,
        augmentations: bool = False,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        cache: bool = False,
        repeat: bool = False,
        drop_remainder: bool = True
    ) -> tf.data.Dataset:
        """
        Create an optimized TensorFlow dataset.
        
        Args:
            split: Dataset split ('train' or 'val')
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
            augmentations: Whether to apply data augmentations
            prefetch_buffer: Prefetch buffer size
            num_parallel_calls: Number of parallel calls for data processing
            cache: Whether to cache the dataset in memory
            repeat: Whether to repeat the dataset indefinitely
            drop_remainder: Whether to drop the last incomplete batch
            
        Returns:
            Optimized tf.data.Dataset
        """
        if self.task_type == 'wc':
            return self._create_wc_dataset(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                augmentations=augmentations,
                prefetch_buffer=prefetch_buffer,
                num_parallel_calls=num_parallel_calls,
                cache=cache,
                repeat=repeat,
                drop_remainder=drop_remainder
            )
        else:  # task_type == 'bm'
            return self._create_bm_dataset(
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                prefetch_buffer=prefetch_buffer,
                num_parallel_calls=num_parallel_calls,
                cache=cache,
                repeat=repeat,
                drop_remainder=drop_remainder
            )
    
    def _create_wc_dataset(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        augmentations: bool,
        prefetch_buffer: int,
        num_parallel_calls: int,
        cache: bool,
        repeat: bool,
        drop_remainder: bool
    ) -> tf.data.Dataset:
        """Create world coordinate dataset."""
        loader = Doc3DWCLoader(
            root=self.root,
            split=split,
            img_size=self.img_size,
            augmentations=augmentations,
            is_transform=True
        )
        
        if len(loader) == 0:
            raise ValueError(f"No data found for split '{split}' in {self.root}")
        
        # Create dataset from generator
        def generator():
            for i in range(len(loader)):
                try:
                    yield loader[i]
                except Exception as e:
                    print(f"Warning: Skipping sample {i} due to error: {e}")
                    continue
        
        # Determine output signature
        try:
            sample_img, sample_lbl = loader[0]
            output_signature = (
                tf.TensorSpec(shape=sample_img.shape, dtype=tf.float32),
                tf.TensorSpec(shape=sample_lbl.shape, dtype=tf.float32)
            )
        except Exception:
            # Fallback signature
            output_signature = (
                tf.TensorSpec(shape=(*self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(*self.img_size, 3), dtype=tf.float32)
            )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        return self._optimize_dataset(
            dataset, len(loader), batch_size, shuffle, prefetch_buffer,
            num_parallel_calls, cache, repeat, drop_remainder
        )
    
    def _create_bm_dataset(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        prefetch_buffer: int,
        num_parallel_calls: int,
        cache: bool,
        repeat: bool,
        drop_remainder: bool
    ) -> tf.data.Dataset:
        """Create backward mapping dataset."""
        loader = Doc3DBMLoader(
            root=self.root,
            altroot=self.altroot,
            split=split,
            img_size=self.img_size,
            is_transform=True
        )
        
        if len(loader) == 0:
            raise ValueError(f"No data found for split '{split}' in {self.root}")
        
        # Create dataset from generator
        def generator():
            for i in range(len(loader)):
                try:
                    yield loader[i]
                except Exception as e:
                    print(f"Warning: Skipping sample {i} due to error: {e}")
                    continue
        
        # Determine output signature
        try:
            sample_img, sample_lbl = loader[0]
            output_signature = (
                tf.TensorSpec(shape=sample_img.shape, dtype=tf.float32),
                tf.TensorSpec(shape=sample_lbl.shape, dtype=tf.float32)
            )
        except Exception:
            # Fallback signature
            output_signature = (
                tf.TensorSpec(shape=(*self.img_size, 6), dtype=tf.float32),
                tf.TensorSpec(shape=(*self.img_size, 2), dtype=tf.float32)
            )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        return self._optimize_dataset(
            dataset, len(loader), batch_size, shuffle, prefetch_buffer,
            num_parallel_calls, cache, repeat, drop_remainder
        )
    
    def _optimize_dataset(
        self,
        dataset: tf.data.Dataset,
        dataset_size: int,
        batch_size: int,
        shuffle: bool,
        prefetch_buffer: int,
        num_parallel_calls: int,
        cache: bool,
        repeat: bool,
        drop_remainder: bool
    ) -> tf.data.Dataset:
        """Apply optimizations to the dataset."""
        
        # Cache dataset if requested (before shuffle for efficiency)
        if cache:
            dataset = dataset.cache()
        
        # Shuffle dataset
        if shuffle:
            # Use a buffer size that's a fraction of the dataset size
            buffer_size = min(1000, max(100, dataset_size // 4))
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        
        # Repeat dataset if requested
        if repeat:
            dataset = dataset.repeat()
        
        # Batch the dataset
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        
        # Apply parallel processing if available
        if num_parallel_calls != 1:
            dataset = dataset.map(
                lambda x, y: (x, y),
                num_parallel_calls=num_parallel_calls
            )
        
        # Prefetch for performance
        dataset = dataset.prefetch(prefetch_buffer)
        
        return dataset
    
    def get_dataset_info(self, split: str = 'train') -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            split: Dataset split to analyze
            
        Returns:
            Dictionary containing dataset information
        """
        if self.task_type == 'wc':
            loader = Doc3DWCLoader(
                root=self.root,
                split=split,
                img_size=self.img_size,
                is_transform=False
            )
        else:
            loader = Doc3DBMLoader(
                root=self.root,
                altroot=self.altroot,
                split=split,
                img_size=self.img_size,
                is_transform=False
            )
        
        info = {
            'task_type': self.task_type,
            'split': split,
            'num_samples': len(loader),
            'img_size': self.img_size,
            'root': self.root
        }
        
        if self.task_type == 'bm':
            info['altroot'] = self.altroot
        
        # Try to get sample shapes
        try:
            if len(loader) > 0:
                sample_img, sample_lbl = loader[0]
                info['input_shape'] = sample_img.shape if hasattr(sample_img, 'shape') else 'Unknown'
                info['label_shape'] = sample_lbl.shape if hasattr(sample_lbl, 'shape') else 'Unknown'
        except Exception as e:
            info['sample_error'] = str(e)
        
        return info


def create_training_pipeline(
    root: str,
    task_type: str = 'wc',
    img_size: Union[int, Tuple[int, int]] = 512,
    batch_size: int = 4,
    augmentations: bool = True,
    altroot: Optional[str] = None,
    cache_train: bool = False,
    cache_val: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create optimized training and validation datasets.
    
    Args:
        root: Root directory path containing the dataset
        task_type: Type of task ('wc' or 'bm')
        img_size: Target image size
        batch_size: Batch size for training
        augmentations: Whether to apply augmentations to training data
        altroot: Alternative root path (for backward mapping)
        cache_train: Whether to cache training dataset
        cache_val: Whether to cache validation dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    pipeline = TFDataPipeline(
        root=root,
        task_type=task_type,
        img_size=img_size,
        altroot=altroot
    )
    
    # Create training dataset
    train_dataset = pipeline.create_dataset(
        split='train',
        batch_size=batch_size,
        shuffle=True,
        augmentations=augmentations,
        cache=cache_train,
        repeat=False,
        drop_remainder=True
    )
    
    # Create validation dataset
    val_dataset = pipeline.create_dataset(
        split='val',
        batch_size=batch_size,
        shuffle=False,
        augmentations=False,
        cache=cache_val,
        repeat=False,
        drop_remainder=False
    )
    
    return train_dataset, val_dataset


def benchmark_dataset_performance(
    dataset: tf.data.Dataset,
    num_batches: int = 100,
    warmup_batches: int = 10
) -> Dict[str, float]:
    """
    Benchmark dataset performance.
    
    Args:
        dataset: Dataset to benchmark
        num_batches: Number of batches to process
        warmup_batches: Number of warmup batches
        
    Returns:
        Dictionary containing performance metrics
    """
    import time
    
    # Warmup
    for i, batch in enumerate(dataset.take(warmup_batches)):
        pass
    
    # Benchmark
    start_time = time.time()
    batch_times = []
    
    for i, batch in enumerate(dataset.take(num_batches)):
        batch_start = time.time()
        # Simulate processing by accessing the data
        _ = tf.reduce_mean(batch[0])
        _ = tf.reduce_mean(batch[1])
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'batches_per_second': num_batches / total_time,
        'min_batch_time': np.min(batch_times),
        'max_batch_time': np.max(batch_times)
    }


def validate_dataset_integrity(
    dataset: tf.data.Dataset,
    num_samples: int = 10
) -> Dict[str, Any]:
    """
    Validate dataset integrity by checking samples.
    
    Args:
        dataset: Dataset to validate
        num_samples: Number of samples to check
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        'valid_samples': 0,
        'invalid_samples': 0,
        'errors': [],
        'shapes': [],
        'dtypes': [],
        'value_ranges': []
    }
    
    for i, (inputs, labels) in enumerate(dataset.take(num_samples)):
        try:
            # Check for NaN or infinite values
            if tf.reduce_any(tf.math.is_nan(inputs)) or tf.reduce_any(tf.math.is_inf(inputs)):
                results['errors'].append(f"Sample {i}: Invalid values in inputs")
                results['invalid_samples'] += 1
                continue
            
            if tf.reduce_any(tf.math.is_nan(labels)) or tf.reduce_any(tf.math.is_inf(labels)):
                results['errors'].append(f"Sample {i}: Invalid values in labels")
                results['invalid_samples'] += 1
                continue
            
            # Record shapes and dtypes
            results['shapes'].append({
                'input_shape': inputs.shape.as_list(),
                'label_shape': labels.shape.as_list()
            })
            
            results['dtypes'].append({
                'input_dtype': inputs.dtype.name,
                'label_dtype': labels.dtype.name
            })
            
            # Record value ranges
            results['value_ranges'].append({
                'input_min': float(tf.reduce_min(inputs)),
                'input_max': float(tf.reduce_max(inputs)),
                'label_min': float(tf.reduce_min(labels)),
                'label_max': float(tf.reduce_max(labels))
            })
            
            results['valid_samples'] += 1
            
        except Exception as e:
            results['errors'].append(f"Sample {i}: {str(e)}")
            results['invalid_samples'] += 1
    
    return results