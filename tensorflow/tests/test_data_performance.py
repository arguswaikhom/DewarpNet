"""
Performance tests for TensorFlow Data Pipeline.
This script demonstrates and benchmarks the data loading performance.
"""

import os
import tempfile
import time
from unittest.mock import patch

import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.data_pipeline import (
    TFDataPipeline, 
    create_training_pipeline, 
    benchmark_dataset_performance,
    validate_dataset_integrity
)


def create_mock_dataset_structure(temp_dir: str, num_samples: int = 100):
    """Create a mock dataset structure for performance testing."""
    # Create directories
    os.makedirs(os.path.join(temp_dir, 'img'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'wc'), exist_ok=True)
    
    # Create file lists
    train_samples = [f'sample_{i:04d}' for i in range(int(num_samples * 0.8))]
    val_samples = [f'val_sample_{i:04d}' for i in range(int(num_samples * 0.2))]
    
    with open(os.path.join(temp_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_samples))
    
    with open(os.path.join(temp_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_samples))
    
    return len(train_samples), len(val_samples)


def run_performance_benchmark():
    """Run comprehensive performance benchmarks."""
    print("=== TensorFlow Data Pipeline Performance Benchmark ===\n")
    
    # Create temporary dataset
    temp_dir = tempfile.mkdtemp()
    num_train, num_val = create_mock_dataset_structure(temp_dir, num_samples=200)
    
    print(f"Created mock dataset with {num_train} training and {num_val} validation samples")
    print(f"Dataset location: {temp_dir}\n")
    
    try:
        # Mock data loading
        with patch('cv2.imread') as mock_imread:
            # Mock image and EXR loading
            mock_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            mock_wc = np.random.uniform(-1, 1, (256, 256, 3)).astype(np.float32)
            
            def imread_side_effect(path, flags=None):
                if path.endswith('.png'):
                    return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
                elif path.endswith('.exr'):
                    return mock_wc
                return None
            
            mock_imread.side_effect = imread_side_effect
            
            # Test different configurations
            configs = [
                {
                    'name': 'Basic Configuration',
                    'batch_size': 4,
                    'prefetch_buffer': 1,
                    'num_parallel_calls': 1,
                    'cache': False
                },
                {
                    'name': 'Optimized Configuration',
                    'batch_size': 8,
                    'prefetch_buffer': tf.data.AUTOTUNE,
                    'num_parallel_calls': tf.data.AUTOTUNE,
                    'cache': False
                },
                {
                    'name': 'Cached Configuration',
                    'batch_size': 8,
                    'prefetch_buffer': tf.data.AUTOTUNE,
                    'num_parallel_calls': tf.data.AUTOTUNE,
                    'cache': True
                }
            ]
            
            results = {}
            
            for config in configs:
                print(f"Testing {config['name']}...")
                
                # Create pipeline
                pipeline = TFDataPipeline(
                    root=temp_dir,
                    task_type='wc',
                    img_size=(256, 256)
                )
                
                # Create dataset
                dataset = pipeline.create_dataset(
                    split='train',
                    batch_size=config['batch_size'],
                    shuffle=True,
                    prefetch_buffer=config['prefetch_buffer'],
                    num_parallel_calls=config['num_parallel_calls'],
                    cache=config['cache']
                )
                
                # Benchmark performance
                metrics = benchmark_dataset_performance(
                    dataset,
                    num_batches=20,
                    warmup_batches=5
                )
                
                results[config['name']] = metrics
                
                print(f"  Total time: {metrics['total_time']:.2f}s")
                print(f"  Avg batch time: {metrics['avg_batch_time']:.4f}s")
                print(f"  Batches per second: {metrics['batches_per_second']:.2f}")
                print(f"  Min/Max batch time: {metrics['min_batch_time']:.4f}s / {metrics['max_batch_time']:.4f}s")
                print()
            
            # Compare results
            print("=== Performance Comparison ===")
            baseline = results['Basic Configuration']['batches_per_second']
            
            for name, metrics in results.items():
                speedup = metrics['batches_per_second'] / baseline
                print(f"{name}: {speedup:.2f}x speedup")
            
            print()
            
            # Test dataset integrity
            print("=== Dataset Integrity Validation ===")
            
            dataset = pipeline.create_dataset(
                split='train',
                batch_size=4,
                shuffle=False
            )
            
            integrity_results = validate_dataset_integrity(dataset, num_samples=10)
            
            print(f"Valid samples: {integrity_results['valid_samples']}")
            print(f"Invalid samples: {integrity_results['invalid_samples']}")
            print(f"Errors: {len(integrity_results['errors'])}")
            
            if integrity_results['errors']:
                print("Errors found:")
                for error in integrity_results['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")
            
            # Show sample shapes and value ranges
            if integrity_results['shapes']:
                sample_shape = integrity_results['shapes'][0]
                sample_range = integrity_results['value_ranges'][0]
                
                print(f"\nSample data info:")
                print(f"  Input shape: {sample_shape['input_shape']}")
                print(f"  Label shape: {sample_shape['label_shape']}")
                print(f"  Input range: [{sample_range['input_min']:.3f}, {sample_range['input_max']:.3f}]")
                print(f"  Label range: [{sample_range['label_min']:.3f}, {sample_range['label_max']:.3f}]")
            
            print()
            
            # Test training pipeline creation
            print("=== Training Pipeline Test ===")
            
            start_time = time.time()
            train_dataset, val_dataset = create_training_pipeline(
                root=temp_dir,
                task_type='wc',
                img_size=(256, 256),
                batch_size=8,
                augmentations=True
            )
            creation_time = time.time() - start_time
            
            print(f"Pipeline creation time: {creation_time:.2f}s")
            
            # Test a few batches from each dataset
            print("Testing training dataset...")
            train_batch_count = 0
            for batch in train_dataset.take(3):
                train_batch_count += 1
                imgs, lbls = batch
                print(f"  Batch {train_batch_count}: imgs {imgs.shape}, lbls {lbls.shape}")
            
            print("Testing validation dataset...")
            val_batch_count = 0
            for batch in val_dataset.take(2):
                val_batch_count += 1
                imgs, lbls = batch
                print(f"  Batch {val_batch_count}: imgs {imgs.shape}, lbls {lbls.shape}")
            
            print(f"\nSuccessfully processed {train_batch_count} training and {val_batch_count} validation batches")
            
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {temp_dir}")
    
    print("\n=== Benchmark Complete ===")


def run_memory_usage_test():
    """Test memory usage with different configurations."""
    print("\n=== Memory Usage Test ===")
    
    # This would require psutil for detailed memory monitoring
    # For now, we'll just test that large datasets can be created without errors
    
    temp_dir = tempfile.mkdtemp()
    create_mock_dataset_structure(temp_dir, num_samples=1000)
    
    try:
        with patch('cv2.imread') as mock_imread:
            mock_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            mock_wc = np.random.uniform(-1, 1, (512, 512, 3)).astype(np.float32)
            
            def imread_side_effect(path, flags=None):
                if path.endswith('.png'):
                    return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
                elif path.endswith('.exr'):
                    return mock_wc
                return None
            
            mock_imread.side_effect = imread_side_effect
            
            # Test large batch sizes
            batch_sizes = [1, 4, 8, 16, 32]
            
            for batch_size in batch_sizes:
                print(f"Testing batch size {batch_size}...")
                
                pipeline = TFDataPipeline(
                    root=temp_dir,
                    task_type='wc',
                    img_size=(512, 512)
                )
                
                dataset = pipeline.create_dataset(
                    split='train',
                    batch_size=batch_size,
                    shuffle=False,
                    cache=False
                )
                
                # Process a few batches
                try:
                    batch_count = 0
                    for batch in dataset.take(5):
                        batch_count += 1
                        imgs, lbls = batch
                        # Force computation to test memory usage
                        _ = tf.reduce_mean(imgs)
                        _ = tf.reduce_mean(lbls)
                    
                    print(f"  Successfully processed {batch_count} batches")
                    
                except tf.errors.ResourceExhaustedError as e:
                    print(f"  Memory exhausted at batch size {batch_size}: {e}")
                    break
                except Exception as e:
                    print(f"  Error at batch size {batch_size}: {e}")
    
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Memory usage test complete")


if __name__ == '__main__':
    # Run performance benchmark
    run_performance_benchmark()
    
    # Run memory usage test
    run_memory_usage_test()
    
    print("\nAll performance tests completed successfully!")