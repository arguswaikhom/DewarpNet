#!/usr/bin/env python3
"""
GPU detection and validation utilities for TensorFlow DewarpNet.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional


def check_tensorflow_version():
    """Check TensorFlow version and compatibility."""
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if version is compatible
    version_parts = tf.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1])
    
    if major < 2 or (major == 2 and minor < 10):
        print("Warning: TensorFlow version < 2.10 may have compatibility issues")
        return False
    
    print("✓ TensorFlow version is compatible")
    return True


def detect_gpus() -> List[tf.config.PhysicalDevice]:
    """Detect available GPUs."""
    print("Detecting GPUs...")
    
    # Get list of physical devices
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("❌ No GPUs detected")
        return []
    
    print(f"✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    return gpus


def configure_gpu_memory_growth():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
            return True
        except RuntimeError as e:
            print(f"❌ Error configuring GPU memory growth: {e}")
            return False
    
    return False


def test_gpu_computation():
    """Test basic GPU computation."""
    print("Testing GPU computation...")
    
    try:
        # Create a simple computation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            result = tf.reduce_sum(c)
        
        print(f"✓ GPU computation test passed. Result: {result.numpy():.2f}")
        return True
        
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        return False


def get_gpu_memory_info() -> Dict[str, any]:
    """Get GPU memory information."""
    gpus = tf.config.list_physical_devices('GPU')
    memory_info = {}
    
    for i, gpu in enumerate(gpus):
        try:
            # Get memory info (this requires GPU to be initialized)
            gpu_details = tf.config.experimental.get_device_details(gpu)
            memory_info[f'GPU_{i}'] = {
                'name': gpu.name,
                'details': gpu_details
            }
        except Exception as e:
            memory_info[f'GPU_{i}'] = {
                'name': gpu.name,
                'error': str(e)
            }
    
    return memory_info


def check_cuda_compatibility():
    """Check CUDA and cuDNN compatibility."""
    print("Checking CUDA compatibility...")
    
    # Check if CUDA is available
    cuda_available = tf.test.is_built_with_cuda()
    print(f"CUDA support built: {cuda_available}")
    
    if cuda_available:
        # Check GPU availability
        gpu_available = tf.test.is_gpu_available()
        print(f"GPU available: {gpu_available}")
        
        if gpu_available:
            print("✓ CUDA and GPU are properly configured")
            return True
        else:
            print("❌ CUDA is built but GPU is not available")
            return False
    else:
        print("❌ TensorFlow was not built with CUDA support")
        return False


def benchmark_gpu_performance():
    """Benchmark GPU performance with typical DewarpNet operations."""
    print("Benchmarking GPU performance...")
    
    if not tf.config.list_physical_devices('GPU'):
        print("❌ No GPU available for benchmarking")
        return None
    
    try:
        import time
        
        # Test convolution operation (common in neural networks)
        with tf.device('/GPU:0'):
            # Create sample data similar to DewarpNet input
            input_tensor = tf.random.normal([8, 256, 256, 3])  # Batch of RGB images
            conv_layer = tf.keras.layers.Conv2D(64, 3, padding='same')
            
            # Warm up
            _ = conv_layer(input_tensor)
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                output = conv_layer(input_tensor)
            tf.keras.backend.clear_session()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            print(f"✓ Average convolution time: {avg_time*1000:.2f} ms")
            
            return {
                'avg_conv_time_ms': avg_time * 1000,
                'throughput_fps': 1.0 / avg_time
            }
            
    except Exception as e:
        print(f"❌ GPU benchmark failed: {e}")
        return None


def validate_environment():
    """Comprehensive environment validation."""
    print("=== TensorFlow DewarpNet Environment Validation ===\n")
    
    validation_results = {
        'tensorflow_version': False,
        'gpu_detection': False,
        'gpu_configuration': False,
        'gpu_computation': False,
        'cuda_compatibility': False,
        'performance_benchmark': None
    }
    
    # Check TensorFlow version
    validation_results['tensorflow_version'] = check_tensorflow_version()
    print()
    
    # Detect GPUs
    gpus = detect_gpus()
    validation_results['gpu_detection'] = len(gpus) > 0
    print()
    
    if gpus:
        # Configure GPU memory
        validation_results['gpu_configuration'] = configure_gpu_memory_growth()
        print()
        
        # Test GPU computation
        validation_results['gpu_computation'] = test_gpu_computation()
        print()
        
        # Check CUDA compatibility
        validation_results['cuda_compatibility'] = check_cuda_compatibility()
        print()
        
        # Benchmark performance
        validation_results['performance_benchmark'] = benchmark_gpu_performance()
        print()
    
    # Print summary
    print("=== Validation Summary ===")
    for key, value in validation_results.items():
        if key == 'performance_benchmark':
            if value:
                print(f"✓ {key}: {value['avg_conv_time_ms']:.2f} ms avg")
            else:
                print(f"❌ {key}: Failed or skipped")
        else:
            status = "✓" if value else "❌"
            print(f"{status} {key}: {value}")
    
    # Overall status
    critical_checks = ['tensorflow_version', 'gpu_detection', 'gpu_computation']
    all_critical_passed = all(validation_results[check] for check in critical_checks)
    
    print(f"\nOverall Status: {'✓ PASSED' if all_critical_passed else '❌ FAILED'}")
    
    if not all_critical_passed:
        print("\nTroubleshooting:")
        if not validation_results['tensorflow_version']:
            print("- Update TensorFlow: pip install tensorflow-gpu>=2.10.0")
        if not validation_results['gpu_detection']:
            print("- Check GPU drivers and CUDA installation")
            print("- Verify GPU is not being used by other processes")
        if not validation_results['gpu_computation']:
            print("- Check GPU memory availability")
            print("- Restart Python session and try again")
    
    return validation_results


def main():
    """Main function for standalone execution."""
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark-only':
        benchmark_gpu_performance()
    else:
        validate_environment()


if __name__ == "__main__":
    main()