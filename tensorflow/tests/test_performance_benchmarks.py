"""
Performance benchmarking tests for TensorFlow DewarpNet implementation.
Compares performance characteristics and validates efficiency.
"""

import unittest
import time
import psutil
import os
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import sys
from typing import Dict, List, Tuple, Any
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import get_model
from losses.loss_factory import LossFactory
from loaders.data_pipeline import TFDataPipeline
from training.metrics import MetricsComputer, PerformanceProfiler


class PerformanceBenchmark:
    """Base class for performance benchmarking."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.results = {}
        self.profiler = PerformanceProfiler()
    
    def measure_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, execution_time)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def measure_memory(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Measure memory usage of a function.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, memory_info)
        """
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_info = {
            'initial_mb': initial_memory,
            'final_mb': final_memory,
            'delta_mb': final_memory - initial_memory
        }
        
        return result, memory_info
    
    def save_results(self, filepath: str):
        """Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


class TestModelPerformanceBenchmarks(unittest.TestCase):
    """Benchmark model performance characteristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.benchmark = PerformanceBenchmark()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_inference_speed(self):
        """Benchmark model inference speed."""
        print("Benchmarking Model Inference Speed...")
        
        # Test configurations
        model_configs = [
            {
                'name': 'UNet-Small',
                'type': 'unet',
                'params': {'input_nc': 3, 'output_nc': 3, 'num_downs': 4, 'ngf': 32},
                'input_shape': (128, 128, 3)
            },
            {
                'name': 'UNet-Medium',
                'type': 'unet', 
                'params': {'input_nc': 3, 'output_nc': 3, 'num_downs': 6, 'ngf': 64},
                'input_shape': (256, 256, 3)
            },
            {
                'name': 'DenseNet-Small',
                'type': 'densenet',
                'params': {'img_size': 64, 'in_channels': 3, 'out_channels': 2, 'filters': 16},
                'input_shape': (64, 64, 3)
            },
            {
                'name': 'DenseNet-Medium',
                'type': 'densenet',
                'params': {'img_size': 128, 'in_channels': 3, 'out_channels': 2, 'filters': 32},
                'input_shape': (128, 128, 3)
            }
        ]
        
        batch_sizes = [1, 2, 4, 8]
        num_iterations = 10
        
        results = {}
        
        for config in model_configs:
            print(f"  Testing {config['name']}...")
            
            # Create model
            model = get_model(config['type'], **config['params'])
            
            # Build model
            dummy_input = tf.random.normal((1,) + config['input_shape'])
            _ = model(dummy_input)
            
            config_results = {
                'parameters': model.count_params(),
                'batch_results': {}
            }
            
            for batch_size in batch_sizes:
                # Create test input
                test_input = tf.random.normal((batch_size,) + config['input_shape'])
                
                # Warmup
                for _ in range(3):
                    _ = model(test_input, training=False)
                
                # Benchmark inference
                times = []
                for _ in range(num_iterations):
                    start_time = time.time()
                    _ = model(test_input, training=False)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                throughput = batch_size / avg_time
                
                config_results['batch_results'][batch_size] = {
                    'avg_time_s': avg_time,
                    'std_time_s': std_time,
                    'throughput_samples_per_s': throughput,
                    'time_per_sample_ms': (avg_time / batch_size) * 1000
                }
                
                print(f"    Batch {batch_size}: {avg_time:.4f}s ± {std_time:.4f}s "
                      f"({throughput:.1f} samples/s)")
            
            results[config['name']] = config_results
        
        self.benchmark.result