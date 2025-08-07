"""
PyTorch-TensorFlow comparison utilities for DewarpNet integration tests.
Provides tools to compare model architectures, outputs, and performance.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
import json

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.model_factory import get_model as get_tf_model


class PyTorchTensorFlowComparator:
    """Compare PyTorch and TensorFlow implementations."""
    
    def __init__(self):
        """Initialize comparator."""
        self.results = {}
        self.pytorch_available = self._check_pytorch_availability()
    
    def _check_pytorch_availability(self) -> bool:
        """Check if PyTorch models are available for comparison."""
        try:
            # Try to import PyTorch models
            from models.unetnc import UnetGenerator
            from models.densenetccnl import dnetccnl
            return True
        except ImportError:
            print("⚠ PyTorch models not available for comparison")
            return False
    
    def compare_model_architectures(self) -> Dict[str, Any]:
        """Compare model architectures between PyTorch and TensorFlow."""
        print("Comparing Model Architectures...")
        
        results = {
            'unet_comparison': {},
            'densenet_comparison': {},
            'pytorch_available': self.pytorch_available
        }
        
        if not self.pytorch_available:
            print("  ⚠ Skipping architecture comparison - PyTorch models not available")
            return results
        
        # Compare UNet architectures
        try:
            results['unet_comparison'] = self._compare_unet_architectures()
        except Exception as e:
            print(f"  ❌ UNet comparison failed: {e}")
            results['unet_comparison'] = {'error': str(e)}
        
        # Compare DenseNet architectures
        try:
            results['densenet_comparison'] = self._compare_densenet_architectures()
        except Exception as e:
            print(f"  ❌ DenseNet comparison failed: {e}")
            results['densenet_comparison'] = {'error': str(e)}
        
        return results
    
    def _compare_unet_architectures(self) -> Dict[str, Any]:
        """Compare UNet architectures."""
        from models.unetnc import UnetGenerator
        import torch
        
        # Create PyTorch model
        pytorch_model = UnetGenerator(input_nc=3, output_nc=3, num_downs=6, ngf=64)
        pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
        
        # Create TensorFlow model
        tf_model = get_tf_model('unet', input_nc=3, output_nc=3, num_downs=6, ngf=64)
        dummy_input = tf.random.normal((1, 256, 256, 3))
        _ = tf_model(dummy_input)
        tf_params = tf_model.count_params()
        
        # Compare outputs
        test_input_np = np.random.randn(1, 3, 256, 256).astype(np.float32)
        
        # PyTorch forward pass
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input_np)
            pytorch_output = pytorch_model(pytorch_input).numpy()
        
        # TensorFlow forward pass (convert NCHW to NHWC)
        tf_input = tf.transpose(tf.constant(test_input_np), [0, 2, 3, 1])
        tf_output = tf_model(tf_input, training=False)
        tf_output_np = tf.transpose(tf_output, [0, 3, 1, 2]).numpy()
        
        # Compare shapes and parameter counts
        shape_match = pytorch_output.shape == tf_output_np.shape
        param_ratio = tf_params / pytorch_params
        
        return {
            'pytorch_params': int(pytorch_params),
            'tensorflow_params': int(tf_params),
            'param_ratio': float(param_ratio),
            'param_difference_pct': float(abs(param_ratio - 1.0) * 100),
            'output_shape_match': shape_match,
            'pytorch_output_shape': pytorch_output.shape,
            'tensorflow_output_shape': tf_output_np.shape,
            'output_range_pytorch': [float(pytorch_output.min()), float(pytorch_output.max())],
            'output_range_tensorflow': [float(tf_output_np.min()), float(tf_output_np.max())]
        }
    
    def _compare_densenet_architectures(self) -> Dict[str, Any]:
        """Compare DenseNet architectures."""
        from models.densenetccnl import dnetccnl
        import torch
        
        # Create PyTorch model
        pytorch_model = dnetccnl(img_size=128, in_channels=3, out_channels=2, filters=32)
        pytorch_params = sum(p.numel() for p in pytorch_model.parameters())
        
        # Create TensorFlow model
        tf_model = get_tf_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=32)
        dummy_input = tf.random.normal((1, 128, 128, 3))
        _ = tf_model(dummy_input)
        tf_params = tf_model.count_params()
        
        # Compare outputs
        test_input_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
        
        # PyTorch forward pass
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input_np)
            pytorch_output = pytorch_model(pytorch_input).numpy()
        
        # TensorFlow forward pass (convert NCHW to NHWC)
        tf_input = tf.transpose(tf.constant(test_input_np), [0, 2, 3, 1])
        tf_output = tf_model(tf_input, training=False)
        tf_output_np = tf.transpose(tf_output, [0, 3, 1, 2]).numpy()
        
        # Compare shapes and parameter counts
        shape_match = pytorch_output.shape == tf_output_np.shape
        param_ratio = tf_params / pytorch_params
        
        return {
            'pytorch_params': int(pytorch_params),
            'tensorflow_params': int(tf_params),
            'param_ratio': float(param_ratio),
            'param_difference_pct': float(abs(param_ratio - 1.0) * 100),
            'output_shape_match': shape_match,
            'pytorch_output_shape': pytorch_output.shape,
            'tensorflow_output_shape': tf_output_np.shape,
            'output_range_pytorch': [float(pytorch_output.min()), float(pytorch_output.max())],
            'output_range_tensorflow': [float(tf_output_np.min()), float(tf_output_np.max())]
        }
    
    def compare_inference_performance(self, num_iterations: int = 10) -> Dict[str, Any]:
        """Compare inference performance between PyTorch and TensorFlow."""
        print(f"Comparing Inference Performance ({num_iterations} iterations)...")
        
        results = {
            'pytorch_available': self.pytorch_available,
            'num_iterations': num_iterations
        }
        
        if not self.pytorch_available:
            print("  ⚠ Skipping performance comparison - PyTorch models not available")
            return results
        
        # Compare UNet performance
        try:
            results['unet_performance'] = self._benchmark_unet_performance(num_iterations)
        except Exception as e:
            print(f"  ❌ UNet performance comparison failed: {e}")
            results['unet_performance'] = {'error': str(e)}
        
        # Compare DenseNet performance
        try:
            results['densenet_performance'] = self._benchmark_densenet_performance(num_iterations)
        except Exception as e:
            print(f"  ❌ DenseNet performance comparison failed: {e}")
            results['densenet_performance'] = {'error': str(e)}
        
        return results
    
    def _benchmark_unet_performance(self, num_iterations: int) -> Dict[str, Any]:
        """Benchmark UNet performance."""
        from models.unetnc import UnetGenerator
        import torch
        
        # Create models
        pytorch_model = UnetGenerator(input_nc=3, output_nc=3, num_downs=5, ngf=32)
        pytorch_model.eval()
        
        tf_model = get_tf_model('unet', input_nc=3, output_nc=3, num_downs=5, ngf=32)
        dummy_input = tf.random.normal((1, 128, 128, 3))
        _ = tf_model(dummy_input)
        
        # Prepare inputs
        test_input_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
        pytorch_input = torch.from_numpy(test_input_np)
        tf_input = tf.transpose(tf.constant(test_input_np), [0, 2, 3, 1])
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = pytorch_model(pytorch_input)
            _ = tf_model(tf_input, training=False)
        
        # Benchmark PyTorch
        pytorch_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = pytorch_model(pytorch_input)
            pytorch_times.append(time.time() - start_time)
        
        # Benchmark TensorFlow
        tf_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = tf_model(tf_input, training=False)
            tf_times.append(time.time() - start_time)
        
        pytorch_avg = np.mean(pytorch_times)
        tf_avg = np.mean(tf_times)
        speedup = pytorch_avg / tf_avg
        
        return {
            'pytorch_avg_time': float(pytorch_avg),
            'tensorflow_avg_time': float(tf_avg),
            'pytorch_std_time': float(np.std(pytorch_times)),
            'tensorflow_std_time': float(np.std(tf_times)),
            'speedup_ratio': float(speedup),
            'tensorflow_faster': speedup > 1.0
        }
    
    def _benchmark_densenet_performance(self, num_iterations: int) -> Dict[str, Any]:
        """Benchmark DenseNet performance."""
        from models.densenetccnl import dnetccnl
        import torch
        
        # Create models
        pytorch_model = dnetccnl(img_size=128, in_channels=3, out_channels=2, filters=16)
        pytorch_model.eval()
        
        tf_model = get_tf_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=16)
        dummy_input = tf.random.normal((1, 128, 128, 3))
        _ = tf_model(dummy_input)
        
        # Prepare inputs
        test_input_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
        pytorch_input = torch.from_numpy(test_input_np)
        tf_input = tf.transpose(tf.constant(test_input_np), [0, 2, 3, 1])
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = pytorch_model(pytorch_input)
            _ = tf_model(tf_input, training=False)
        
        # Benchmark PyTorch
        pytorch_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = pytorch_model(pytorch_input)
            pytorch_times.append(time.time() - start_time)
        
        # Benchmark TensorFlow
        tf_times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = tf_model(tf_input, training=False)
            tf_times.append(time.time() - start_time)
        
        pytorch_avg = np.mean(pytorch_times)
        tf_avg = np.mean(tf_times)
        speedup = pytorch_avg / tf_avg
        
        return {
            'pytorch_avg_time': float(pytorch_avg),
            'tensorflow_avg_time': float(tf_avg),
            'pytorch_std_time': float(np.std(pytorch_times)),
            'tensorflow_std_time': float(np.std(tf_times)),
            'speedup_ratio': float(speedup),
            'tensorflow_faster': speedup > 1.0
        }
    
    def generate_comparison_report(self, output_path: str):
        """Generate comprehensive comparison report."""
        print("Generating Comparison Report...")
        
        # Run all comparisons
        architecture_results = self.compare_model_architectures()
        performance_results = self.compare_inference_performance()
        
        # Combine results
        full_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tensorflow_version': tf.__version__,
            'architecture_comparison': architecture_results,
            'performance_comparison': performance_results
        }
        
        # Save JSON results
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        # Generate human-readable report
        with open(output_path, 'w') as f:
            f.write("PyTorch-TensorFlow Comparison Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {full_results['timestamp']}\n")
            f.write(f"TensorFlow Version: {tf.__version__}\n")
            f.write(f"PyTorch Available: {self.pytorch_available}\n\n")
            
            if self.pytorch_available:
                # Architecture comparison
                f.write("Architecture Comparison\n")
                f.write("-" * 25 + "\n")
                
                if 'unet_comparison' in architecture_results:
                    unet = architecture_results['unet_comparison']
                    if 'error' not in unet:
                        f.write(f"UNet Parameters:\n")
                        f.write(f"  PyTorch: {unet['pytorch_params']:,}\n")
                        f.write(f"  TensorFlow: {unet['tensorflow_params']:,}\n")
                        f.write(f"  Difference: {unet['param_difference_pct']:.2f}%\n")
                        f.write(f"  Shape Match: {unet['output_shape_match']}\n\n")
                
                if 'densenet_comparison' in architecture_results:
                    densenet = architecture_results['densenet_comparison']
                    if 'error' not in densenet:
                        f.write(f"DenseNet Parameters:\n")
                        f.write(f"  PyTorch: {densenet['pytorch_params']:,}\n")
                        f.write(f"  TensorFlow: {densenet['tensorflow_params']:,}\n")
                        f.write(f"  Difference: {densenet['param_difference_pct']:.2f}%\n")
                        f.write(f"  Shape Match: {densenet['output_shape_match']}\n\n")
                
                # Performance comparison
                f.write("Performance Comparison\n")
                f.write("-" * 22 + "\n")
                
                if 'unet_performance' in performance_results:
                    unet_perf = performance_results['unet_performance']
                    if 'error' not in unet_perf:
                        f.write(f"UNet Inference Time:\n")
                        f.write(f"  PyTorch: {unet_perf['pytorch_avg_time']:.4f}s ± {unet_perf['pytorch_std_time']:.4f}s\n")
                        f.write(f"  TensorFlow: {unet_perf['tensorflow_avg_time']:.4f}s ± {unet_perf['tensorflow_std_time']:.4f}s\n")
                        f.write(f"  Speedup: {unet_perf['speedup_ratio']:.2f}x\n")
                        f.write(f"  TensorFlow Faster: {unet_perf['tensorflow_faster']}\n\n")
                
                if 'densenet_performance' in performance_results:
                    densenet_perf = performance_results['densenet_performance']
                    if 'error' not in densenet_perf:
                        f.write(f"DenseNet Inference Time:\n")
                        f.write(f"  PyTorch: {densenet_perf['pytorch_avg_time']:.4f}s ± {densenet_perf['pytorch_std_time']:.4f}s\n")
                        f.write(f"  TensorFlow: {densenet_perf['tensorflow_avg_time']:.4f}s ± {densenet_perf['tensorflow_std_time']:.4f}s\n")
                        f.write(f"  Speedup: {densenet_perf['speedup_ratio']:.2f}x\n")
                        f.write(f"  TensorFlow Faster: {densenet_perf['tensorflow_faster']}\n\n")
            else:
                f.write("PyTorch models not available for comparison.\n")
                f.write("To enable comparison, ensure PyTorch models are in the parent directory.\n")
        
        print(f"✓ Comparison report saved to: {output_path}")
        print(f"✓ JSON results saved to: {json_path}")


def main():
    """Main entry point for comparison utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare PyTorch and TensorFlow DewarpNet implementations')
    parser.add_argument('--output', '-o', default='pytorch_tf_comparison.txt',
                       help='Output file for comparison report')
    parser.add_argument('--iterations', '-i', type=int, default=10,
                       help='Number of iterations for performance benchmarking')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = PyTorchTensorFlowComparator()
    
    # Generate report
    comparator.generate_comparison_report(args.output)


if __name__ == '__main__':
    main()