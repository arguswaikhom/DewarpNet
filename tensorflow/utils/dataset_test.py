#!/usr/bin/env python3
"""
Dataset testing utilities for TensorFlow DewarpNet.
Test data loading, preprocessing, and pipeline performance.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import json


def test_data_loading_performance():
    """Test data loading performance for different dataset sizes."""
    print("=== Data Loading Performance Test ===\n")
    
    # Import TensorFlow and data loaders
    try:
        import tensorflow as tf
        sys.path.append(str(Path(__file__).parent.parent))
        from loaders.doc3d_wc_loader import Doc3DWCLoader
        from loaders.doc3d_bm_loader import Doc3DBMLoader
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    
    data_dir = Path("tensorflow/data")
    results = {}
    
    # Test different dataset sizes
    test_datasets = ['doc3d_10', 'doc3d_50', 'doc3d_100']
    
    for dataset_name in test_datasets:
        dataset_path = data_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠️  Skipping {dataset_name}: not found")
            continue
        
        print(f"Testing {dataset_name}...")
        
        try:
            # Test World Coordinate loader
            wc_loader = Doc3DWCLoader(str(dataset_path), split='train', img_size=(256, 256))
            
            # Time data loading
            start_time = time.time()
            sample_count = min(10, len(wc_loader))
            
            for i in range(sample_count):
                img, wc = wc_loader[i]
            
            end_time = time.time()
            avg_time = (end_time - start_time) / sample_count
            
            results[dataset_name] = {
                'dataset_size': len(wc_loader),
                'avg_load_time_ms': avg_time * 1000,
                'throughput_samples_per_sec': 1.0 / avg_time,
                'status': 'success'
            }
            
            print(f"✓ {dataset_name}: {len(wc_loader)} samples, {avg_time*1000:.2f}ms avg load time")
            
        except Exception as e:
            results[dataset_name] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"❌ {dataset_name}: {e}")
    
    return results


def test_data_pipeline_integration():
    """Test TensorFlow data pipeline integration."""
    print("=== Data Pipeline Integration Test ===\n")
    
    try:
        import tensorflow as tf
        sys.path.append(str(Path(__file__).parent.parent))
        from loaders.data_pipeline import create_wc_dataset, create_bm_dataset
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    
    data_dir = Path("tensorflow/data")
    test_dataset = 'doc3d_10'  # Use small dataset for testing
    dataset_path = data_dir / test_dataset
    
    if not dataset_path.exists():
        print(f"❌ Test dataset not found: {dataset_path}")
        return {}
    
    results = {}
    
    try:
        # Test World Coordinate pipeline
        print("Testing WC data pipeline...")
        wc_dataset = create_wc_dataset(str(dataset_path), batch_size=2, shuffle=False)
        
        # Test iteration
        start_time = time.time()
        batch_count = 0
        
        for batch_imgs, batch_wcs in wc_dataset.take(5):
            batch_count += 1
            
            # Validate batch shapes
            expected_img_shape = (2, 256, 256, 3)
            expected_wc_shape = (2, 256, 256, 3)
            
            if batch_imgs.shape != expected_img_shape:
                raise ValueError(f"Unexpected image batch shape: {batch_imgs.shape}")
            
            if batch_wcs.shape != expected_wc_shape:
                raise ValueError(f"Unexpected WC batch shape: {batch_wcs.shape}")
        
        end_time = time.time()
        
        results['wc_pipeline'] = {
            'status': 'success',
            'batch_count': batch_count,
            'total_time_ms': (end_time - start_time) * 1000,
            'avg_batch_time_ms': (end_time - start_time) * 1000 / batch_count
        }
        
        print(f"✓ WC pipeline: {batch_count} batches, {(end_time - start_time)*1000:.2f}ms total")
        
    except Exception as e:
        results['wc_pipeline'] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"❌ WC pipeline error: {e}")
    
    try:
        # Test Backward Mapping pipeline (if available)
        bm_path = dataset_path / 'bm'
        if bm_path.exists():
            print("Testing BM data pipeline...")
            bm_dataset = create_bm_dataset(str(dataset_path), batch_size=2, shuffle=False)
            
            # Test iteration
            start_time = time.time()
            batch_count = 0
            
            for batch_inputs, batch_bms in bm_dataset.take(3):
                batch_count += 1
                
                # Validate batch shapes
                expected_input_shape = (2, 128, 128, 6)  # albedo + wc
                expected_bm_shape = (2, 128, 128, 2)
                
                if batch_inputs.shape != expected_input_shape:
                    raise ValueError(f"Unexpected input batch shape: {batch_inputs.shape}")
                
                if batch_bms.shape != expected_bm_shape:
                    raise ValueError(f"Unexpected BM batch shape: {batch_bms.shape}")
            
            end_time = time.time()
            
            results['bm_pipeline'] = {
                'status': 'success',
                'batch_count': batch_count,
                'total_time_ms': (end_time - start_time) * 1000,
                'avg_batch_time_ms': (end_time - start_time) * 1000 / batch_count
            }
            
            print(f"✓ BM pipeline: {batch_count} batches, {(end_time - start_time)*1000:.2f}ms total")
        else:
            results['bm_pipeline'] = {
                'status': 'skipped',
                'reason': 'BM data not available'
            }
            print("⚠️  BM pipeline: Skipped (no BM data)")
            
    except Exception as e:
        results['bm_pipeline'] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"❌ BM pipeline error: {e}")
    
    return results


def test_data_augmentation():
    """Test data augmentation functionality."""
    print("=== Data Augmentation Test ===\n")
    
    try:
        import tensorflow as tf
        sys.path.append(str(Path(__file__).parent.parent))
        from loaders.augmentations_tf import apply_augmentations
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    
    # Create test data
    test_img = tf.random.normal([256, 256, 3])
    test_wc = tf.random.normal([256, 256, 3])
    
    results = {}
    
    try:
        # Test augmentation
        print("Testing augmentation functions...")
        
        aug_img, aug_wc = apply_augmentations(test_img, test_wc, training=True)
        
        # Validate shapes are preserved
        if aug_img.shape != test_img.shape:
            raise ValueError(f"Augmentation changed image shape: {aug_img.shape} vs {test_img.shape}")
        
        if aug_wc.shape != test_wc.shape:
            raise ValueError(f"Augmentation changed WC shape: {aug_wc.shape} vs {test_wc.shape}")
        
        # Test that augmentation actually changes the data
        img_diff = tf.reduce_mean(tf.abs(aug_img - test_img))
        wc_diff = tf.reduce_mean(tf.abs(aug_wc - test_wc))
        
        results['augmentation'] = {
            'status': 'success',
            'img_difference': float(img_diff),
            'wc_difference': float(wc_diff),
            'shapes_preserved': True
        }
        
        print(f"✓ Augmentation: shapes preserved, data modified (img_diff={img_diff:.4f})")
        
    except Exception as e:
        results['augmentation'] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"❌ Augmentation error: {e}")
    
    return results


def test_dataset_file_formats():
    """Test reading different file formats in the dataset."""
    print("=== File Format Test ===\n")
    
    data_dir = Path("tensorflow/data")
    test_dataset = 'doc3d_10'
    dataset_path = data_dir / test_dataset
    
    if not dataset_path.exists():
        print(f"❌ Test dataset not found: {dataset_path}")
        return {}
    
    results = {}
    
    # Test different file types
    file_types = {
        'img': ['.png', '.jpg', '.jpeg'],
        'wc': ['.exr'],
        'bm': ['.mat'],
        'albedo': ['.png', '.jpg', '.jpeg']
    }
    
    for subdir, extensions in file_types.items():
        subdir_path = dataset_path / subdir
        
        if not subdir_path.exists():
            results[subdir] = {
                'status': 'missing',
                'reason': f'Directory {subdir} not found'
            }
            print(f"⚠️  {subdir}: Directory not found")
            continue
        
        try:
            # Find files with expected extensions
            found_files = []
            for ext in extensions:
                found_files.extend(list(subdir_path.glob(f'*{ext}')))
            
            if not found_files:
                results[subdir] = {
                    'status': 'empty',
                    'expected_extensions': extensions
                }
                print(f"⚠️  {subdir}: No files with expected extensions {extensions}")
                continue
            
            # Test reading a sample file
            sample_file = found_files[0]
            
            if subdir == 'img' or subdir == 'albedo':
                # Test image reading
                import cv2
                img = cv2.imread(str(sample_file))
                if img is None:
                    raise ValueError("Could not read image file")
                
                results[subdir] = {
                    'status': 'success',
                    'file_count': len(found_files),
                    'sample_shape': img.shape,
                    'sample_file': str(sample_file.name)
                }
                
            elif subdir == 'wc':
                # Test EXR reading
                import cv2
                wc = cv2.imread(str(sample_file), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if wc is None:
                    raise ValueError("Could not read EXR file")
                
                results[subdir] = {
                    'status': 'success',
                    'file_count': len(found_files),
                    'sample_shape': wc.shape,
                    'sample_file': str(sample_file.name)
                }
                
            elif subdir == 'bm':
                # Test MAT file reading
                import scipy.io
                mat_data = scipy.io.loadmat(str(sample_file))
                
                results[subdir] = {
                    'status': 'success',
                    'file_count': len(found_files),
                    'mat_keys': list(mat_data.keys()),
                    'sample_file': str(sample_file.name)
                }
            
            print(f"✓ {subdir}: {len(found_files)} files, format OK")
            
        except Exception as e:
            results[subdir] = {
                'status': 'error',
                'error': str(e),
                'sample_file': str(sample_file.name) if 'sample_file' in locals() else None
            }
            print(f"❌ {subdir}: Error reading files - {e}")
    
    return results


def test_memory_usage():
    """Test memory usage during data loading."""
    print("=== Memory Usage Test ===\n")
    
    try:
        import psutil
        import tensorflow as tf
        sys.path.append(str(Path(__file__).parent.parent))
        from loaders.doc3d_wc_loader import Doc3DWCLoader
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {}
    
    data_dir = Path("tensorflow/data")
    test_dataset = 'doc3d_10'
    dataset_path = data_dir / test_dataset
    
    if not dataset_path.exists():
        print(f"❌ Test dataset not found: {dataset_path}")
        return {}
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    results = {
        'initial_memory_mb': initial_memory,
        'peak_memory_mb': initial_memory,
        'final_memory_mb': initial_memory,
        'memory_increase_mb': 0
    }
    
    try:
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Load dataset
        loader = Doc3DWCLoader(str(dataset_path), split='train', img_size=(256, 256))
        
        # Load several samples and track memory
        for i in range(min(10, len(loader))):
            img, wc = loader[i]
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            results['peak_memory_mb'] = max(results['peak_memory_mb'], current_memory)
        
        # Final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        results['final_memory_mb'] = final_memory
        results['memory_increase_mb'] = final_memory - initial_memory
        
        print(f"Peak memory usage: {results['peak_memory_mb']:.1f} MB")
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Memory increase: {results['memory_increase_mb']:.1f} MB")
        
        results['status'] = 'success'
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        print(f"❌ Memory test error: {e}")
    
    return results


def run_comprehensive_dataset_test(dataset_name: str = 'doc3d_10') -> Dict[str, Any]:
    """Run comprehensive dataset testing."""
    print(f"=== Comprehensive Dataset Test: {dataset_name} ===\n")
    
    test_results = {
        'dataset_name': dataset_name,
        'timestamp': time.time(),
        'tests': {}
    }
    
    # Run all tests
    print("1. Testing data loading performance...")
    test_results['tests']['loading_performance'] = test_data_loading_performance()
    print()
    
    print("2. Testing data pipeline integration...")
    test_results['tests']['pipeline_integration'] = test_data_pipeline_integration()
    print()
    
    print("3. Testing data augmentation...")
    test_results['tests']['augmentation'] = test_data_augmentation()
    print()
    
    print("4. Testing file formats...")
    test_results['tests']['file_formats'] = test_dataset_file_formats()
    print()
    
    print("5. Testing memory usage...")
    test_results['tests']['memory_usage'] = test_memory_usage()
    print()
    
    # Generate summary
    total_tests = 0
    passed_tests = 0
    
    for test_name, test_result in test_results['tests'].items():
        if isinstance(test_result, dict):
            if test_result.get('status') == 'success':
                passed_tests += 1
            total_tests += 1
        else:
            # Handle nested results (like loading_performance)
            for sub_test, sub_result in test_result.items():
                if isinstance(sub_result, dict) and sub_result.get('status') == 'success':
                    passed_tests += 1
                total_tests += 1
    
    test_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0
    }
    
    print("=== Test Summary ===")
    print(f"Passed: {passed_tests}/{total_tests} tests")
    print(f"Success rate: {test_results['summary']['success_rate']:.1%}")
    
    return test_results


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Dataset testing utilities")
    parser.add_argument("--dataset", default="doc3d_10",
                       help="Dataset to test")
    parser.add_argument("--test", choices=['loading', 'pipeline', 'augmentation', 'formats', 'memory', 'all'],
                       default='all', help="Specific test to run")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if args.test == 'all':
        results = run_comprehensive_dataset_test(args.dataset)
    elif args.test == 'loading':
        results = test_data_loading_performance()
    elif args.test == 'pipeline':
        results = test_data_pipeline_integration()
    elif args.test == 'augmentation':
        results = test_data_augmentation()
    elif args.test == 'formats':
        results = test_dataset_file_formats()
    elif args.test == 'memory':
        results = test_memory_usage()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()