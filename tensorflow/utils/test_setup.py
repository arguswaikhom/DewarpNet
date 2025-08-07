#!/usr/bin/env python3
"""
Setup testing utilities to validate TensorFlow DewarpNet installation.
"""

import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional


def test_directory_structure():
    """Test if all required directories exist."""
    print("Testing directory structure...")
    
    required_dirs = [
        "tensorflow",
        "tensorflow/models",
        "tensorflow/loaders", 
        "tensorflow/losses",
        "tensorflow/training",
        "tensorflow/inference",
        "tensorflow/utils"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✓ All required directories exist")
    return True


def test_dataset_links():
    """Test if dataset symbolic links are properly created."""
    print("Testing dataset links...")
    
    data_dir = Path("tensorflow/data")
    if not data_dir.exists():
        print("❌ tensorflow/data directory does not exist")
        return False
    
    expected_links = ["doc3d", "doc3d_100", "doc3d_1000", "input_crop"]
    existing_links = []
    missing_links = []
    
    for link_name in expected_links:
        link_path = data_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            existing_links.append(link_name)
        else:
            missing_links.append(link_name)
    
    if existing_links:
        print(f"✓ Found dataset links: {existing_links}")
    
    if missing_links:
        print(f"⚠️  Missing dataset links: {missing_links}")
        print("   This is expected if datasets are not available")
    
    return len(existing_links) > 0


def test_tensorflow_imports():
    """Test if TensorFlow and related packages can be imported."""
    print("Testing TensorFlow imports...")
    
    required_packages = [
        "tensorflow",
        "numpy", 
        "scipy",
        "cv2",
        "PIL",
        "matplotlib",
        "sklearn"
    ]
    
    import_results = {}
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "PIL":
                from PIL import Image
            elif package == "sklearn":
                import sklearn
            else:
                __import__(package)
            
            import_results[package] = True
            print(f"✓ {package}")
            
        except ImportError as e:
            import_results[package] = False
            print(f"❌ {package}: {e}")
    
    success_count = sum(import_results.values())
    total_count = len(import_results)
    
    print(f"Import success rate: {success_count}/{total_count}")
    return success_count == total_count


def test_basic_tensorflow_operations():
    """Test basic TensorFlow operations."""
    print("Testing basic TensorFlow operations...")
    
    try:
        # Test tensor creation
        a = tf.constant([1, 2, 3, 4])
        b = tf.constant([5, 6, 7, 8])
        c = tf.add(a, b)
        
        expected = [6, 8, 10, 12]
        result = c.numpy().tolist()
        
        if result == expected:
            print("✓ Basic tensor operations work")
        else:
            print(f"❌ Tensor operation failed. Expected {expected}, got {result}")
            return False
        
        # Test neural network layer
        layer = tf.keras.layers.Dense(10, activation='relu')
        input_tensor = tf.random.normal([5, 20])
        output = layer(input_tensor)
        
        if output.shape == (5, 10):
            print("✓ Neural network layer creation works")
        else:
            print(f"❌ Layer output shape incorrect. Expected (5, 10), got {output.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow operations failed: {e}")
        return False


def test_model_creation():
    """Test if we can create a simple model similar to DewarpNet components."""
    print("Testing model creation...")
    
    try:
        # Test UNet-like model creation
        inputs = tf.keras.Input(shape=(256, 256, 3))
        
        # Encoder
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        # Decoder  
        x = tf.keras.layers.Conv2DTranspose(32, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        
        # Output
        outputs = tf.keras.layers.Conv2D(3, 1, activation='tanh')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Test forward pass
        test_input = tf.random.normal([1, 256, 256, 3])
        test_output = model(test_input)
        
        if test_output.shape == (1, 256, 256, 3):
            print("✓ Model creation and forward pass successful")
            print(f"  Model parameters: {model.count_params():,}")
            return True
        else:
            print(f"❌ Model output shape incorrect. Expected (1, 256, 256, 3), got {test_output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_data_pipeline():
    """Test basic data pipeline operations."""
    print("Testing data pipeline...")
    
    try:
        # Create dummy data
        dummy_images = tf.random.normal([10, 64, 64, 3])
        dummy_labels = tf.random.normal([10, 64, 64, 3])
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
        dataset = dataset.batch(2)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Test iteration
        for batch_images, batch_labels in dataset.take(1):
            if batch_images.shape == (2, 64, 64, 3) and batch_labels.shape == (2, 64, 64, 3):
                print("✓ Data pipeline creation and iteration successful")
                return True
            else:
                print(f"❌ Batch shapes incorrect")
                return False
                
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all setup tests."""
    print("=== TensorFlow DewarpNet Setup Test ===\n")
    
    test_results = {
        'directory_structure': test_directory_structure(),
        'dataset_links': test_dataset_links(),
        'tensorflow_imports': test_tensorflow_imports(),
        'basic_operations': test_basic_tensorflow_operations(),
        'model_creation': test_model_creation(),
        'data_pipeline': test_data_pipeline()
    }
    
    print("\n=== Test Summary ===")
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Setup is ready for development.")
    else:
        print("⚠️  Some tests failed. Please check the setup.")
        
        # Provide specific guidance
        if not test_results['tensorflow_imports']:
            print("\nTo fix import issues:")
            print("pip install -r tensorflow/requirements_tf.txt")
        
        if not test_results['directory_structure']:
            print("\nTo fix directory issues:")
            print("python tensorflow/setup_env.py")
    
    return test_results


def main():
    """Main function for standalone execution."""
    run_comprehensive_test()


if __name__ == "__main__":
    main()