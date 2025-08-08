#!/usr/bin/env python3
"""
Simple test script to validate TensorFlow DewarpNet training components.
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_factory import ModelFactory
from loaders import Doc3DWCLoader
from utils.gpu_utils import setup_gpu, get_gpu_info

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        # Create world coordinate model
        wc_model = ModelFactory.create_world_coordinate_model()
        
        # Test with dummy input
        dummy_input = tf.random.normal((1, 256, 256, 3))
        output = wc_model(dummy_input)
        
        print(f"✓ WC Model created successfully")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {wc_model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    try:
        # Create data loader
        loader = Doc3DWCLoader(
            root='tensorflow/data/doc3d_1000',
            split='train',
            img_size=(256, 256),
            augmentations=False
        )
        
        print(f"✓ Data loader created successfully")
        print(f"  Dataset size: {len(loader)}")
        
        # Test loading a sample
        if len(loader) > 0:
            img, label = loader[0]
            print(f"  Sample image shape: {img.shape}")
            print(f"  Sample label shape: {label.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_gpu_setup():
    """Test GPU setup."""
    print("\nTesting GPU setup...")
    
    try:
        setup_gpu()
        gpu_info = get_gpu_info()
        
        print(f"✓ GPU setup successful")
        print(f"  GPU available: {gpu_info['gpu_available']}")
        print(f"  GPU count: {gpu_info['gpu_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        return False

def test_simple_training_step():
    """Test a simple training step."""
    print("\nTesting simple training step...")
    
    try:
        # Create model
        model = ModelFactory.create_world_coordinate_model()
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Create dummy data
        batch_size = 2
        dummy_images = tf.random.normal((batch_size, 256, 256, 3))
        dummy_labels = tf.random.normal((batch_size, 256, 256, 3))
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = model(dummy_images, training=True)
            loss = tf.keras.losses.MeanSquaredError()(dummy_labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== TensorFlow DewarpNet Component Tests ===\n")
    
    tests = [
        test_gpu_setup,
        test_model_creation,
        test_data_loading,
        test_simple_training_step
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ All tests passed! Training pipeline should work.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)