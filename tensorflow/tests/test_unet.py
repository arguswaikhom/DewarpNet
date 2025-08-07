"""Unit tests for TensorFlow UNet implementation."""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# Add the tensorflow directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.unet_tf import UnetGenerator, UnetSkipConnectionBlock, create_unet_generator


class TestUnetSkipConnectionBlock(unittest.TestCase):
    """Test cases for UnetSkipConnectionBlock."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        
    def test_innermost_block_shape(self):
        """Test innermost block output shape."""
        block = UnetSkipConnectionBlock(
            outer_nc=512, inner_nc=512, innermost=True
        )
        
        # Build the model with a sample input
        input_tensor = tf.random.normal((1, 4, 4, 512))
        output = block(input_tensor)
        
        # Innermost block should downsample then upsample back to original size
        # It should not concatenate, so output channels = outer_nc
        # The spatial dimensions should remain the same after down+up sampling
        expected_shape = (1, 4, 4, 512)  # Same spatial size, no concatenation
        self.assertEqual(output.shape, expected_shape)
    
    def test_intermediate_block_shape(self):
        """Test intermediate block output shape with skip connection."""
        # Create a simple innermost block first
        innermost = UnetSkipConnectionBlock(
            outer_nc=512, inner_nc=512, innermost=True
        )
        
        # Create intermediate block
        block = UnetSkipConnectionBlock(
            outer_nc=256, inner_nc=512, submodule=innermost
        )
        
        input_tensor = tf.random.normal((1, 8, 8, 256))
        output = block(input_tensor)
        
        # Should concatenate input with upsampled output: 256 + 256 = 512
        expected_shape = (1, 8, 8, 512)
        self.assertEqual(output.shape, expected_shape)
    
    def test_outermost_block_shape(self):
        """Test outermost block output shape."""
        # Create a simple submodule
        submodule = UnetSkipConnectionBlock(
            outer_nc=64, inner_nc=64, innermost=True
        )
        
        block = UnetSkipConnectionBlock(
            outer_nc=3, inner_nc=64, input_nc=3, 
            submodule=submodule, outermost=True
        )
        
        input_tensor = tf.random.normal((1, 16, 16, 3))
        output = block(input_tensor)
        
        # Outermost block should output the final channels without concatenation
        expected_shape = (1, 16, 16, 3)
        self.assertEqual(output.shape, expected_shape)


class TestUnetGenerator(unittest.TestCase):
    """Test cases for UnetGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_unet_generator_output_shape(self):
        """Test UNet generator output shape for standard configuration."""
        model = UnetGenerator(
            input_nc=3, output_nc=3, num_downs=7, ngf=64
        )
        
        # Test with 256x256 input (standard for world coordinate prediction)
        input_tensor = tf.random.normal((1, 256, 256, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 256, 256, 3)
        self.assertEqual(output.shape, expected_shape)
    
    def test_unet_generator_different_sizes(self):
        """Test UNet generator with different input sizes."""
        model = UnetGenerator(
            input_nc=3, output_nc=3, num_downs=6, ngf=64
        )
        
        # Test with 128x128 input
        input_tensor = tf.random.normal((1, 128, 128, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 128, 128, 3)
        self.assertEqual(output.shape, expected_shape)
    
    def test_unet_generator_batch_processing(self):
        """Test UNet generator with batch input."""
        model = UnetGenerator(
            input_nc=3, output_nc=3, num_downs=7, ngf=64
        )
        
        # Test with batch size > 1
        batch_size = 4
        input_tensor = tf.random.normal((batch_size, 256, 256, 3))
        output = model(input_tensor)
        
        expected_shape = (batch_size, 256, 256, 3)
        self.assertEqual(output.shape, expected_shape)
    
    def test_unet_generator_parameter_count(self):
        """Test UNet generator parameter count."""
        model = UnetGenerator(
            input_nc=3, output_nc=3, num_downs=7, ngf=64
        )
        
        # Build the model
        input_tensor = tf.random.normal((1, 256, 256, 3))
        _ = model(input_tensor)
        
        # Count parameters
        total_params = model.count_params()
        
        # The parameter count should be substantial for a 7-level UNet
        # This is approximately equivalent to the PyTorch version
        self.assertGreater(total_params, 35_000_000)  # Should be around 37-38M parameters
        self.assertLess(total_params, 45_000_000)
        
        print(f"UNet Generator total parameters: {total_params:,}")
    
    def test_create_unet_generator_factory(self):
        """Test the factory function for creating UNet generator."""
        model = create_unet_generator(
            input_nc=3, output_nc=3, num_downs=7, ngf=64
        )
        
        self.assertIsInstance(model, UnetGenerator)
        
        # Test forward pass
        input_tensor = tf.random.normal((1, 256, 256, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 256, 256, 3)
        self.assertEqual(output.shape, expected_shape)
    
    def test_unet_generator_training_mode(self):
        """Test UNet generator in training vs inference mode."""
        model = UnetGenerator(
            input_nc=3, output_nc=3, num_downs=7, ngf=64, use_dropout=True
        )
        
        input_tensor = tf.random.normal((2, 256, 256, 3))
        
        # Test training mode
        output_train = model(input_tensor, training=True)
        
        # Test inference mode
        output_inference = model(input_tensor, training=False)
        
        # Shapes should be the same
        self.assertEqual(output_train.shape, output_inference.shape)
        
        # Outputs might be different due to dropout (if used)
        # But shapes should match expected
        expected_shape = (2, 256, 256, 3)
        self.assertEqual(output_train.shape, expected_shape)
        self.assertEqual(output_inference.shape, expected_shape)
    
    def test_unet_output_range(self):
        """Test that UNet output is in the expected range due to Tanh activation."""
        model = UnetGenerator(
            input_nc=3, output_nc=3, num_downs=7, ngf=64
        )
        
        input_tensor = tf.random.normal((1, 256, 256, 3))
        output = model(input_tensor)
        
        # Output should be in range [-1, 1] due to Tanh activation
        self.assertGreaterEqual(tf.reduce_min(output).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(output).numpy(), 1.0)


if __name__ == '__main__':
    # Enable GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    unittest.main(verbosity=2)