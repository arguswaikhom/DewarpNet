"""Unit tests for TensorFlow DenseNet implementation."""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os

# Add the tensorflow directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.densenet_tf import (
    DenseBlockEncoder, DenseBlockDecoder,
    DenseTransitionBlockEncoder, DenseTransitionBlockDecoder,
    WaspDenseEncoder128, WaspDenseDecoder128, DNetCCNL,
    add_coordconv_channels, create_dnet_ccnl
)


class TestCoordConvChannels(unittest.TestCase):
    """Test cases for coordinate channel addition."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_coordconv_channels_shape(self):
        """Test that coordinate channels are added correctly."""
        input_tensor = tf.random.normal((2, 64, 64, 3))
        output = add_coordconv_channels(input_tensor)
        
        # Should add 2 coordinate channels
        expected_shape = (2, 64, 64, 5)
        self.assertEqual(output.shape, expected_shape)
    
    def test_coordconv_channels_range(self):
        """Test that coordinate channels are in correct range [-1, 1]."""
        input_tensor = tf.random.normal((1, 32, 32, 1))
        output = add_coordconv_channels(input_tensor)
        
        # Extract coordinate channels
        xx_coord = output[0, :, :, -2]
        yy_coord = output[0, :, :, -1]
        
        # Check range
        self.assertGreaterEqual(tf.reduce_min(xx_coord).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(xx_coord).numpy(), 1.0)
        self.assertGreaterEqual(tf.reduce_min(yy_coord).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(yy_coord).numpy(), 1.0)


class TestDenseBlocks(unittest.TestCase):
    """Test cases for Dense blocks."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_dense_block_encoder_shape(self):
        """Test DenseBlockEncoder output shape."""
        block = DenseBlockEncoder(n_channels=64, n_convs=6)
        
        input_tensor = tf.random.normal((2, 32, 32, 64))
        output = block(input_tensor)
        
        # Output should have same spatial dimensions and channels
        expected_shape = (2, 32, 32, 64)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dense_block_decoder_shape(self):
        """Test DenseBlockDecoder output shape."""
        block = DenseBlockDecoder(n_channels=64, n_convs=6)
        
        input_tensor = tf.random.normal((2, 32, 32, 64))
        output = block(input_tensor)
        
        # Output should have same spatial dimensions and channels
        expected_shape = (2, 32, 32, 64)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dense_transition_encoder_shape(self):
        """Test DenseTransitionBlockEncoder output shape."""
        block = DenseTransitionBlockEncoder(
            n_channels_in=64, n_channels_out=128, mp=2
        )
        
        input_tensor = tf.random.normal((2, 32, 32, 64))
        output = block(input_tensor)
        
        # Should downsample by factor of 2 and change channels
        expected_shape = (2, 16, 16, 128)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dense_transition_decoder_shape(self):
        """Test DenseTransitionBlockDecoder output shape."""
        block = DenseTransitionBlockDecoder(
            n_channels_in=128, n_channels_out=64
        )
        
        input_tensor = tf.random.normal((2, 16, 16, 128))
        output = block(input_tensor)
        
        # Should upsample by factor of 2 and change channels
        expected_shape = (2, 32, 32, 64)
        self.assertEqual(output.shape, expected_shape)


class TestWaspDenseEncoder128(unittest.TestCase):
    """Test cases for WaspDenseEncoder128."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_encoder_output_shape(self):
        """Test encoder output shape for 128x128 input."""
        encoder = WaspDenseEncoder128(nc=3, ndf=32, ndim=128)
        
        input_tensor = tf.random.normal((2, 128, 128, 3))
        output = encoder(input_tensor)
        
        # Should output flattened vector of size ndim
        expected_shape = (2, 128)
        self.assertEqual(output.shape, expected_shape)
    
    def test_encoder_different_channels(self):
        """Test encoder with different input channels."""
        encoder = WaspDenseEncoder128(nc=1, ndf=32, ndim=256)
        
        input_tensor = tf.random.normal((1, 128, 128, 1))
        output = encoder(input_tensor)
        
        expected_shape = (1, 256)
        self.assertEqual(output.shape, expected_shape)
    
    def test_encoder_parameter_count(self):
        """Test encoder parameter count."""
        encoder = WaspDenseEncoder128(nc=3, ndf=32, ndim=128)
        
        # Build the model
        input_tensor = tf.random.normal((1, 128, 128, 3))
        _ = encoder(input_tensor)
        
        # Count parameters
        total_params = encoder.count_params()
        
        # Should have substantial number of parameters
        self.assertGreater(total_params, 1_000_000)
        print(f"WaspDenseEncoder128 total parameters: {total_params:,}")


class TestWaspDenseDecoder128(unittest.TestCase):
    """Test cases for WaspDenseDecoder128."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_decoder_output_shape(self):
        """Test decoder output shape."""
        decoder = WaspDenseDecoder128(nz=128, nc=2, ngf=32)
        
        # Input should be 4D tensor (batch, 1, 1, nz)
        input_tensor = tf.random.normal((2, 1, 1, 128))
        output = decoder(input_tensor)
        
        # Should output 128x128 image with nc channels
        expected_shape = (2, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_decoder_different_channels(self):
        """Test decoder with different output channels."""
        decoder = WaspDenseDecoder128(nz=256, nc=1, ngf=32)
        
        input_tensor = tf.random.normal((1, 1, 1, 256))
        output = decoder(input_tensor)
        
        expected_shape = (1, 128, 128, 1)
        self.assertEqual(output.shape, expected_shape)
    
    def test_decoder_parameter_count(self):
        """Test decoder parameter count."""
        decoder = WaspDenseDecoder128(nz=128, nc=2, ngf=32)
        
        # Build the model
        input_tensor = tf.random.normal((1, 1, 1, 128))
        _ = decoder(input_tensor)
        
        # Count parameters
        total_params = decoder.count_params()
        
        # Should have substantial number of parameters
        self.assertGreater(total_params, 1_000_000)
        print(f"WaspDenseDecoder128 total parameters: {total_params:,}")


class TestDNetCCNL(unittest.TestCase):
    """Test cases for complete DNetCCNL model."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_dnet_ccnl_output_shape(self):
        """Test DNetCCNL output shape for standard configuration."""
        model = DNetCCNL(
            img_size=128, in_channels=3, out_channels=2, filters=32
        )
        
        input_tensor = tf.random.normal((2, 128, 128, 3))
        output = model(input_tensor)
        
        # Should output 128x128 image with 2 channels (optical flow)
        expected_shape = (2, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dnet_ccnl_single_channel(self):
        """Test DNetCCNL with single input channel."""
        model = DNetCCNL(
            img_size=128, in_channels=1, out_channels=2, filters=32
        )
        
        input_tensor = tf.random.normal((1, 128, 128, 1))
        output = model(input_tensor)
        
        expected_shape = (1, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dnet_ccnl_batch_processing(self):
        """Test DNetCCNL with batch input."""
        model = DNetCCNL(
            img_size=128, in_channels=3, out_channels=2, filters=32
        )
        
        batch_size = 4
        input_tensor = tf.random.normal((batch_size, 128, 128, 3))
        output = model(input_tensor)
        
        expected_shape = (batch_size, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dnet_ccnl_parameter_count(self):
        """Test DNetCCNL parameter count."""
        model = DNetCCNL(
            img_size=128, in_channels=3, out_channels=2, filters=32
        )
        
        # Build the model
        input_tensor = tf.random.normal((1, 128, 128, 3))
        _ = model(input_tensor)
        
        # Count parameters
        total_params = model.count_params()
        
        # Should have substantial number of parameters
        self.assertGreater(total_params, 5_000_000)
        print(f"DNetCCNL total parameters: {total_params:,}")
    
    def test_dnet_ccnl_training_mode(self):
        """Test DNetCCNL in training vs inference mode."""
        model = DNetCCNL(
            img_size=128, in_channels=3, out_channels=2, filters=32
        )
        
        input_tensor = tf.random.normal((2, 128, 128, 3))
        
        # Test training mode
        output_train = model(input_tensor, training=True)
        
        # Test inference mode
        output_inference = model(input_tensor, training=False)
        
        # Shapes should be the same
        self.assertEqual(output_train.shape, output_inference.shape)
        
        expected_shape = (2, 128, 128, 2)
        self.assertEqual(output_train.shape, expected_shape)
        self.assertEqual(output_inference.shape, expected_shape)
    
    def test_create_dnet_ccnl_factory(self):
        """Test the factory function for creating DNetCCNL."""
        model = create_dnet_ccnl(
            img_size=128, in_channels=3, out_channels=2, filters=32
        )
        
        self.assertIsInstance(model, DNetCCNL)
        
        # Test forward pass
        input_tensor = tf.random.normal((1, 128, 128, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_dnet_output_range(self):
        """Test that DNet output is in reasonable range."""
        model = DNetCCNL(
            img_size=128, in_channels=3, out_channels=2, filters=32
        )
        
        input_tensor = tf.random.normal((1, 128, 128, 3))
        output = model(input_tensor)
        
        # Output should be bounded (due to final activation)
        # The exact range depends on the activation function used
        self.assertGreaterEqual(tf.reduce_min(output).numpy(), 0.0)
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