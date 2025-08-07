"""Unit tests for reconstruction loss function."""

import tensorflow as tf
import numpy as np
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.recon_loss import ReconLoss, UnwarpLoss, unwarp, bilinear_sample, recon_loss, unwarp_loss
from losses.ssim_loss import SSIMLoss, ssim


class TestReconLoss(unittest.TestCase):
    """Test cases for reconstruction loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 64
        self.width = 64
        self.channels = 6  # RGB + additional channels
        
        # Create test tensors
        np.random.seed(42)
        self.inputs = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
        
        # Create backward mapping coordinates in [-1, 1] range
        self.y_true = tf.constant(
            (np.random.rand(self.batch_size, self.height, self.width, 2).astype(np.float32) - 0.5) * 2
        )
        self.y_pred = tf.constant(
            (np.random.rand(self.batch_size, self.height, self.width, 2).astype(np.float32) - 0.5) * 2
        )
    
    def test_bilinear_sample(self):
        """Test bilinear sampling function."""
        # Create simple test image
        test_img = tf.constant([[[[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0]],
                                [[0.0, 0.0, 1.0],
                                 [1.0, 1.0, 1.0]]]], dtype=tf.float64)
        
        # Sample at integer coordinates - shape should match image spatial dimensions
        x_coords = tf.constant([[[0.0, 1.0], [0.0, 1.0]]], dtype=tf.float64)
        y_coords = tf.constant([[[0.0, 0.0], [1.0, 1.0]]], dtype=tf.float64)
        
        result = bilinear_sample(test_img, x_coords, y_coords)
        
        # Check output shape
        self.assertEqual(result.shape, (1, 2, 2, 3))
        
        # Check that sampling at integer coordinates gives exact values
        expected = test_img[0]
        np.testing.assert_allclose(result[0].numpy(), expected.numpy(), rtol=1e-5)
    
    def test_unwarp_function(self):
        """Test unwarp function."""
        # Create simple test image
        img = tf.random.normal([1, 32, 32, 3])
        
        # Create identity mapping (should return original image)
        h, w = 32, 32
        y_grid, x_grid = tf.meshgrid(tf.linspace(-1.0, 1.0, h), tf.linspace(-1.0, 1.0, w), indexing='ij')
        identity_mapping = tf.stack([x_grid, y_grid], axis=-1)
        identity_mapping = tf.expand_dims(identity_mapping, axis=0)
        
        # Unwarp with identity mapping
        unwarped = unwarp(img, identity_mapping)
        
        # Should be close to original (within interpolation error)
        self.assertEqual(unwarped.shape, img.shape)
        # Note: Due to interpolation, we expect some small differences
        diff = tf.reduce_mean(tf.abs(tf.cast(unwarped, tf.float32) - img))
        self.assertLess(diff.numpy(), 0.1)  # Allow for interpolation error
    
    def test_recon_loss_class(self):
        """Test ReconLoss class."""
        loss_fn = ReconLoss(use_ssim=True)
        
        # Compute loss
        result = loss_fn(self.inputs, self.y_true, self.y_pred)
        
        # Check that result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('total_loss', result)
        self.assertIn('mse_loss', result)
        self.assertIn('ssim_loss', result)
        self.assertIn('uworg', result)
        self.assertIn('uwpred', result)
        
        # Check that losses are positive scalars
        self.assertEqual(result['total_loss'].shape, ())
        self.assertEqual(result['mse_loss'].shape, ())
        self.assertEqual(result['ssim_loss'].shape, ())
        self.assertGreater(result['total_loss'].numpy(), 0)
        self.assertGreater(result['mse_loss'].numpy(), 0)
        
        # Check that unwarped images have correct shape
        expected_shape = (self.batch_size, self.height, self.width, 3)
        self.assertEqual(result['uworg'].shape, expected_shape)
        self.assertEqual(result['uwpred'].shape, expected_shape)
    
    def test_recon_loss_without_ssim(self):
        """Test ReconLoss without SSIM component."""
        loss_fn = ReconLoss(use_ssim=False)
        
        result = loss_fn(self.inputs, self.y_true, self.y_pred)
        
        # SSIM loss should be zero
        self.assertEqual(result['ssim_loss'], 0.0)
        
        # Total loss should equal MSE loss
        self.assertAlmostEqual(
            result['total_loss'].numpy(), 
            result['mse_loss'].numpy(), 
            places=5
        )
    
    def test_unwarp_loss_class(self):
        """Test UnwarpLoss class matching PyTorch implementation."""
        loss_fn = UnwarpLoss()
        
        # Compute loss
        mse_loss, ssim_loss, uworg, uwpred = loss_fn(self.inputs, self.y_pred, self.y_true)
        
        # Check output types and shapes
        self.assertEqual(mse_loss.shape, ())
        self.assertEqual(ssim_loss.shape, ())
        self.assertGreater(mse_loss.numpy(), 0)
        self.assertGreater(ssim_loss.numpy(), 0)
        
        # Check unwarped image shapes
        expected_shape = (self.batch_size, self.height, self.width, 3)
        self.assertEqual(uworg.shape, expected_shape)
        self.assertEqual(uwpred.shape, expected_shape)
    
    def test_functional_interfaces(self):
        """Test functional interfaces."""
        # Test recon_loss function
        result = recon_loss(self.inputs, self.y_true, self.y_pred)
        self.assertIsInstance(result, dict)
        self.assertIn('total_loss', result)
        
        # Test unwarp_loss function
        mse_loss, ssim_loss, uworg, uwpred = unwarp_loss(self.inputs, self.y_pred, self.y_true)
        self.assertEqual(mse_loss.shape, ())
        self.assertEqual(ssim_loss.shape, ())
    
    def test_identical_mappings(self):
        """Test that identical mappings give zero MSE loss."""
        loss_fn = ReconLoss(use_ssim=False)
        
        # Use identical mappings
        result = loss_fn(self.inputs, self.y_true, self.y_true)
        
        # MSE loss should be very small (near zero)
        self.assertLess(result['mse_loss'].numpy(), 1e-5)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        loss_fn = ReconLoss(use_ssim=True, ssim_weight=0.5, mse_weight=2.0)
        config = loss_fn.get_config()
        
        self.assertEqual(config['use_ssim'], True)
        self.assertEqual(config['ssim_weight'], 0.5)
        self.assertEqual(config['mse_weight'], 2.0)


class TestSSIMLoss(unittest.TestCase):
    """Test cases for SSIM loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.channels = 3
        
        # Create test tensors
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
    
    def test_ssim_loss_class(self):
        """Test SSIMLoss class."""
        loss_fn = SSIMLoss()
        
        # Compute loss
        loss_value = loss_fn(self.y_true, self.y_pred)
        
        # Check that loss is a scalar
        self.assertEqual(loss_value.shape, ())
        
        # SSIM loss should be between 0 and 2 (since it's 1 - SSIM and SSIM can be negative)
        self.assertGreaterEqual(loss_value.numpy(), 0)
        self.assertLessEqual(loss_value.numpy(), 2)
    
    def test_ssim_identical_images(self):
        """Test SSIM with identical images."""
        loss_fn = SSIMLoss()
        
        # Identical images should have SSIM = 1, so loss = 0
        loss_value = loss_fn(self.y_true, self.y_true)
        self.assertAlmostEqual(loss_value.numpy(), 0, places=5)
    
    def test_ssim_function(self):
        """Test SSIM function (not loss)."""
        ssim_value = ssim(self.y_true, self.y_pred)
        
        # SSIM should be between -1 and 1
        self.assertGreaterEqual(ssim_value.numpy(), -1)
        self.assertLessEqual(ssim_value.numpy(), 1)
        
        # Identical images should have SSIM = 1
        ssim_identical = ssim(self.y_true, self.y_true)
        self.assertAlmostEqual(ssim_identical.numpy(), 1, places=5)


class TestReconLossIntegration(unittest.TestCase):
    """Integration tests for reconstruction loss components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete reconstruction loss pipeline."""
        batch_size, height, width = 1, 32, 32
        
        # Create realistic test data
        # Input image (RGB + additional channels)
        inputs = tf.random.normal([batch_size, height, width, 6])
        
        # Create smooth backward mapping coordinates
        y_coords, x_coords = tf.meshgrid(
            tf.linspace(-0.8, 0.8, height), 
            tf.linspace(-0.8, 0.8, width), 
            indexing='ij'
        )
        
        # Add some distortion
        y_true = tf.stack([x_coords + 0.1 * tf.sin(x_coords * 3), 
                          y_coords + 0.1 * tf.cos(y_coords * 3)], axis=-1)
        y_pred = tf.stack([x_coords + 0.15 * tf.sin(x_coords * 3), 
                          y_coords + 0.12 * tf.cos(y_coords * 3)], axis=-1)
        
        y_true = tf.expand_dims(y_true, axis=0)
        y_pred = tf.expand_dims(y_pred, axis=0)
        
        # Test reconstruction loss
        loss_fn = ReconLoss(use_ssim=True)
        result = loss_fn(inputs, y_true, y_pred)
        
        # Verify all components are computed
        self.assertGreater(result['total_loss'].numpy(), 0)
        self.assertGreater(result['mse_loss'].numpy(), 0)
        self.assertGreater(result['ssim_loss'].numpy(), 0)
        
        # Verify unwarped images are reasonable
        self.assertFalse(tf.reduce_any(tf.math.is_nan(result['uworg'])))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(result['uwpred'])))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)