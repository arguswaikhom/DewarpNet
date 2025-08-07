"""Unit tests for gradient loss function."""

import tensorflow as tf
import numpy as np
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.grad_loss import GradLoss, sobel, create_window, gradient, grad_loss


class TestGradLoss(unittest.TestCase):
    """Test cases for gradient loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 64
        self.width = 64
        self.channels = 3
        
        # Create test tensors
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
    
    def test_sobel_kernel_creation(self):
        """Test Sobel kernel creation."""
        # Test 3x3 kernel
        matx, maty = sobel(3)
        self.assertEqual(matx.shape, (3, 3))
        self.assertEqual(maty.shape, (3, 3))
        
        # Test 5x5 kernel
        matx, maty = sobel(5)
        self.assertEqual(matx.shape, (5, 5))
        self.assertEqual(maty.shape, (5, 5))
        
        # Test that kernels are different
        self.assertFalse(tf.reduce_all(tf.equal(matx, maty)))
    
    def test_create_window(self):
        """Test convolution window creation."""
        window_size = 3
        channels = 3
        
        windowx, windowy = create_window(window_size, channels)
        
        # Check shapes: [height, width, in_channels, out_channels]
        expected_shape = [window_size, window_size, channels, 1]
        self.assertEqual(windowx.shape.as_list(), expected_shape)
        self.assertEqual(windowy.shape.as_list(), expected_shape)
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        window_size = 3
        channels = 3
        
        # Create test image
        test_img = tf.random.normal([1, 32, 32, channels])
        
        # Create windows
        windowx, windowy = create_window(window_size, channels)
        
        # Compute gradients
        gradx, grady = gradient(test_img, windowx, windowy, window_size, 'SAME', channels)
        
        # Check output shapes
        self.assertEqual(gradx.shape, test_img.shape)
        self.assertEqual(grady.shape, test_img.shape)
    
    def test_grad_loss_class(self):
        """Test GradLoss class."""
        loss_fn = GradLoss(window_size=3)
        
        # Compute loss
        loss_value = loss_fn(self.y_true, self.y_pred)
        
        # Check that loss is a scalar
        self.assertEqual(loss_value.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss_value.numpy(), 0)
        
        # Test that identical inputs give zero loss
        zero_loss = loss_fn(self.y_true, self.y_true)
        self.assertAlmostEqual(zero_loss.numpy(), 0, places=5)
    
    def test_grad_loss_functional(self):
        """Test functional interface."""
        loss_value = grad_loss(self.y_true, self.y_pred, window_size=3)
        
        # Check that loss is a scalar
        self.assertEqual(loss_value.shape, ())
        
        # Check that loss is positive
        self.assertGreater(loss_value.numpy(), 0)
    
    def test_different_window_sizes(self):
        """Test gradient loss with different window sizes."""
        loss_fn_3 = GradLoss(window_size=3)
        loss_fn_5 = GradLoss(window_size=5)
        
        loss_3 = loss_fn_3(self.y_true, self.y_pred)
        loss_5 = loss_fn_5(self.y_true, self.y_pred)
        
        # Both should be positive scalars
        self.assertEqual(loss_3.shape, ())
        self.assertEqual(loss_5.shape, ())
        self.assertGreater(loss_3.numpy(), 0)
        self.assertGreater(loss_5.numpy(), 0)
    
    def test_single_channel_input(self):
        """Test gradient loss with single channel input."""
        # Create single channel test data
        y_true_single = tf.random.normal([2, 32, 32, 1])
        y_pred_single = tf.random.normal([2, 32, 32, 1])
        
        loss_fn = GradLoss(window_size=3)
        loss_value = loss_fn(y_true_single, y_pred_single)
        
        # Check that loss is computed correctly
        self.assertEqual(loss_value.shape, ())
        self.assertGreater(loss_value.numpy(), 0)
    
    def test_gradient_consistency(self):
        """Test that gradients are computed consistently."""
        # Create a simple test pattern
        test_pattern = np.zeros((1, 10, 10, 1), dtype=np.float32)
        test_pattern[0, 2:8, 2:8, 0] = 1.0  # Square in the middle
        test_tensor = tf.constant(test_pattern)
        
        loss_fn = GradLoss(window_size=3)
        
        # Compute loss with itself (should be zero)
        loss_value = loss_fn(test_tensor, test_tensor)
        self.assertAlmostEqual(loss_value.numpy(), 0, places=5)
        
        # Create shifted version
        shifted_pattern = np.zeros((1, 10, 10, 1), dtype=np.float32)
        shifted_pattern[0, 3:9, 3:9, 0] = 1.0  # Shifted square
        shifted_tensor = tf.constant(shifted_pattern)
        
        # Loss should be positive
        loss_shifted = loss_fn(test_tensor, shifted_tensor)
        self.assertGreater(loss_shifted.numpy(), 0)
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        loss_fn = GradLoss(window_size=5, padding='VALID')
        config = loss_fn.get_config()
        
        self.assertEqual(config['window_size'], 5)
        self.assertEqual(config['padding'], 'VALID')


class TestGradLossComparison(unittest.TestCase):
    """Test gradient loss against reference implementation patterns."""
    
    def test_sobel_kernel_values(self):
        """Test that Sobel kernel values match expected patterns."""
        # Test 3x3 Sobel kernel
        matx, maty = sobel(3)
        
        # Check that center values are zero (as per original implementation)
        center_idx = 1  # For 3x3 kernel
        self.assertEqual(matx[center_idx, center_idx].numpy(), 0)
        self.assertEqual(maty[center_idx, center_idx].numpy(), 0)
        
        # Check symmetry properties
        # X-gradient kernel should be antisymmetric about vertical axis
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    matx[i, j].numpy(), 
                    -matx[i, 2-j].numpy(), 
                    places=5
                )
        
        # Y-gradient kernel should be antisymmetric about horizontal axis
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    maty[i, j].numpy(), 
                    -maty[2-i, j].numpy(), 
                    places=5
                )


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)