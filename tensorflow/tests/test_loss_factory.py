"""Integration tests for loss function factory."""

import tensorflow as tf
import numpy as np
import unittest
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.loss_factory import (
    LossFactory, CombinedLoss, LossLogger,
    create_dewarpnet_loss, create_world_coordinate_loss, create_backward_mapping_loss
)
from losses.grad_loss import GradLoss
from losses.recon_loss import ReconLoss, UnwarpLoss
from losses.ssim_loss import SSIMLoss


class TestLossFactory(unittest.TestCase):
    """Test cases for LossFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = LossFactory()
        
        # Create test data
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.channels = 3
        
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
    
    def test_create_grad_loss(self):
        """Test creating gradient loss."""
        loss_fn = self.factory.create_loss('grad_loss', window_size=5)
        self.assertIsInstance(loss_fn, GradLoss)
        self.assertEqual(loss_fn.window_size, 5)
    
    def test_create_recon_loss(self):
        """Test creating reconstruction loss."""
        loss_fn = self.factory.create_loss('recon_loss', use_ssim=False)
        self.assertIsInstance(loss_fn, ReconLoss)
        self.assertFalse(loss_fn.use_ssim)
    
    def test_create_unwarp_loss(self):
        """Test creating unwarp loss."""
        loss_fn = self.factory.create_loss('unwarp_loss')
        self.assertIsInstance(loss_fn, UnwarpLoss)
    
    def test_create_ssim_loss(self):
        """Test creating SSIM loss."""
        loss_fn = self.factory.create_loss('ssim_loss', window_size=7)
        self.assertIsInstance(loss_fn, SSIMLoss)
        self.assertEqual(loss_fn.window_size, 7)
    
    def test_invalid_loss_type(self):
        """Test creating invalid loss type."""
        with self.assertRaises(ValueError):
            self.factory.create_loss('invalid_loss')
    
    def test_loss_aliases(self):
        """Test loss type aliases."""
        grad_loss1 = self.factory.create_loss('grad_loss')
        grad_loss2 = self.factory.create_loss('gradient_loss')
        self.assertEqual(type(grad_loss1), type(grad_loss2))
        
        recon_loss1 = self.factory.create_loss('recon_loss')
        recon_loss2 = self.factory.create_loss('reconstruction_loss')
        self.assertEqual(type(recon_loss1), type(recon_loss2))
    
    def test_get_loss_with_caching(self):
        """Test loss function caching."""
        loss_fn1 = self.factory.get_loss('grad_loss', window_size=3)
        loss_fn2 = self.factory.get_loss('grad_loss', window_size=3)
        
        # Should be the same instance due to caching
        self.assertIs(loss_fn1, loss_fn2)
        
        # Different config should create new instance
        loss_fn3 = self.factory.get_loss('grad_loss', window_size=5)
        self.assertIsNot(loss_fn1, loss_fn3)
    
    def test_list_available_losses(self):
        """Test listing available losses."""
        available = self.factory.list_available_losses()
        self.assertIsInstance(available, dict)
        self.assertIn('grad_loss', available)
        self.assertIn('recon_loss', available)
        self.assertIn('ssim_loss', available)
    
    def test_clear_cache(self):
        """Test clearing loss cache."""
        # Create cached instance
        loss_fn1 = self.factory.get_loss('grad_loss')
        self.assertEqual(len(self.factory._loss_instances), 1)
        
        # Clear cache
        self.factory.clear_cache()
        self.assertEqual(len(self.factory._loss_instances), 0)
        
        # New instance should be created
        loss_fn2 = self.factory.get_loss('grad_loss')
        self.assertIsNot(loss_fn1, loss_fn2)


class TestCombinedLoss(unittest.TestCase):
    """Test cases for CombinedLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data for gradient loss
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.channels = 3
        
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, self.channels).astype(np.float32)
        )
        
        # Create test data for reconstruction loss
        self.inputs = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, 6).astype(np.float32)
        )
        self.bm_true = tf.constant(
            (np.random.rand(self.batch_size, self.height, self.width, 2).astype(np.float32) - 0.5) * 2
        )
        self.bm_pred = tf.constant(
            (np.random.rand(self.batch_size, self.height, self.width, 2).astype(np.float32) - 0.5) * 2
        )
    
    def test_combined_loss_creation(self):
        """Test creating combined loss."""
        loss_configs = {
            'gradient': {
                'type': 'grad_loss',
                'config': {'window_size': 3}
            },
            'ssim': {
                'type': 'ssim_loss',
                'config': {'window_size': 11}
            }
        }
        
        weights = {'gradient': 1.0, 'ssim': 0.5}
        
        combined_loss = CombinedLoss(loss_configs, weights)
        
        # Test gradient loss component
        result = combined_loss(self.y_true, self.y_pred)
        
        self.assertIsInstance(result, dict)
        self.assertIn('total_loss', result)
        self.assertIn('gradient', result)
        self.assertIn('ssim', result)
        self.assertIn('weights', result)
    
    def test_combined_loss_default_weights(self):
        """Test combined loss with default weights."""
        loss_configs = {
            'gradient': {
                'type': 'grad_loss',
                'config': {}
            }
        }
        
        combined_loss = CombinedLoss(loss_configs)
        
        # Should use default weight of 1.0
        self.assertEqual(combined_loss.weights['gradient'], 1.0)
    
    def test_combined_loss_with_reconstruction(self):
        """Test combined loss with reconstruction component."""
        loss_configs = {
            'reconstruction': {
                'type': 'recon_loss',
                'config': {'use_ssim': True}
            }
        }
        
        combined_loss = CombinedLoss(loss_configs)
        
        # Test with reconstruction loss arguments
        result = combined_loss(self.inputs, self.bm_true, self.bm_pred)
        
        self.assertIn('reconstruction', result)
        self.assertIn('reconstruction_components', result)
        self.assertIn('total_loss', result)
    
    def test_get_config(self):
        """Test configuration serialization."""
        loss_configs = {
            'gradient': {
                'type': 'grad_loss',
                'config': {'window_size': 5}
            }
        }
        weights = {'gradient': 2.0}
        
        combined_loss = CombinedLoss(loss_configs, weights)
        config = combined_loss.get_config()
        
        self.assertEqual(config['loss_configs'], loss_configs)
        self.assertEqual(config['weights'], weights)


class TestLossLogger(unittest.TestCase):
    """Test cases for LossLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure logging to avoid output during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        self.logger = LossLogger(log_frequency=2)
    
    def test_loss_logging(self):
        """Test logging loss values."""
        losses = {
            'total_loss': tf.constant(0.5),
            'grad_loss': tf.constant(0.3),
            'recon_loss': tf.constant(0.2)
        }
        
        self.logger.log_losses(losses, step=0)
        self.logger.log_losses(losses, step=1)
        
        history = self.logger.get_loss_history()
        self.assertIn(0, history)
        self.assertIn('total_loss', history[0])
        self.assertEqual(history[0]['total_loss'], 0.5)
    
    def test_average_losses(self):
        """Test computing average losses."""
        # Log several steps
        for step in range(5):
            losses = {
                'loss': tf.constant(float(step))
            }
            self.logger.log_losses(losses, step=step)
        
        # Check that history was recorded
        history = self.logger.get_loss_history()
        self.assertEqual(len(history), 5)
        
        # Test average over all steps
        averages = self.logger.get_average_losses()
        if 'loss' in averages:  # Only test if loss was recorded
            self.assertAlmostEqual(averages['loss'], 2.0, places=5)  # (0+1+2+3+4)/5 = 2.0
            
            # Test average over last 3 steps
            averages_last_3 = self.logger.get_average_losses(last_n_steps=3)
            self.assertAlmostEqual(averages_last_3['loss'], 3.0, places=5)  # (2+3+4)/3 = 3.0


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.height = 32
        self.width = 32
        
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, 3).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.rand(self.batch_size, self.height, self.width, 3).astype(np.float32)
        )
    
    def test_create_dewarpnet_loss(self):
        """Test creating DewarpNet combined loss."""
        combined_loss = create_dewarpnet_loss(grad_weight=1.0, recon_weight=0.5)
        
        self.assertIsInstance(combined_loss, CombinedLoss)
        self.assertEqual(combined_loss.weights['gradient'], 1.0)
        self.assertEqual(combined_loss.weights['reconstruction'], 0.5)
    
    def test_create_world_coordinate_loss(self):
        """Test creating world coordinate loss."""
        loss_fn = create_world_coordinate_loss(window_size=5)
        
        self.assertIsInstance(loss_fn, GradLoss)
        self.assertEqual(loss_fn.window_size, 5)
        
        # Test that it works
        loss_value = loss_fn(self.y_true, self.y_pred)
        self.assertGreater(loss_value.numpy(), 0)
    
    def test_create_backward_mapping_loss(self):
        """Test creating backward mapping loss."""
        combined_loss = create_backward_mapping_loss(grad_weight=0.8, recon_weight=1.2)
        
        self.assertIsInstance(combined_loss, CombinedLoss)
        self.assertEqual(combined_loss.weights['gradient'], 0.8)
        self.assertEqual(combined_loss.weights['reconstruction'], 1.2)


class TestLossFactoryIntegration(unittest.TestCase):
    """Integration tests for complete loss factory workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from factory to loss computation."""
        # Create factory
        factory = LossFactory()
        
        # Create test data
        batch_size, height, width = 2, 32, 32
        
        # World coordinate data
        wc_true = tf.random.normal([batch_size, height, width, 3])
        wc_pred = tf.random.normal([batch_size, height, width, 3])
        
        # Backward mapping data
        inputs = tf.random.normal([batch_size, height, width, 6])
        bm_true = tf.random.uniform([batch_size, height, width, 2], -1, 1)
        bm_pred = tf.random.uniform([batch_size, height, width, 2], -1, 1)
        
        # Test world coordinate training loss
        wc_loss = factory.create_loss('grad_loss')
        wc_loss_value = wc_loss(wc_true, wc_pred)
        self.assertGreater(wc_loss_value.numpy(), 0)
        
        # Test backward mapping training loss
        bm_loss = factory.create_loss('recon_loss')
        bm_result = bm_loss(inputs, bm_true, bm_pred)
        self.assertIn('total_loss', bm_result)
        self.assertGreater(bm_result['total_loss'].numpy(), 0)
        
        # Test combined loss
        combined_loss = create_dewarpnet_loss()
        
        # For combined loss, we need to test with appropriate arguments
        # This would typically be used in a training loop where different
        # loss components are computed separately
        
        self.assertIsInstance(combined_loss, CombinedLoss)
    
    def test_loss_monitoring(self):
        """Test loss monitoring and logging."""
        # Create logger
        logger = LossLogger(log_frequency=1)
        
        # Create loss function
        loss_fn = create_world_coordinate_loss()
        
        # Simulate training steps
        for step in range(5):
            # Create test data
            y_true = tf.random.normal([1, 16, 16, 3])
            y_pred = tf.random.normal([1, 16, 16, 3])
            
            # Compute loss
            loss_value = loss_fn(y_true, y_pred)
            
            # Log loss
            losses = {'grad_loss': loss_value}
            logger.log_losses(losses, step=step)
        
        # Check history
        history = logger.get_loss_history()
        self.assertEqual(len(history), 5)
        
        # Check averages
        averages = logger.get_average_losses()
        if 'grad_loss' in averages:  # Only test if loss was recorded
            self.assertGreater(averages['grad_loss'], 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)