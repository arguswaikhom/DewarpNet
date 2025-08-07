"""Unit tests for TrainingManager and related training utilities."""

import unittest
import tempfile
import shutil
import os
import json
import tensorflow as tf
import numpy as np
import sys
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.manager import TrainingManager
from training.config import (
    TrainingPipelineConfig, ModelConfig, DataConfig, OptimizerConfig,
    LossConfig, TrainingConfig, LoggingConfig, ConfigManager
)
from training.metrics import (
    MetricsComputer, TrainingMonitor, PerformanceProfiler, 
    MetricsAggregator, ValidationMetrics
)


class TestTrainingManager(unittest.TestCase):
    """Test cases for TrainingManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config for testing
        self.config = TrainingPipelineConfig()
        self.config.task_type = 'world_coordinate'
        self.config.data.data_path = self.temp_dir
        self.config.data.img_rows = 64
        self.config.data.img_cols = 64
        self.config.data.batch_size = 2
        self.config.training.n_epoch = 2
        self.config.logging.logdir = os.path.join(self.temp_dir, 'logs')
        self.config.logging.tensorboard_log_dir = os.path.join(self.temp_dir, 'tb')
        
        # Create mock data files
        self._create_mock_data_files()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_data_files(self):
        """Create mock data files for testing."""
        # Create directory structure
        os.makedirs(os.path.join(self.temp_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'wc'), exist_ok=True)
        
        # Create train.txt and val.txt
        with open(os.path.join(self.temp_dir, 'train.txt'), 'w') as f:
            f.write('sample1\nsample2\n')
        
        with open(os.path.join(self.temp_dir, 'val.txt'), 'w') as f:
            f.write('val_sample1\n')
    
    @patch('tensorflow.training.manager.setup_gpu')
    @patch('tensorflow.training.manager.Doc3DWCLoader')
    @patch('tensorflow.training.manager.create_wc_dataset')
    def test_training_manager_initialization(self, mock_create_dataset, 
                                           mock_loader, mock_setup_gpu):
        """Test TrainingManager initialization."""
        # Mock GPU setup
        mock_setup_gpu.return_value = {'gpu_available': False}
        
        # Mock data loader
        mock_loader_instance = Mock()
        mock_loader_instance.__len__ = Mock(return_value=2)
        mock_loader.return_value = mock_loader_instance
        
        # Mock dataset creation
        mock_dataset = Mock()
        mock_create_dataset.return_value = mock_dataset
        
        # Initialize training manager
        manager = TrainingManager(self.config)
        
        # Check initialization
        self.assertIsNotNone(manager.model)
        self.assertIsNotNone(manager.optimizer)
        self.assertIsNotNone(manager.metrics_computer)
        self.assertEqual(manager.current_epoch, 0)
        self.assertEqual(manager.global_step, 0)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Test valid config
        issues = config_manager.validate_config(self.config)
        # May have issues due to missing data files, but should not crash
        self.assertIsInstance(issues, list)
        
        # Test invalid config
        invalid_config = TrainingPipelineConfig()
        invalid_config.data.data_path = ''
        invalid_config.data.batch_size = -1
        
        issues = config_manager.validate_config(invalid_config)
        self.assertGreater(len(issues), 0)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config_manager = ConfigManager()
        
        # Save config
        config_path = os.path.join(self.temp_dir, 'config.json')
        config_manager.save_config(self.config, config_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load config
        loaded_config = config_manager.load_config(config_path)
        
        # Check loaded config
        self.assertEqual(loaded_config.task_type, self.config.task_type)
        self.assertEqual(loaded_config.data.batch_size, self.config.data.batch_size)


class TestMetricsComputer(unittest.TestCase):
    """Test cases for MetricsComputer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics_computer = MetricsComputer()
        
        # Create test tensors
        self.predictions = tf.random.normal((2, 32, 32, 3))
        self.targets = tf.random.normal((2, 32, 32, 3))
    
    def test_compute_mse(self):
        """Test MSE computation."""
        mse = self.metrics_computer.compute_mse(self.predictions, self.targets)
        
        self.assertIsInstance(mse, tf.Tensor)
        self.assertEqual(mse.shape, ())
        self.assertGreater(mse.numpy(), 0)
    
    def test_compute_mae(self):
        """Test MAE computation."""
        mae = self.metrics_computer.compute_mae(self.predictions, self.targets)
        
        self.assertIsInstance(mae, tf.Tensor)
        self.assertEqual(mae.shape, ())
        self.assertGreater(mae.numpy(), 0)
    
    def test_compute_psnr(self):
        """Test PSNR computation."""
        psnr = self.metrics_computer.compute_psnr(self.predictions, self.targets)
        
        self.assertIsInstance(psnr, tf.Tensor)
        self.assertEqual(psnr.shape, ())
        # PSNR can be negative for very different images
        self.assertTrue(tf.math.is_finite(psnr))
    
    def test_compute_ssim(self):
        """Test SSIM computation."""
        ssim = self.metrics_computer.compute_ssim(self.predictions, self.targets)
        
        self.assertIsInstance(ssim, tf.Tensor)
        self.assertEqual(ssim.shape, (2,))  # Batch dimension
        self.assertTrue(tf.reduce_all(ssim >= -1.0))
        self.assertTrue(tf.reduce_all(ssim <= 1.0))
    
    def test_compute_gradient_error(self):
        """Test gradient error computation."""
        grad_error = self.metrics_computer.compute_gradient_error(
            self.predictions, self.targets
        )
        
        self.assertIsInstance(grad_error, tf.Tensor)
        self.assertEqual(grad_error.shape, ())
        self.assertGreater(grad_error.numpy(), 0)
    
    def test_compute_all_metrics(self):
        """Test computing all metrics."""
        metrics = self.metrics_computer.compute_all_metrics(
            self.predictions, self.targets, prefix='test_'
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('test_mse', metrics)
        self.assertIn('test_mae', metrics)
        self.assertIn('test_psnr', metrics)
        self.assertIn('test_ssim', metrics)
        self.assertIn('test_gradient_error', metrics)
        
        # Check all metrics are finite
        for name, value in metrics.items():
            self.assertTrue(tf.math.is_finite(value).numpy().all())
    
    def test_masked_metrics(self):
        """Test metrics with mask."""
        # Create mask
        mask = tf.ones_like(self.predictions[:, :, :, :1])
        mask = tf.concat([mask, mask, mask], axis=-1)
        
        # Set some regions to zero
        mask = mask.numpy()
        mask[:, 10:20, 10:20, :] = 0
        mask = tf.constant(mask)
        
        # Compute masked metrics
        mse = self.metrics_computer.compute_mse(
            self.predictions, self.targets, mask
        )
        mae = self.metrics_computer.compute_mae(
            self.predictions, self.targets, mask
        )
        
        self.assertIsInstance(mse, tf.Tensor)
        self.assertIsInstance(mae, tf.Tensor)
        self.assertGreater(mse.numpy(), 0)
        self.assertGreater(mae.numpy(), 0)


class TestTrainingMonitor(unittest.TestCase):
    """Test cases for TrainingMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = TrainingMonitor(window_size=10)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(len(self.monitor.loss_history), 0)
        self.assertFalse(self.monitor.loss_exploded)
        self.assertFalse(self.monitor.loss_plateaued)
        self.assertFalse(self.monitor.gradients_vanished)
    
    def test_update_losses(self):
        """Test updating with loss values."""
        losses = {'total_loss': 0.5, 'mse_loss': 0.3}
        metrics = {'accuracy': 0.8}
        
        self.monitor.update(losses, metrics)
        
        self.assertEqual(len(self.monitor.loss_history), 1)
        self.assertEqual(self.monitor.loss_history[0], 0.5)
        self.assertEqual(len(self.monitor.metric_history['accuracy']), 1)
    
    def test_loss_explosion_detection(self):
        """Test loss explosion detection."""
        # Normal losses
        for i in range(5):
            self.monitor.update({'total_loss': 0.1 + i * 0.01})
        
        self.assertFalse(self.monitor.loss_exploded)
        
        # Exploded loss
        self.monitor.update({'total_loss': 1e7})
        self.assertTrue(self.monitor.loss_exploded)
    
    def test_nan_loss_detection(self):
        """Test NaN loss detection."""
        self.monitor.update({'total_loss': float('nan')})
        self.assertTrue(self.monitor.loss_exploded)
    
    def test_gradient_vanishing_detection(self):
        """Test gradient vanishing detection."""
        # Create mock gradients
        normal_gradients = [tf.constant(0.01), tf.constant(0.02)]
        vanishing_gradients = [tf.constant(1e-8), tf.constant(1e-9)]
        
        # Normal gradients
        self.monitor.update({'total_loss': 0.5}, gradients=normal_gradients)
        self.assertFalse(self.monitor.gradients_vanished)
        
        # Vanishing gradients
        self.monitor.update({'total_loss': 0.5}, gradients=vanishing_gradients)
        self.assertTrue(self.monitor.gradients_vanished)
    
    def test_get_status(self):
        """Test getting monitor status."""
        # Add some data
        for i in range(15):
            self.monitor.update({'total_loss': 1.0 - i * 0.05})
        
        status = self.monitor.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('loss_exploded', status)
        self.assertIn('loss_plateaued', status)
        self.assertIn('gradients_vanished', status)
        self.assertIn('recent_loss_trend', status)
        self.assertEqual(status['loss_history_length'], 10)  # Window size
        self.assertEqual(status['recent_loss_trend'], 'improving')
    
    def test_reset(self):
        """Test monitor reset."""
        # Add some data
        self.monitor.update({'total_loss': 0.5})
        self.monitor.loss_exploded = True
        
        # Reset
        self.monitor.reset()
        
        self.assertEqual(len(self.monitor.loss_history), 0)
        self.assertFalse(self.monitor.loss_exploded)


class TestPerformanceProfiler(unittest.TestCase):
    """Test cases for PerformanceProfiler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
    
    def test_timing_operations(self):
        """Test timing operations."""
        import time
        
        # Time an operation
        self.profiler.start_timer('test_op')
        time.sleep(0.01)  # Small delay
        self.profiler.end_timer('test_op')
        
        # Check timing was recorded
        stats = self.profiler.get_stats()
        self.assertIn('test_op', stats)
        self.assertGreater(stats['test_op']['mean'], 0)
        self.assertEqual(stats['test_op']['count'], 1)
    
    def test_multiple_timings(self):
        """Test multiple timing operations."""
        import time
        
        # Time multiple operations
        for i in range(3):
            self.profiler.start_timer('repeated_op')
            time.sleep(0.001)
            self.profiler.end_timer('repeated_op')
        
        stats = self.profiler.get_stats()
        self.assertEqual(stats['repeated_op']['count'], 3)
        self.assertGreater(stats['repeated_op']['mean'], 0)
        self.assertGreater(stats['repeated_op']['total'], 0)
    
    def test_reset(self):
        """Test profiler reset."""
        # Add some timing data
        self.profiler.start_timer('test')
        self.profiler.end_timer('test')
        
        # Reset
        self.profiler.reset()
        
        stats = self.profiler.get_stats()
        self.assertEqual(len(stats), 0)


class TestMetricsAggregator(unittest.TestCase):
    """Test cases for MetricsAggregator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = MetricsAggregator()
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        self.assertEqual(len(self.aggregator.metrics), 0)
        self.assertEqual(len(self.aggregator.weights), 0)
    
    def test_update_metrics(self):
        """Test updating metrics."""
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        self.aggregator.update(metrics, weight=2.0)
        
        self.assertEqual(len(self.aggregator.metrics['loss']), 1)
        self.assertEqual(len(self.aggregator.weights['loss']), 1)
        self.assertEqual(self.aggregator.weights['loss'][0], 2.0)
    
    def test_tensor_metrics(self):
        """Test updating with tensor metrics."""
        metrics = {'loss': tf.constant(0.5), 'accuracy': tf.constant(0.8)}
        self.aggregator.update(metrics)
        
        averages = self.aggregator.get_averages()
        self.assertAlmostEqual(averages['loss'], 0.5, places=5)
        self.assertAlmostEqual(averages['accuracy'], 0.8, places=5)
    
    def test_weighted_averages(self):
        """Test weighted average computation."""
        # Add metrics with different weights
        self.aggregator.update({'loss': 1.0}, weight=1.0)
        self.aggregator.update({'loss': 2.0}, weight=2.0)
        self.aggregator.update({'loss': 3.0}, weight=1.0)
        
        averages = self.aggregator.get_averages()
        # Weighted average: (1*1 + 2*2 + 3*1) / (1+2+1) = 8/4 = 2.0
        self.assertAlmostEqual(averages['loss'], 2.0, places=5)
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        # Add multiple values
        for i in range(5):
            self.aggregator.update({'metric': float(i)})
        
        summary = self.aggregator.get_summary()
        
        self.assertIn('metric', summary)
        self.assertIn('mean', summary['metric'])
        self.assertIn('std', summary['metric'])
        self.assertIn('min', summary['metric'])
        self.assertIn('max', summary['metric'])
        self.assertIn('count', summary['metric'])
        
        self.assertAlmostEqual(summary['metric']['mean'], 2.0, places=5)
        self.assertEqual(summary['metric']['count'], 5)
        self.assertEqual(summary['metric']['min'], 0.0)
        self.assertEqual(summary['metric']['max'], 4.0)
    
    def test_reset(self):
        """Test aggregator reset."""
        self.aggregator.update({'loss': 0.5})
        self.aggregator.reset()
        
        self.assertEqual(len(self.aggregator.metrics), 0)
        self.assertEqual(len(self.aggregator.weights), 0)


class TestValidationMetrics(unittest.TestCase):
    """Test cases for ValidationMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validation_metrics = ValidationMetrics(save_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validation_metrics_initialization(self):
        """Test ValidationMetrics initialization."""
        self.assertEqual(self.validation_metrics.save_dir, self.temp_dir)
        self.assertEqual(len(self.validation_metrics.results_history), 0)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_evaluate_epoch(self):
        """Test epoch evaluation."""
        # Create mock model and dataset
        mock_model = Mock()
        mock_model.return_value = tf.random.normal((2, 32, 32, 3))
        
        # Create mock dataset
        def dataset_generator():
            for _ in range(3):
                inputs = tf.random.normal((2, 32, 32, 3))
                targets = tf.random.normal((2, 32, 32, 3))
                yield inputs, targets
        
        mock_dataset = tf.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                tf.TensorSpec(shape=(2, 32, 32, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(2, 32, 32, 3), dtype=tf.float32)
            )
        )
        
        # Create metrics computer
        metrics_computer = MetricsComputer()
        
        # Evaluate epoch
        results = self.validation_metrics.evaluate_epoch(
            mock_model, mock_dataset, metrics_computer
        )
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertIn('val_mse', results)
        self.assertIn('val_mae', results)
        self.assertEqual(len(self.validation_metrics.results_history), 1)
        
        # Check that results file was saved
        result_files = os.listdir(self.temp_dir)
        self.assertEqual(len(result_files), 1)
        self.assertTrue(result_files[0].startswith('validation_epoch_'))
    
    def test_get_best_epoch(self):
        """Test getting best epoch."""
        # Add some mock results
        self.validation_metrics.results_history = [
            {'val_mse': 0.5, 'val_mae': 0.3},
            {'val_mse': 0.3, 'val_mae': 0.2},  # Best MSE
            {'val_mse': 0.4, 'val_mae': 0.1}   # Best MAE
        ]
        
        # Test best MSE (min)
        best_epoch, best_value = self.validation_metrics.get_best_epoch('val_mse', 'min')
        self.assertEqual(best_epoch, 1)
        self.assertEqual(best_value, 0.3)
        
        # Test best MAE (min)
        best_epoch, best_value = self.validation_metrics.get_best_epoch('val_mae', 'min')
        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_value, 0.1)
    
    def test_empty_results(self):
        """Test behavior with empty results."""
        best_epoch, best_value = self.validation_metrics.get_best_epoch()
        self.assertEqual(best_epoch, -1)
        self.assertEqual(best_value, float('inf'))


class TestTrainingUtilsIntegration(unittest.TestCase):
    """Integration tests for training utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_metrics_flow(self):
        """Test complete metrics computation flow."""
        # Create components
        metrics_computer = MetricsComputer()
        aggregator = MetricsAggregator()
        monitor = TrainingMonitor()
        
        # Simulate training steps
        for step in range(5):
            # Create mock predictions and targets
            predictions = tf.random.normal((2, 16, 16, 3))
            targets = tf.random.normal((2, 16, 16, 3))
            
            # Compute metrics
            step_metrics = metrics_computer.compute_all_metrics(
                predictions, targets, prefix='train_'
            )
            
            # Update aggregator
            aggregator.update(step_metrics, weight=2.0)
            
            # Update monitor
            losses = {'total_loss': float(step_metrics['train_mse'])}
            monitor.update(losses)
        
        # Get results
        avg_metrics = aggregator.get_averages()
        monitor_status = monitor.get_status()
        
        # Check results
        self.assertIn('train_mse', avg_metrics)
        self.assertIn('train_mae', avg_metrics)
        self.assertGreater(avg_metrics['train_mse'], 0)
        
        self.assertIn('loss_history_length', monitor_status)
        self.assertEqual(monitor_status['loss_history_length'], 5)
    
    def test_config_workflow(self):
        """Test complete configuration workflow."""
        # Create config
        config = ConfigManager.create_world_coordinate_config(
            data_path=self.temp_dir,
            batch_size=4,
            n_epoch=10
        )
        
        # Validate config
        config_manager = ConfigManager()
        issues = config_manager.validate_config(config)
        
        # Save config
        config_path = os.path.join(self.temp_dir, 'test_config.json')
        config_manager.save_config(config, config_path)
        
        # Load config
        loaded_config = config_manager.load_config(config_path)
        
        # Verify loaded config
        self.assertEqual(loaded_config.data.batch_size, 4)
        self.assertEqual(loaded_config.training.n_epoch, 10)
        self.assertEqual(loaded_config.task_type, 'world_coordinate')


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