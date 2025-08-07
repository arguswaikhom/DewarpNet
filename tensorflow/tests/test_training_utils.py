"""Unit tests for training utilities."""

import unittest
import tempfile
import shutil
import os
import json
import tensorflow as tf
import numpy as np
import sys
from unittest.mock import patch, MagicMock, Mock
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_utils import (
    TrainingLogger, CheckpointManager, TensorBoardLogger, EarlyStopping,
    LearningRateScheduler, ModelCheckpointer, TrainingStateManager
)


class TestTrainingLogger(unittest.TestCase):
    """Test cases for TrainingLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TrainingLogger(self.temp_dir, 'test_experiment')
        
        # Suppress logging output during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        self.assertEqual(self.logger.log_dir, self.temp_dir)
        self.assertEqual(self.logger.experiment_name, 'test_experiment')
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_log_epoch(self):
        """Test epoch logging."""
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        self.logger.log_epoch(1, 'train', metrics, learning_rate=0.001)
        
        # Check that log entry was created
        self.assertEqual(len(self.logger.training_log), 1)
        
        entry = self.logger.training_log[0]
        self.assertEqual(entry['epoch'], 1)
        self.assertEqual(entry['phase'], 'train')
        self.assertEqual(entry['metrics'], metrics)
        self.assertEqual(entry['learning_rate'], 0.001)
    
    def test_log_checkpoint(self):
        """Test checkpoint logging."""
        checkpoint_path = '/path/to/checkpoint.ckpt'
        self.logger.log_checkpoint(5, checkpoint_path, is_best=True)
        
        # Check checkpoint log
        self.assertEqual(len(self.logger.checkpoint_log), 1)
        
        entry = self.logger.checkpoint_log[0]
        self.assertEqual(entry['epoch'], 5)
        self.assertEqual(entry['path'], checkpoint_path)
        self.assertTrue(entry['is_best'])
    
    def test_save_logs(self):
        """Test saving logs to file."""
        # Add some log entries
        self.logger.log_epoch(1, 'train', {'loss': 0.5}, 0.001)
        self.logger.log_checkpoint(1, '/path/to/ckpt', False)
        
        # Save logs
        self.logger.save_logs()
        
        # Check files exist
        training_log_path = os.path.join(self.temp_dir, 'training_log.json')
        checkpoint_log_path = os.path.join(self.temp_dir, 'checkpoint_log.json')
        
        self.assertTrue(os.path.exists(training_log_path))
        self.assertTrue(os.path.exists(checkpoint_log_path))
        
        # Check file contents
        with open(training_log_path, 'r') as f:
            training_data = json.load(f)
        
        self.assertEqual(len(training_data), 1)
        self.assertEqual(training_data[0]['epoch'], 1)
    
    def test_get_training_summary(self):
        """Test getting training summary."""
        # Add multiple epochs
        for epoch in range(1, 6):
            metrics = {'loss': 1.0 - epoch * 0.1, 'accuracy': epoch * 0.15}
            self.logger.log_epoch(epoch, 'train', metrics, 0.001)
        
        summary = self.logger.get_training_summary()
        
        self.assertIn('total_epochs', summary)
        self.assertIn('best_loss', summary)
        self.assertIn('best_accuracy', summary)
        self.assertIn('final_metrics', summary)
        
        self.assertEqual(summary['total_epochs'], 5)
        self.assertAlmostEqual(summary['best_loss'], 0.5, places=5)
        self.assertAlmostEqual(summary['best_accuracy'], 0.75, places=5)


class TestCheckpointManager(unittest.TestCase):
    """Test cases for CheckpointManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = CheckpointManager(self.temp_dir, max_to_keep=3)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test checkpoint manager initialization."""
        self.assertEqual(self.manager.checkpoint_dir, self.temp_dir)
        self.assertEqual(self.manager.max_to_keep, 3)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_save_checkpoint(self):
        """Test saving checkpoint."""
        # Create mock checkpoint
        mock_checkpoint = Mock()
        mock_checkpoint.save = Mock(return_value='checkpoint_path')
        
        # Save checkpoint
        saved_path = self.manager.save_checkpoint(
            mock_checkpoint, 'test_checkpoint.ckpt', is_best=False
        )
        
        # Check that save was called
        mock_checkpoint.save.assert_called_once()
        self.assertIsNotNone(saved_path)
        
        # Check checkpoint tracking
        self.assertEqual(len(self.manager.checkpoint_history), 1)
    
    def test_save_best_checkpoint(self):
        """Test saving best checkpoint."""
        mock_checkpoint = Mock()
        mock_checkpoint.save = Mock(return_value='best_checkpoint_path')
        
        # Save best checkpoint
        saved_path = self.manager.save_checkpoint(
            mock_checkpoint, 'best_model.ckpt', is_best=True
        )
        
        # Check best checkpoint tracking
        self.assertEqual(self.manager.best_checkpoint_path, saved_path)
    
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        mock_checkpoint = Mock()
        mock_checkpoint.save = Mock(side_effect=lambda x: x)
        
        # Save more checkpoints than max_to_keep
        for i in range(5):
            self.manager.save_checkpoint(
                mock_checkpoint, f'checkpoint_{i}.ckpt', is_best=False
            )
        
        # Should only keep max_to_keep checkpoints
        self.assertEqual(len(self.manager.checkpoint_history), 3)
    
    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint."""
        # No checkpoints initially
        latest = self.manager.get_latest_checkpoint()
        self.assertIsNone(latest)
        
        # Add checkpoint
        mock_checkpoint = Mock()
        mock_checkpoint.save = Mock(return_value='latest_checkpoint')
        
        self.manager.save_checkpoint(mock_checkpoint, 'latest.ckpt', False)
        
        latest = self.manager.get_latest_checkpoint()
        self.assertEqual(latest, 'latest_checkpoint')
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        mock_checkpoint = Mock()
        mock_checkpoint.save = Mock(side_effect=lambda x: x)
        
        # Add multiple checkpoints
        for i in range(3):
            self.manager.save_checkpoint(
                mock_checkpoint, f'checkpoint_{i}.ckpt', is_best=(i == 1)
            )
        
        checkpoints = self.manager.list_checkpoints()
        
        self.assertEqual(len(checkpoints), 3)
        self.assertTrue(any(ckpt['is_best'] for ckpt in checkpoints))


class TestTensorBoardLogger(unittest.TestCase):
    """Test cases for TensorBoardLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TensorBoardLogger(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_logger_initialization(self):
        """Test TensorBoard logger initialization."""
        self.assertEqual(self.logger.log_dir, self.temp_dir)
        self.assertIsNotNone(self.logger.writer)
    
    def test_log_scalars(self):
        """Test logging scalar values."""
        scalars = {'loss': 0.5, 'accuracy': 0.8, 'learning_rate': 0.001}
        
        # Should not raise exception
        self.logger.log_scalars(scalars, step=1)
        
        # Flush to ensure data is written
        self.logger.flush()
    
    def test_log_images(self):
        """Test logging images."""
        # Create test images
        images = tf.random.uniform((2, 32, 32, 3), 0, 1)
        
        # Should not raise exception
        self.logger.log_images(images, 'test_images', step=1, max_outputs=2)
        
        self.logger.flush()
    
    def test_log_histogram(self):
        """Test logging histograms."""
        data = tf.random.normal((100,))
        
        # Should not raise exception
        self.logger.log_histogram(data, 'test_histogram', step=1)
        
        self.logger.flush()
    
    def test_close_logger(self):
        """Test closing logger."""
        # Should not raise exception
        self.logger.close()


class TestEarlyStopping(unittest.TestCase):
    """Test cases for EarlyStopping class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.early_stopping = EarlyStopping(
            patience=3, min_delta=0.01, mode='min', restore_best_weights=True
        )
    
    def test_early_stopping_initialization(self):
        """Test early stopping initialization."""
        self.assertEqual(self.early_stopping.patience, 3)
        self.assertEqual(self.early_stopping.min_delta, 0.01)
        self.assertEqual(self.early_stopping.mode, 'min')
        self.assertTrue(self.early_stopping.restore_best_weights)
        self.assertEqual(self.early_stopping.wait, 0)
        self.assertIsNone(self.early_stopping.best)
    
    def test_improving_metric(self):
        """Test with improving metric."""
        mock_model = Mock()
        
        # Improving metrics (decreasing for 'min' mode)
        metrics = [1.0, 0.8, 0.6, 0.4]
        
        for metric in metrics:
            should_stop = self.early_stopping(metric, mock_model)
            self.assertFalse(should_stop)
        
        self.assertEqual(self.early_stopping.wait, 0)
        self.assertEqual(self.early_stopping.best, 0.4)
    
    def test_non_improving_metric(self):
        """Test with non-improving metric."""
        mock_model = Mock()
        
        # Start with good metric
        self.early_stopping(0.5, mock_model)
        
        # Non-improving metrics
        for _ in range(3):
            should_stop = self.early_stopping(0.6, mock_model)
            if should_stop:
                break
        
        # Should trigger early stopping after patience is exceeded
        self.assertTrue(should_stop)
    
    def test_small_improvement(self):
        """Test with improvement smaller than min_delta."""
        mock_model = Mock()
        
        # Start with baseline
        self.early_stopping(1.0, mock_model)
        
        # Small improvement (less than min_delta=0.01)
        should_stop = self.early_stopping(0.995, mock_model)
        self.assertFalse(should_stop)
        
        # Should increase wait counter
        self.assertEqual(self.early_stopping.wait, 1)
    
    def test_max_mode(self):
        """Test early stopping in 'max' mode."""
        early_stopping = EarlyStopping(patience=2, mode='max')
        mock_model = Mock()
        
        # Improving metrics (increasing for 'max' mode)
        metrics = [0.5, 0.7, 0.9]
        
        for metric in metrics:
            should_stop = early_stopping(metric, mock_model)
            self.assertFalse(should_stop)
        
        # Non-improving metric
        should_stop = early_stopping(0.8, mock_model)
        self.assertFalse(should_stop)  # First non-improvement
        
        should_stop = early_stopping(0.8, mock_model)
        self.assertFalse(should_stop)  # Second non-improvement
        
        should_stop = early_stopping(0.8, mock_model)
        self.assertTrue(should_stop)   # Should stop after patience


class TestLearningRateScheduler(unittest.TestCase):
    """Test cases for LearningRateScheduler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_lr = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.initial_lr)
    
    def test_step_decay_scheduler(self):
        """Test step decay scheduler."""
        scheduler = LearningRateScheduler(
            scheduler_type='step_decay',
            initial_lr=self.initial_lr,
            decay_factor=0.5,
            decay_steps=10
        )
        
        # Test learning rate at different steps
        lr_0 = scheduler.get_lr(0)
        lr_10 = scheduler.get_lr(10)
        lr_20 = scheduler.get_lr(20)
        
        self.assertAlmostEqual(lr_0, self.initial_lr, places=6)
        self.assertAlmostEqual(lr_10, self.initial_lr * 0.5, places=6)
        self.assertAlmostEqual(lr_20, self.initial_lr * 0.25, places=6)
    
    def test_exponential_decay_scheduler(self):
        """Test exponential decay scheduler."""
        scheduler = LearningRateScheduler(
            scheduler_type='exponential_decay',
            initial_lr=self.initial_lr,
            decay_rate=0.9,
            decay_steps=1
        )
        
        # Test exponential decay
        lr_0 = scheduler.get_lr(0)
        lr_1 = scheduler.get_lr(1)
        lr_2 = scheduler.get_lr(2)
        
        self.assertAlmostEqual(lr_0, self.initial_lr, places=6)
        self.assertAlmostEqual(lr_1, self.initial_lr * 0.9, places=6)
        self.assertAlmostEqual(lr_2, self.initial_lr * 0.81, places=6)
    
    def test_cosine_decay_scheduler(self):
        """Test cosine decay scheduler."""
        scheduler = LearningRateScheduler(
            scheduler_type='cosine_decay',
            initial_lr=self.initial_lr,
            decay_steps=100,
            alpha=0.1
        )
        
        # Test cosine decay
        lr_0 = scheduler.get_lr(0)
        lr_50 = scheduler.get_lr(50)  # Halfway point
        lr_100 = scheduler.get_lr(100)  # End point
        
        self.assertAlmostEqual(lr_0, self.initial_lr, places=6)
        self.assertLess(lr_50, self.initial_lr)
        self.assertGreater(lr_50, lr_100)
        self.assertAlmostEqual(lr_100, self.initial_lr * 0.1, places=6)
    
    def test_update_optimizer(self):
        """Test updating optimizer learning rate."""
        scheduler = LearningRateScheduler(
            scheduler_type='step_decay',
            initial_lr=self.initial_lr,
            decay_factor=0.5,
            decay_steps=10
        )
        
        # Update optimizer at step 10
        scheduler.update_optimizer(self.optimizer, step=10)
        
        # Check that optimizer learning rate was updated
        current_lr = float(self.optimizer.learning_rate.numpy())
        expected_lr = self.initial_lr * 0.5
        self.assertAlmostEqual(current_lr, expected_lr, places=6)


class TestModelCheckpointer(unittest.TestCase):
    """Test cases for ModelCheckpointer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpointer = ModelCheckpointer(
            self.temp_dir,
            monitor='val_loss',
            mode='min',
            save_best_only=False,
            save_weights_only=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpointer_initialization(self):
        """Test checkpointer initialization."""
        self.assertEqual(self.checkpointer.checkpoint_dir, self.temp_dir)
        self.assertEqual(self.checkpointer.monitor, 'val_loss')
        self.assertEqual(self.checkpointer.mode, 'min')
        self.assertFalse(self.checkpointer.save_best_only)
        self.assertFalse(self.checkpointer.save_weights_only)
    
    def test_should_save_checkpoint(self):
        """Test checkpoint saving decision."""
        # First epoch should always save
        should_save, is_best = self.checkpointer.should_save_checkpoint(
            epoch=1, metrics={'val_loss': 0.5}
        )
        self.assertTrue(should_save)
        self.assertTrue(is_best)
        
        # Better metric should save
        should_save, is_best = self.checkpointer.should_save_checkpoint(
            epoch=2, metrics={'val_loss': 0.3}
        )
        self.assertTrue(should_save)
        self.assertTrue(is_best)
        
        # Worse metric should still save (save_best_only=False)
        should_save, is_best = self.checkpointer.should_save_checkpoint(
            epoch=3, metrics={'val_loss': 0.7}
        )
        self.assertTrue(should_save)
        self.assertFalse(is_best)
    
    def test_save_best_only(self):
        """Test save_best_only functionality."""
        checkpointer = ModelCheckpointer(
            self.temp_dir,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
        
        # First epoch should save
        should_save, is_best = checkpointer.should_save_checkpoint(
            epoch=1, metrics={'val_loss': 0.5}
        )
        self.assertTrue(should_save)
        self.assertTrue(is_best)
        
        # Worse metric should not save
        should_save, is_best = checkpointer.should_save_checkpoint(
            epoch=2, metrics={'val_loss': 0.7}
        )
        self.assertFalse(should_save)
        self.assertFalse(is_best)
    
    def test_max_mode(self):
        """Test checkpointer in 'max' mode."""
        checkpointer = ModelCheckpointer(
            self.temp_dir,
            monitor='val_accuracy',
            mode='max'
        )
        
        # First epoch
        should_save, is_best = checkpointer.should_save_checkpoint(
            epoch=1, metrics={'val_accuracy': 0.7}
        )
        self.assertTrue(should_save)
        self.assertTrue(is_best)
        
        # Higher accuracy should be better
        should_save, is_best = checkpointer.should_save_checkpoint(
            epoch=2, metrics={'val_accuracy': 0.8}
        )
        self.assertTrue(should_save)
        self.assertTrue(is_best)
        
        # Lower accuracy should not be best
        should_save, is_best = checkpointer.should_save_checkpoint(
            epoch=3, metrics={'val_accuracy': 0.6}
        )
        self.assertTrue(should_save)  # save_best_only=False by default
        self.assertFalse(is_best)


class TestTrainingStateManager(unittest.TestCase):
    """Test cases for TrainingStateManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = TrainingStateManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_state_manager_initialization(self):
        """Test state manager initialization."""
        self.assertEqual(self.state_manager.state_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_save_and_load_state(self):
        """Test saving and loading training state."""
        # Create mock training state
        state = {
            'epoch': 10,
            'global_step': 1000,
            'best_metrics': {'val_loss': 0.3},
            'optimizer_state': {'learning_rate': 0.001},
            'random_state': 42
        }
        
        # Save state
        state_path = self.state_manager.save_state(state, 'training_state.json')
        self.assertTrue(os.path.exists(state_path))
        
        # Load state
        loaded_state = self.state_manager.load_state('training_state.json')
        
        # Verify loaded state
        self.assertEqual(loaded_state['epoch'], 10)
        self.assertEqual(loaded_state['global_step'], 1000)
        self.assertEqual(loaded_state['best_metrics']['val_loss'], 0.3)
    
    def test_get_latest_state(self):
        """Test getting latest state file."""
        # No state files initially
        latest = self.state_manager.get_latest_state()
        self.assertIsNone(latest)
        
        # Save multiple states
        for i in range(3):
            state = {'epoch': i + 1}
            self.state_manager.save_state(state, f'state_{i}.json')
        
        # Should return the most recent state file
        latest = self.state_manager.get_latest_state()
        self.assertIsNotNone(latest)
        self.assertIn('state_', latest)
    
    def test_cleanup_old_states(self):
        """Test cleanup of old state files."""
        # Save multiple states
        for i in range(10):
            state = {'epoch': i + 1}
            self.state_manager.save_state(state, f'state_{i:03d}.json')
        
        # Cleanup keeping only 5 most recent
        self.state_manager.cleanup_old_states(keep_last=5)
        
        # Check that only 5 files remain
        state_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
        self.assertEqual(len(state_files), 5)
        
        # Check that the most recent files are kept
        state_files.sort()
        self.assertIn('state_009.json', state_files)
        self.assertIn('state_005.json', state_files)
        self.assertNotIn('state_000.json', state_files)


class TestTrainingUtilsIntegration(unittest.TestCase):
    """Integration tests for training utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_training_utilities_workflow(self):
        """Test complete workflow with all training utilities."""
        # Initialize components
        logger = TrainingLogger(self.temp_dir, 'integration_test')
        checkpoint_manager = CheckpointManager(
            os.path.join(self.temp_dir, 'checkpoints'), max_to_keep=3
        )
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Create mock model and optimizer
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_checkpoint = Mock()
        mock_checkpoint.save = Mock(side_effect=lambda x: x)
        
        # Simulate training epochs
        metrics_history = []
        for epoch in range(1, 8):
            # Simulate improving then worsening metrics
            if epoch <= 4:
                val_loss = 1.0 - epoch * 0.2  # Improving
            else:
                val_loss = 0.2 + (epoch - 4) * 0.1  # Worsening
            
            metrics = {'train_loss': val_loss + 0.1, 'val_loss': val_loss}
            metrics_history.append(metrics)
            
            # Log epoch
            logger.log_epoch(epoch, 'train', metrics, learning_rate=0.001)
            
            # Check if should save checkpoint
            should_save = (epoch % 2 == 0) or (val_loss < 0.5)
            if should_save:
                checkpoint_path = checkpoint_manager.save_checkpoint(
                    mock_checkpoint, f'model_epoch_{epoch}.ckpt', 
                    is_best=(val_loss == min(m['val_loss'] for m in metrics_history))
                )
                logger.log_checkpoint(epoch, checkpoint_path, 
                                    is_best=(val_loss < 0.5))
            
            # Check early stopping
            if early_stopping(val_loss, mock_model):
                logger.log_epoch(epoch, 'early_stop', 
                               {'reason': 'Early stopping triggered'}, 0.001)
                break
        
        # Verify results
        self.assertGreater(len(logger.training_log), 0)
        self.assertGreater(len(logger.checkpoint_log), 0)
        self.assertGreater(len(checkpoint_manager.checkpoint_history), 0)
        
        # Get training summary
        summary = logger.get_training_summary()
        self.assertIn('total_epochs', summary)
        self.assertIn('best_loss', summary)
        
        # Save logs
        logger.save_logs()
        
        # Verify log files exist
        self.assertTrue(os.path.exists(
            os.path.join(self.temp_dir, 'training_log.json')
        ))
        self.assertTrue(os.path.exists(
            os.path.join(self.temp_dir, 'checkpoint_log.json')
        ))


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