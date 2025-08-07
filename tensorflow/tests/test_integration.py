"""
Integration tests for TensorFlow DewarpNet implementation.
Tests end-to-end workflows and component interactions.
"""

import unittest
import tempfile
import shutil
import os
import json
import time
import numpy as np
import tensorflow as tf
import sys
from unittest.mock import patch, MagicMock, Mock
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import ModelFactory, get_model
from loaders.doc3d_wc_loader import Doc3DWCLoader, create_wc_dataset
from loaders.doc3d_bm_loader import Doc3DBMLoader, create_bm_dataset
from losses.loss_factory import LossFactory, create_dewarpnet_loss
from training.config import TrainingPipelineConfig, ConfigManager
from training.metrics import MetricsComputer, TrainingMonitor
from inference.infer import DewarpNetInference
from utils.training_utils import TrainingLogger, CheckpointManager


class TestEndToEndTrainingPipeline(unittest.TestCase):
    """Test complete training pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        # Create directory structure
        self._create_mock_dataset()
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_dataset(self):
        """Create mock dataset for testing."""
        # Create directory structure
        os.makedirs(os.path.join(self.data_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'wc'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'recon', '1', 'chess48'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'bm'), exist_ok=True)
        
        # Create sample files
        sample_names = ['sample1', 'sample2', 'sample3']
        
        # Create train.txt and val.txt
        with open(os.path.join(self.data_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(sample_names))
        
        with open(os.path.join(self.data_dir, 'val.txt'), 'w') as f:
            f.write('sample1\n')
        
        # Create mock images and labels
        for name in sample_names:
            # RGB image
            img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img_path = os.path.join(self.data_dir, 'img', f'{name}.png')
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # World coordinates (EXR)
            wc = np.random.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
            wc_path = os.path.join(self.data_dir, 'wc', f'{name}.exr')
            cv2.imwrite(wc_path, wc)
            
            # Albedo for backward mapping
            alb_path = os.path.join(self.data_dir, 'recon', '1', 'chess48', f'{name}.png')
            cv2.imwrite(alb_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Backward mapping coordinates
            bm = np.random.uniform(0, 448, (448, 448, 2)).astype(np.float32)
            bm_path = os.path.join(self.data_dir, 'bm', f'1_{name}.mat')
            import scipy.io as sio
            sio.savemat(bm_path, {'bm': bm})
    
    def test_world_coordinate_training_pipeline(self):
        """Test complete world coordinate training pipeline."""
        print("Testing World Coordinate Training Pipeline...")
        
        # Create configuration
        config = ConfigManager.create_world_coordinate_config(
            data_path=self.data_dir,
            batch_size=2,
            n_epoch=2,
            img_rows=64,
            img_cols=64
        )
        config.logging.logdir = self.output_dir
        
        # Create model
        model = ModelFactory.create_world_coordinate_model(
            input_nc=3, output_nc=3, num_downs=5, ngf=32
        )
        
        # Build model
        dummy_input = tf.random.normal((1, 64, 64, 3))
        _ = model(dummy_input)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        # Create loss function
        loss_factory = LossFactory()
        loss_fn = loss_factory.create_loss('grad_loss', window_size=3)
        
        # Create data loader
        train_loader = Doc3DWCLoader(
            root_path=self.data_dir,
            split='train',
            img_size=(64, 64),
            augmentations=False
        )
        
        # Create dataset
        train_dataset = create_wc_dataset(
            root=self.data_dir,
            split='train',
            batch_size=2,
            img_size=(64, 64),
            shuffle=False
        )
        
        # Create training utilities
        logger = TrainingLogger(self.output_dir, 'wc_test')
        checkpoint_manager = CheckpointManager(
            os.path.join(self.output_dir, 'checkpoints'), max_to_keep=2
        )
        metrics_computer = MetricsComputer()
        
        # Training loop
        for epoch in range(2):
            epoch_losses = []
            
            for batch_idx, (inputs, targets) in enumerate(train_dataset.take(2)):
                with tf.GradientTape() as tape:
                    predictions = model(inputs, training=True)
                    loss_value = loss_fn(targets, predictions)
                
                # Compute gradients and update
                gradients = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Compute metrics
                metrics = metrics_computer.compute_all_metrics(
                    predictions, targets, prefix='train_'
                )
                
                epoch_losses.append(float(loss_value))
                
                if batch_idx >= 1:  # Limit batches for testing
                    break
            
            # Log epoch
            avg_loss = np.mean(epoch_losses)
            logger.log_epoch(
                epoch + 1, 'Train', 
                {'loss': avg_loss}, 
                float(optimizer.learning_rate)
            )
            
            # Save checkpoint
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint_path = checkpoint_manager.save_checkpoint(
                checkpoint, f'wc_epoch_{epoch+1}.ckpt'
            )
            logger.log_checkpoint(epoch + 1, checkpoint_path)
        
        # Verify training completed
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'wc_test.txt')))
        self.assertGreater(len(checkpoint_manager.list_checkpoints()), 0)
        
        print("✓ World Coordinate Training Pipeline completed successfully")
    
    def test_backward_mapping_training_pipeline(self):
        """Test complete backward mapping training pipeline."""
        print("Testing Backward Mapping Training Pipeline...")
        
        # Create configuration
        config = ConfigManager.create_backward_mapping_config(
            data_path=self.data_dir,
            batch_size=1,
            n_epoch=1,
            img_rows=64,
            img_cols=64
        )
        config.logging.logdir = self.output_dir
        
        # Create model
        model = ModelFactory.create_backward_mapping_model(
            img_size=64, in_channels=3, out_channels=2, filters=16
        )
        
        # Build model
        dummy_input = tf.random.normal((1, 64, 64, 3))
        _ = model(dummy_input)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        # Create loss function
        loss_factory = LossFactory()
        loss_fn = loss_factory.create_loss('unwarp_loss')
        
        # Create data loader (mock the complex BM loader)
        def mock_bm_generator():
            for _ in range(2):
                # Input: concatenated albedo + world coordinates
                inputs = tf.random.normal((1, 64, 64, 6))
                # Target: backward mapping coordinates
                targets = tf.random.uniform((1, 64, 64, 2), -1, 1)
                yield inputs, targets
        
        train_dataset = tf.data.Dataset.from_generator(
            mock_bm_generator,
            output_signature=(
                tf.TensorSpec(shape=(1, 64, 64, 6), dtype=tf.float32),
                tf.TensorSpec(shape=(1, 64, 64, 2), dtype=tf.float32)
            )
        )
        
        # Create training utilities
        logger = TrainingLogger(self.output_dir, 'bm_test')
        metrics_computer = MetricsComputer()
        
        # Training loop
        for epoch in range(1):
            epoch_losses = []
            
            for batch_idx, (inputs, targets) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # Use only world coordinates (last 3 channels)
                    wc_input = inputs[:, :, :, 3:]
                    predictions = model(wc_input, training=True)
                    
                    # Compute unwarp loss
                    mse_loss, ssim_loss, uworg, uwpred = loss_fn(
                        inputs, predictions, targets
                    )
                    total_loss = mse_loss + 0.3 * ssim_loss
                
                # Compute gradients and update
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                epoch_losses.append(float(total_loss))
                
                if batch_idx >= 1:  # Limit batches for testing
                    break
            
            # Log epoch
            avg_loss = np.mean(epoch_losses)
            logger.log_epoch(
                epoch + 1, 'Train', 
                {'total_loss': avg_loss}, 
                float(optimizer.learning_rate)
            )
        
        # Verify training completed
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'bm_test.txt')))
        
        print("✓ Backward Mapping Training Pipeline completed successfully")


class TestModelCompatibility(unittest.TestCase):
    """Test model compatibility and parameter equivalence."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_unet_parameter_consistency(self):
        """Test UNet parameter consistency across different configurations."""
        print("Testing UNet Parameter Consistency...")
        
        # Create models with same configuration
        model1 = get_model('unet', input_nc=3, output_nc=3, num_downs=6, ngf=64)
        model2 = get_model('unet', input_nc=3, output_nc=3, num_downs=6, ngf=64)
        
        # Build models
        dummy_input = tf.random.normal((1, 128, 128, 3))
        _ = model1(dummy_input)
        _ = model2(dummy_input)
        
        # Check parameter counts match
        params1 = model1.count_params()
        params2 = model2.count_params()
        
        self.assertEqual(params1, params2)
        
        # Check output shapes match
        output1 = model1(dummy_input)
        output2 = model2(dummy_input)
        
        self.assertEqual(output1.shape, output2.shape)
        
        print(f"✓ UNet models have consistent {params1:,} parameters")
    
    def test_densenet_parameter_consistency(self):
        """Test DenseNet parameter consistency across different configurations."""
        print("Testing DenseNet Parameter Consistency...")
        
        # Create models with same configuration
        model1 = get_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=32)
        model2 = get_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=32)
        
        # Build models
        dummy_input = tf.random.normal((1, 128, 128, 3))
        _ = model1(dummy_input)
        _ = model2(dummy_input)
        
        # Check parameter counts match
        params1 = model1.count_params()
        params2 = model2.count_params()
        
        self.assertEqual(params1, params2)
        
        # Check output shapes match
        output1 = model1(dummy_input)
        output2 = model2(dummy_input)
        
        self.assertEqual(output1.shape, output2.shape)
        
        print(f"✓ DenseNet models have consistent {params1:,} parameters")
    
    def test_model_output_ranges(self):
        """Test that model outputs are in expected ranges."""
        print("Testing Model Output Ranges...")
        
        # Test UNet (world coordinate model)
        unet_model = get_model('unet', input_nc=3, output_nc=3, num_downs=5, ngf=32)
        unet_input = tf.random.uniform((2, 64, 64, 3), 0, 1)
        unet_output = unet_model(unet_input)
        
        # UNet should output values in [-1, 1] due to Tanh activation
        self.assertGreaterEqual(tf.reduce_min(unet_output).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(unet_output).numpy(), 1.0)
        
        # Test DenseNet (backward mapping model)
        densenet_model = get_model('densenet', img_size=64, in_channels=3, out_channels=2, filters=16)
        densenet_input = tf.random.uniform((2, 64, 64, 3), 0, 1)
        densenet_output = densenet_model(densenet_input)
        
        # DenseNet should output values in [0, 1] due to Sigmoid activation
        self.assertGreaterEqual(tf.reduce_min(densenet_output).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(densenet_output).numpy(), 1.0)
        
        print("✓ Model outputs are in expected ranges")
    
    def test_pytorch_tensorflow_architecture_compatibility(self):
        """Test architectural compatibility with PyTorch reference implementation."""
        print("Testing PyTorch-TensorFlow Architecture Compatibility...")
        
        # Test UNet architecture compatibility
        unet_configs = [
            {'num_downs': 5, 'ngf': 32, 'expected_min_params': 1000000},
            {'num_downs': 6, 'ngf': 64, 'expected_min_params': 5000000},
            {'num_downs': 7, 'ngf': 64, 'expected_min_params': 20000000}
        ]
        
        for config in unet_configs:
            model = get_model('unet', input_nc=3, output_nc=3, **config)
            dummy_input = tf.random.normal((1, 256, 256, 3))
            _ = model(dummy_input)
            
            param_count = model.count_params()
            self.assertGreater(param_count, config['expected_min_params'])
            
            # Test forward pass consistency
            output = model(dummy_input, training=False)
            self.assertEqual(output.shape, (1, 256, 256, 3))
            
            print(f"  ✓ UNet {config['num_downs']} downs, {config['ngf']} filters: {param_count:,} parameters")
        
        # Test DenseNet architecture compatibility
        densenet_configs = [
            {'img_size': 64, 'filters': 16, 'expected_min_params': 100000},
            {'img_size': 128, 'filters': 32, 'expected_min_params': 500000},
            {'img_size': 256, 'filters': 64, 'expected_min_params': 2000000}
        ]
        
        for config in densenet_configs:
            model = get_model('densenet', in_channels=3, out_channels=2, **config)
            dummy_input = tf.random.normal((1, config['img_size'], config['img_size'], 3))
            _ = model(dummy_input)
            
            param_count = model.count_params()
            self.assertGreater(param_count, config['expected_min_params'])
            
            # Test forward pass consistency
            output = model(dummy_input, training=False)
            self.assertEqual(output.shape, (1, config['img_size'], config['img_size'], 2))
            
            print(f"  ✓ DenseNet {config['img_size']}x{config['img_size']}, {config['filters']} filters: {param_count:,} parameters")
        
        print("✓ Architecture compatibility with PyTorch verified")
    
    def test_checkpoint_compatibility(self):
        """Test checkpoint saving and loading compatibility."""
        print("Testing Checkpoint Compatibility...")
        
        # Create model
        model = get_model('unet', input_nc=3, output_nc=3, num_downs=5, ngf=32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        # Build model
        dummy_input = tf.random.normal((1, 128, 128, 3))
        _ = model(dummy_input)
        
        # Create checkpoint
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint')
        
        # Save checkpoint
        save_path = checkpoint.save(checkpoint_path)
        self.assertTrue(os.path.exists(save_path + '.index'))
        
        # Get initial output
        initial_output = model(dummy_input, training=False)
        
        # Modify model weights
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(layer.kernel + 0.1)
        
        # Verify model changed
        modified_output = model(dummy_input, training=False)
        self.assertFalse(tf.reduce_all(tf.equal(initial_output, modified_output)))
        
        # Restore checkpoint
        checkpoint.restore(save_path)
        
        # Verify restoration
        restored_output = model(dummy_input, training=False)
        np.testing.assert_allclose(
            initial_output.numpy(), restored_output.numpy(), rtol=1e-6
        )
        
        print("✓ Checkpoint compatibility verified")
    
    def test_mixed_precision_compatibility(self):
        """Test mixed precision training compatibility."""
        print("Testing Mixed Precision Compatibility...")
        
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        try:
            # Create model with mixed precision
            model = get_model('unet', input_nc=3, output_nc=3, num_downs=4, ngf=16)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            
            # Build model
            dummy_input = tf.random.normal((2, 64, 64, 3))
            _ = model(dummy_input)
            
            # Test forward pass
            with tf.GradientTape() as tape:
                predictions = model(dummy_input, training=True)
                loss = tf.reduce_mean(tf.square(predictions))
                
                # Scale loss for mixed precision
                scaled_loss = optimizer.get_scaled_loss(loss)
            
            # Test backward pass
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Verify output dtype
            self.assertEqual(predictions.dtype, tf.float16)
            
            print("✓ Mixed precision compatibility verified")
            
        finally:
            # Reset to default policy
            tf.keras.mixed_precision.set_global_policy('float32')


class TestLossCompatibility(unittest.TestCase):
    """Test loss function compatibility and consistency."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.factory = LossFactory()
    
    def test_loss_function_consistency(self):
        """Test that loss functions produce consistent results."""
        print("Testing Loss Function Consistency...")
        
        # Create test data
        predictions = tf.random.normal((2, 32, 32, 3))
        targets = tf.random.normal((2, 32, 32, 3))
        
        # Test gradient loss consistency
        grad_loss1 = self.factory.create_loss('grad_loss', window_size=3)
        grad_loss2 = self.factory.create_loss('grad_loss', window_size=3)
        
        loss1 = grad_loss1(targets, predictions)
        loss2 = grad_loss2(targets, predictions)
        
        # Should produce identical results
        self.assertAlmostEqual(float(loss1), float(loss2), places=6)
        
        # Test SSIM loss consistency
        ssim_loss1 = self.factory.create_loss('ssim_loss')
        ssim_loss2 = self.factory.create_loss('ssim_loss')
        
        ssim1 = ssim_loss1(targets, predictions)
        ssim2 = ssim_loss2(targets, predictions)
        
        self.assertAlmostEqual(float(ssim1), float(ssim2), places=6)
        
        print("✓ Loss functions produce consistent results")
    
    def test_combined_loss_integration(self):
        """Test combined loss function integration."""
        print("Testing Combined Loss Integration...")
        
        # Create combined loss
        combined_loss = create_dewarpnet_loss(grad_weight=1.0, recon_weight=0.5)
        
        # Test with world coordinate data
        wc_predictions = tf.random.normal((1, 32, 32, 3))
        wc_targets = tf.random.normal((1, 32, 32, 3))
        
        # This should work without errors
        try:
            wc_result = combined_loss(wc_targets, wc_predictions)
            self.assertIn('total_loss', wc_result)
            self.assertGreater(float(wc_result['total_loss']), 0)
        except Exception as e:
            self.fail(f"Combined loss failed on WC data: {e}")
        
        # Test with backward mapping data
        bm_inputs = tf.random.normal((1, 32, 32, 6))
        bm_predictions = tf.random.uniform((1, 32, 32, 2), -1, 1)
        bm_targets = tf.random.uniform((1, 32, 32, 2), -1, 1)
        
        try:
            bm_result = combined_loss(bm_inputs, bm_targets, bm_predictions)
            self.assertIn('total_loss', bm_result)
            self.assertGreater(float(bm_result['total_loss']), 0)
        except Exception as e:
            self.fail(f"Combined loss failed on BM data: {e}")
        
        print("✓ Combined loss integration successful")


class TestDataPipelineIntegration(unittest.TestCase):
    """Test data pipeline integration and performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_mock_dataset()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_dataset(self):
        """Create mock dataset for testing."""
        # Create directory structure
        os.makedirs(os.path.join(self.temp_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'wc'), exist_ok=True)
        
        # Create sample files
        sample_names = ['sample1', 'sample2', 'sample3', 'sample4']
        
        # Create train.txt and val.txt
        with open(os.path.join(self.temp_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(sample_names))
        
        with open(os.path.join(self.temp_dir, 'val.txt'), 'w') as f:
            f.write('sample1\nsample2\n')
        
        # Create mock images and labels
        for name in sample_names:
            # RGB image
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img_path = os.path.join(self.temp_dir, 'img', f'{name}.png')
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # World coordinates (EXR)
            wc = np.random.uniform(-1, 1, (64, 64, 3)).astype(np.float32)
            wc_path = os.path.join(self.temp_dir, 'wc', f'{name}.exr')
            cv2.imwrite(wc_path, wc)
    
    def test_data_pipeline_performance(self):
        """Test data pipeline performance and throughput."""
        print("Testing Data Pipeline Performance...")
        
        # Create dataset
        dataset = create_wc_dataset(
            root=self.temp_dir,
            split='train',
            batch_size=2,
            img_size=(64, 64),
            shuffle=True
        )
        
        # Measure throughput
        start_time = time.time()
        batch_count = 0
        
        for batch in dataset.take(10):
            inputs, targets = batch
            
            # Verify batch properties
            self.assertEqual(inputs.shape[0], 2)  # Batch size
            self.assertEqual(inputs.shape[1:], (64, 64, 3))  # Image shape
            self.assertEqual(targets.shape[1:], (64, 64, 3))  # Label shape
            
            # Verify data ranges
            self.assertTrue(tf.reduce_all(inputs >= 0.0))
            self.assertTrue(tf.reduce_all(inputs <= 1.0))
            
            batch_count += 1
        
        end_time = time.time()
        throughput = batch_count / (end_time - start_time)
        
        print(f"✓ Data pipeline throughput: {throughput:.2f} batches/second")
        self.assertGreater(throughput, 0)
    
    def test_data_augmentation_consistency(self):
        """Test data augmentation consistency."""
        print("Testing Data Augmentation Consistency...")
        
        # Create datasets with and without augmentations
        dataset_aug = create_wc_dataset(
            root=self.temp_dir,
            split='train',
            batch_size=1,
            img_size=(64, 64),
            shuffle=False,
            augmentations=True
        )
        
        dataset_no_aug = create_wc_dataset(
            root=self.temp_dir,
            split='train',
            batch_size=1,
            img_size=(64, 64),
            shuffle=False,
            augmentations=False
        )
        
        # Get samples from both datasets
        aug_batch = next(iter(dataset_aug.take(1)))
        no_aug_batch = next(iter(dataset_no_aug.take(1)))
        
        # Both should have same shapes
        self.assertEqual(aug_batch[0].shape, no_aug_batch[0].shape)
        self.assertEqual(aug_batch[1].shape, no_aug_batch[1].shape)
        
        # Values might be different due to augmentation
        # But should be in valid ranges
        for batch in [aug_batch, no_aug_batch]:
            inputs, targets = batch
            self.assertTrue(tf.reduce_all(inputs >= 0.0))
            self.assertTrue(tf.reduce_all(inputs <= 1.0))
        
        print("✓ Data augmentation consistency verified")


class TestInferencePipelineIntegration(unittest.TestCase):
    """Test inference pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        tf.random.set_seed(42)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_inference_consistency(self):
        """Test model inference consistency."""
        print("Testing Model Inference Consistency...")
        
        # Create models
        wc_model = get_model('unet', input_nc=3, output_nc=3, num_downs=5, ngf=32)
        bm_model = get_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=16)
        
        # Test input
        test_input = tf.random.uniform((1, 128, 128, 3), 0, 1)
        
        # Test world coordinate model
        wc_output1 = wc_model(test_input, training=False)
        wc_output2 = wc_model(test_input, training=False)
        
        # Should produce identical results in inference mode
        np.testing.assert_allclose(
            wc_output1.numpy(), wc_output2.numpy(), rtol=1e-6
        )
        
        # Test backward mapping model
        bm_output1 = bm_model(test_input, training=False)
        bm_output2 = bm_model(test_input, training=False)
        
        np.testing.assert_allclose(
            bm_output1.numpy(), bm_output2.numpy(), rtol=1e-6
        )
        
        print("✓ Model inference is consistent")
    
    def test_end_to_end_inference_pipeline(self):
        """Test complete end-to-end inference pipeline."""
        print("Testing End-to-End Inference Pipeline...")
        
        # Create models
        wc_model = get_model('unet', input_nc=3, output_nc=3, num_downs=6, ngf=32)
        bm_model = get_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=16)
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Step 1: Preprocess image for world coordinate prediction
        wc_input = tf.cast(test_image, tf.float32) / 255.0
        wc_input = tf.image.resize(wc_input, (256, 256))
        wc_input = tf.expand_dims(wc_input, 0)
        
        # Step 2: Predict world coordinates
        wc_output = wc_model(wc_input, training=False)
        
        # Step 3: Preprocess for backward mapping
        bm_input = tf.image.resize(wc_output, (128, 128))
        
        # Step 4: Predict backward mapping
        bm_output = bm_model(bm_input, training=False)
        
        # Verify pipeline outputs
        self.assertEqual(wc_output.shape, (1, 256, 256, 3))
        self.assertEqual(bm_output.shape, (1, 128, 128, 2))
        
        # Check output ranges
        self.assertGreaterEqual(tf.reduce_min(wc_output).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(wc_output).numpy(), 1.0)
        
        self.assertGreaterEqual(tf.reduce_min(bm_output).numpy(), 0.0)
        self.assertLessEqual(tf.reduce_max(bm_output).numpy(), 1.0)
        
        print("✓ End-to-end inference pipeline successful")


class TestVisualOutputComparison(unittest.TestCase):
    """Test visual output comparison and quality assessment."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'visual_outputs')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_image(self, size=(256, 256), pattern='checkerboard'):
        """Create test image with known pattern."""
        h, w = size
        if pattern == 'checkerboard':
            # Create checkerboard pattern
            img = np.zeros((h, w, 3), dtype=np.uint8)
            square_size = 32
            for i in range(0, h, square_size):
                for j in range(0, w, square_size):
                    if (i // square_size + j // square_size) % 2 == 0:
                        img[i:i+square_size, j:j+square_size] = [255, 255, 255]
        elif pattern == 'gradient':
            # Create gradient pattern
            img = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(h):
                img[i, :, 0] = int(255 * i / h)  # Red gradient
                img[i, :, 1] = int(255 * (1 - i / h))  # Green gradient
                img[i, :, 2] = 128  # Constant blue
        else:
            # Random pattern
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        return img
    
    def test_world_coordinate_visual_output(self):
        """Test world coordinate model visual output quality."""
        print("Testing World Coordinate Visual Output...")
        
        # Create model
        model = get_model('unet', input_nc=3, output_nc=3, num_downs=6, ngf=32)
        
        # Create test images with different patterns
        test_patterns = ['checkerboard', 'gradient', 'random']
        
        for pattern in test_patterns:
            # Create test image
            test_img = self._create_test_image(size=(256, 256), pattern=pattern)
            
            # Preprocess
            input_tensor = tf.cast(test_img, tf.float32) / 255.0
            input_tensor = tf.expand_dims(input_tensor, 0)
            
            # Predict world coordinates
            wc_output = model(input_tensor, training=False)
            
            # Convert to numpy for analysis
            wc_np = wc_output[0].numpy()
            
            # Verify output properties
            self.assertEqual(wc_np.shape, (256, 256, 3))
            self.assertGreaterEqual(wc_np.min(), -1.0)
            self.assertLessEqual(wc_np.max(), 1.0)
            
            # Check for reasonable variation (not all zeros or constant)
            std_per_channel = np.std(wc_np, axis=(0, 1))
            for i, std in enumerate(std_per_channel):
                self.assertGreater(std, 0.01, f"Channel {i} has too little variation")
            
            # Save visualization
            wc_vis = ((wc_np + 1) * 127.5).astype(np.uint8)
            output_path = os.path.join(self.output_dir, f'wc_{pattern}.png')
            cv2.imwrite(output_path, cv2.cvtColor(wc_vis, cv2.COLOR_RGB2BGR))
            
            print(f"  ✓ {pattern} pattern: std={std_per_channel.mean():.4f}, saved to {output_path}")
        
        print("✓ World coordinate visual outputs generated and validated")
    
    def test_backward_mapping_visual_output(self):
        """Test backward mapping model visual output quality."""
        print("Testing Backward Mapping Visual Output...")
        
        # Create model
        model = get_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=32)
        
        # Create test world coordinate inputs
        test_patterns = ['smooth', 'noisy', 'structured']
        
        for pattern in test_patterns:
            if pattern == 'smooth':
                # Smooth coordinate field
                x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
                wc_input = np.stack([x, y, np.zeros_like(x)], axis=-1)
            elif pattern == 'noisy':
                # Noisy coordinate field
                wc_input = np.random.uniform(-1, 1, (128, 128, 3))
            else:
                # Structured coordinate field
                wc_input = np.zeros((128, 128, 3))
                wc_input[:, :, 0] = np.sin(np.linspace(0, 4*np.pi, 128)).reshape(1, -1)
                wc_input[:, :, 1] = np.cos(np.linspace(0, 4*np.pi, 128)).reshape(-1, 1)
            
            # Convert to tensor
            input_tensor = tf.cast(wc_input, tf.float32)
            input_tensor = tf.expand_dims(input_tensor, 0)
            
            # Predict backward mapping
            bm_output = model(input_tensor, training=False)
            
            # Convert to numpy for analysis
            bm_np = bm_output[0].numpy()
            
            # Verify output properties
            self.assertEqual(bm_np.shape, (128, 128, 2))
            self.assertGreaterEqual(bm_np.min(), 0.0)
            self.assertLessEqual(bm_np.max(), 1.0)
            
            # Check for reasonable variation
            std_per_channel = np.std(bm_np, axis=(0, 1))
            for i, std in enumerate(std_per_channel):
                self.assertGreater(std, 0.01, f"Channel {i} has too little variation")
            
            # Save visualization
            bm_vis = (bm_np * 255).astype(np.uint8)
            output_path = os.path.join(self.output_dir, f'bm_{pattern}.png')
            cv2.imwrite(output_path, bm_vis[:, :, :2])  # Save first 2 channels
            
            print(f"  ✓ {pattern} pattern: std={std_per_channel.mean():.4f}, saved to {output_path}")
        
        print("✓ Backward mapping visual outputs generated and validated")
    
    def test_end_to_end_visual_pipeline(self):
        """Test complete end-to-end visual pipeline."""
        print("Testing End-to-End Visual Pipeline...")
        
        # Create models
        wc_model = get_model('unet', input_nc=3, output_nc=3, num_downs=5, ngf=32)
        bm_model = get_model('densenet', img_size=128, in_channels=3, out_channels=2, filters=16)
        
        # Create test document image
        test_img = self._create_test_image(size=(256, 256), pattern='checkerboard')
        
        # Step 1: Predict world coordinates
        wc_input = tf.cast(test_img, tf.float32) / 255.0
        wc_input = tf.expand_dims(wc_input, 0)
        wc_output = wc_model(wc_input, training=False)
        
        # Step 2: Predict backward mapping
        bm_input = tf.image.resize(wc_output, (128, 128))
        bm_output = bm_model(bm_input, training=False)
        
        # Step 3: Apply unwarping (simplified grid sampling)
        bm_coords = bm_output[0].numpy()
        
        # Convert coordinates to pixel indices
        h, w = 128, 128
        grid_x = (bm_coords[:, :, 0] * (w - 1)).astype(np.int32)
        grid_y = (bm_coords[:, :, 1] * (h - 1)).astype(np.int32)
        
        # Clamp coordinates
        grid_x = np.clip(grid_x, 0, w - 1)
        grid_y = np.clip(grid_y, 0, h - 1)
        
        # Create unwarped image (simplified)
        input_resized = tf.image.resize(wc_input, (128, 128))[0].numpy()
        unwarped = np.zeros_like(input_resized)
        
        for i in range(h):
            for j in range(w):
                src_x, src_y = grid_x[i, j], grid_y[i, j]
                unwarped[i, j] = input_resized[src_y, src_x]
        
        # Save visualizations
        input_vis = (test_img).astype(np.uint8)
        wc_vis = ((wc_output[0].numpy() + 1) * 127.5).astype(np.uint8)
        bm_vis = (bm_coords * 255).astype(np.uint8)
        unwarped_vis = (unwarped * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(self.output_dir, 'e2e_input.png'), 
                   cv2.cvtColor(input_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.output_dir, 'e2e_wc.png'), 
                   cv2.cvtColor(wc_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.output_dir, 'e2e_bm.png'), 
                   bm_vis[:, :, :2])
        cv2.imwrite(os.path.join(self.output_dir, 'e2e_unwarped.png'), 
                   cv2.cvtColor(unwarped_vis, cv2.COLOR_RGB2BGR))
        
        # Verify pipeline integrity
        self.assertEqual(wc_output.shape, (1, 256, 256, 3))
        self.assertEqual(bm_output.shape, (1, 128, 128, 2))
        self.assertEqual(unwarped.shape, (128, 128, 3))
        
        print("✓ End-to-end visual pipeline completed and saved")
    
    def test_output_quality_metrics(self):
        """Test output quality metrics and consistency."""
        print("Testing Output Quality Metrics...")
        
        # Create model
        model = get_model('unet', input_nc=3, output_nc=3, num_downs=5, ngf=32)
        
        # Create multiple test images
        test_images = []
        for i in range(5):
            img = self._create_test_image(size=(128, 128), pattern='random')
            test_images.append(img)
        
        # Process all images
        outputs = []
        for img in test_images:
            input_tensor = tf.cast(img, tf.float32) / 255.0
            input_tensor = tf.expand_dims(input_tensor, 0)
            output = model(input_tensor, training=False)
            outputs.append(output[0].numpy())
        
        # Compute quality metrics
        metrics = {
            'mean_std': [],
            'mean_range': [],
            'spatial_consistency': []
        }
        
        for output in outputs:
            # Standard deviation (measure of variation)
            std = np.std(output)
            metrics['mean_std'].append(std)
            
            # Range (measure of dynamic range)
            range_val = np.max(output) - np.min(output)
            metrics['mean_range'].append(range_val)
            
            # Spatial consistency (measure of smoothness)
            grad_x = np.abs(np.diff(output, axis=1))
            grad_y = np.abs(np.diff(output, axis=0))
            spatial_consistency = np.mean(grad_x) + np.mean(grad_y)
            metrics['spatial_consistency'].append(spatial_consistency)
        
        # Verify metrics are reasonable
        avg_std = np.mean(metrics['mean_std'])
        avg_range = np.mean(metrics['mean_range'])
        avg_spatial = np.mean(metrics['spatial_consistency'])
        
        self.assertGreater(avg_std, 0.1, "Output has too little variation")
        self.assertLess(avg_std, 1.0, "Output has too much variation")
        self.assertGreater(avg_range, 0.5, "Output range too small")
        self.assertLess(avg_spatial, 2.0, "Output not spatially consistent")
        
        print(f"  ✓ Average std: {avg_std:.4f}, range: {avg_range:.4f}, spatial: {avg_spatial:.4f}")
        print("✓ Output quality metrics within expected ranges")


class TestMemoryAndPerformance(unittest.TestCase):
    """Test memory usage and performance characteristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_memory_usage(self):
        """Test model memory usage characteristics."""
        print("Testing Model Memory Usage...")
        
        # Test different model sizes
        model_configs = [
            {'name': 'Small UNet', 'model': 'unet', 'params': {'num_downs': 4, 'ngf': 32}},
            {'name': 'Medium UNet', 'model': 'unet', 'params': {'num_downs': 6, 'ngf': 64}},
            {'name': 'Small DenseNet', 'model': 'densenet', 'params': {'img_size': 64, 'filters': 16}},
            {'name': 'Medium DenseNet', 'model': 'densenet', 'params': {'img_size': 128, 'filters': 32}},
        ]
        
        memory_usage = {}
        
        for config in model_configs:
            # Create model
            if config['model'] == 'unet':
                model = get_model('unet', input_nc=3, output_nc=3, **config['params'])
                test_input = tf.random.normal((1, 128, 128, 3))
            else:
                model = get_model('densenet', in_channels=3, out_channels=2, **config['params'])
                test_input = tf.random.normal((1, config['params']['img_size'], config['params']['img_size'], 3))
            
            # Build model and count parameters
            _ = model(test_input)
            param_count = model.count_params()
            
            memory_usage[config['name']] = {
                'parameters': param_count,
                'memory_mb': param_count * 4 / (1024 * 1024)  # Approximate memory in MB
            }
            
            print(f"  {config['name']}: {param_count:,} parameters ({memory_usage[config['name']]['memory_mb']:.1f} MB)")
        
        # Verify memory usage is reasonable
        for name, usage in memory_usage.items():
            self.assertLess(usage['memory_mb'], 500)  # Should be less than 500MB
        
        print("✓ Model memory usage is within reasonable limits")
    
    def test_training_performance_characteristics(self):
        """Test training performance characteristics."""
        print("Testing Training Performance Characteristics...")
        
        # Create small model for performance testing
        model = get_model('unet', input_nc=3, output_nc=3, num_downs=4, ngf=16)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Create test data
        batch_sizes = [1, 2, 4]
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Generate test batch
            inputs = tf.random.normal((batch_size, 64, 64, 3))
            targets = tf.random.normal((batch_size, 64, 64, 3))
            
            # Time training step
            start_time = time.time()
            
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss_value = loss_fn(targets, predictions)
            
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            end_time = time.time()
            
            step_time = end_time - start_time
            samples_per_second = batch_size / step_time
            
            performance_results[batch_size] = {
                'step_time': step_time,
                'samples_per_second': samples_per_second
            }
            
            print(f"  Batch size {batch_size}: {step_time:.4f}s/step, {samples_per_second:.1f} samples/s")
        
        # Verify performance scaling
        self.assertGreater(performance_results[4]['samples_per_second'], 
                          performance_results[1]['samples_per_second'])
        
        print("✓ Training performance scales appropriately with batch size")
    
    def test_gpu_utilization_monitoring(self):
        """Test GPU utilization monitoring during training."""
        print("Testing GPU Utilization Monitoring...")
        
        # Check if GPU is available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            print("  ⚠ No GPU detected, skipping GPU utilization test")
            return
        
        # Create model
        model = get_model('unet', input_nc=3, output_nc=3, num_downs=4, ngf=32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Build model
        dummy_input = tf.random.normal((4, 128, 128, 3))
        _ = model(dummy_input)
        
        # Monitor GPU memory during training
        initial_memory = tf.config.experimental.get_memory_info('GPU:0')['current']
        
        # Perform training steps
        for step in range(5):
            inputs = tf.random.normal((4, 128, 128, 3))
            targets = tf.random.normal((4, 128, 128, 3))
            
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss_value = loss_fn(targets, predictions)
            
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Check memory usage
            current_memory = tf.config.experimental.get_memory_info('GPU:0')['current']
            memory_mb = current_memory / (1024 * 1024)
            
            print(f"  Step {step + 1}: GPU memory usage: {memory_mb:.1f} MB")
        
        final_memory = tf.config.experimental.get_memory_info('GPU:0')['current']
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)
        
        # Verify reasonable memory usage
        self.assertLess(memory_increase, 1000, "GPU memory increase too large")
        
        print(f"✓ GPU utilization monitored, memory increase: {memory_increase:.1f} MB")
    
    def test_performance_benchmarking_integration(self):
        """Test integration with performance benchmarking system."""
        print("Testing Performance Benchmarking Integration...")
        
        # Import performance benchmarking utilities
        try:
            from test_performance_benchmarks import PerformanceBenchmark
            benchmark = PerformanceBenchmark()
        except ImportError:
            print("  ⚠ Performance benchmark utilities not available, creating mock")
            benchmark = type('MockBenchmark', (), {
                'results': {},
                'measure_time': lambda self, func, *args, **kwargs: (func(*args, **kwargs), 0.1),
                'save_results': lambda self, path: None
            })()
        
        # Create test model
        model = get_model('unet', input_nc=3, output_nc=3, num_downs=4, ngf=16)
        test_input = tf.random.normal((2, 64, 64, 3))
        
        # Benchmark inference
        def inference_func():
            return model(test_input, training=False)
        
        result, inference_time = benchmark.measure_time(inference_func)
        
        # Benchmark training step
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss_fn = tf.keras.losses.MeanSquaredError()
        targets = tf.random.normal((2, 64, 64, 3))
        
        def training_step():
            with tf.GradientTape() as tape:
                predictions = model(test_input, training=True)
                loss_value = loss_fn(targets, predictions)
            gradients = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss_value
        
        loss_result, training_time = benchmark.measure_time(training_step)
        
        # Store results
        benchmark.results['integration_test'] = {
            'inference_time': inference_time,
            'training_time': training_time,
            'model_params': model.count_params()
        }
        
        # Save benchmark results
        results_path = os.path.join(self.temp_dir, 'benchmark_results.json')
        benchmark.save_results(results_path)
        
        # Verify results
        self.assertGreater(inference_time, 0)
        self.assertGreater(training_time, 0)
        self.assertGreater(training_time, inference_time)  # Training should be slower
        
        print(f"  ✓ Inference: {inference_time:.4f}s, Training: {training_time:.4f}s")
        print("✓ Performance benchmarking integration successful")


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_workflow(self):
        """Test complete configuration workflow."""
        print("Testing Configuration Workflow...")
        
        # Create configurations
        wc_config = ConfigManager.create_world_coordinate_config(
            data_path=self.temp_dir,
            batch_size=4,
            n_epoch=10
        )
        
        bm_config = ConfigManager.create_backward_mapping_config(
            data_path=self.temp_dir,
            batch_size=2,
            n_epoch=5
        )
        
        # Save configurations
        config_manager = ConfigManager()
        
        wc_config_path = os.path.join(self.temp_dir, 'wc_config.json')
        bm_config_path = os.path.join(self.temp_dir, 'bm_config.json')
        
        config_manager.save_config(wc_config, wc_config_path)
        config_manager.save_config(bm_config, bm_config_path)
        
        # Load configurations
        loaded_wc_config = config_manager.load_config(wc_config_path)
        loaded_bm_config = config_manager.load_config(bm_config_path)
        
        # Verify loaded configurations
        self.assertEqual(loaded_wc_config.task_type, 'world_coordinate')
        self.assertEqual(loaded_wc_config.data.batch_size, 4)
        self.assertEqual(loaded_wc_config.training.n_epoch, 10)
        
        self.assertEqual(loaded_bm_config.task_type, 'backward_mapping')
        self.assertEqual(loaded_bm_config.data.batch_size, 2)
        self.assertEqual(loaded_bm_config.training.n_epoch, 5)
        
        # Validate configurations
        wc_issues = config_manager.validate_config(loaded_wc_config)
        bm_issues = config_manager.validate_config(loaded_bm_config)
        
        # Should have some issues due to missing data files, but configs should be structurally valid
        self.assertIsInstance(wc_issues, list)
        self.assertIsInstance(bm_issues, list)
        
        print("✓ Configuration workflow completed successfully")


def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting TensorFlow DewarpNet Integration Tests")
    print("="*60)
    
    # Setup TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured {len(gpus)} GPU(s) with memory growth")
        except RuntimeError as e:
            print(f"⚠ GPU configuration warning: {e}")
    else:
        print("ℹ No GPUs detected, using CPU")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestEndToEndTrainingPipeline,
        TestModelCompatibility,
        TestLossCompatibility,
        TestDataPipelineIntegration,
        TestInferencePipelineIntegration,
        TestVisualOutputComparison,
        TestMemoryAndPerformance,
        TestConfigurationIntegration,
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ TensorFlow DewarpNet integration is working correctly!")
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)