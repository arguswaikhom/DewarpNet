"""
Unit tests for TensorFlow Data Pipeline.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.data_pipeline import (
    TFDataPipeline, 
    create_training_pipeline, 
    benchmark_dataset_performance,
    validate_dataset_integrity
)


class TestTFDataPipeline(unittest.TestCase):
    """Test cases for TFDataPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.alt_temp_dir = tempfile.mkdtemp()
        self.img_size = (128, 128)
        self.batch_size = 2
        
        # Create mock directory structure for WC task
        os.makedirs(os.path.join(self.temp_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'wc'), exist_ok=True)
        
        # Create mock directory structure for BM task
        os.makedirs(os.path.join(self.temp_dir, 'recon', '1', 'chess48'), exist_ok=True)
        os.makedirs(os.path.join(self.alt_temp_dir, 'wc'), exist_ok=True)
        os.makedirs(os.path.join(self.alt_temp_dir, 'bm'), exist_ok=True)
        
        # Create mock train.txt files
        with open(os.path.join(self.temp_dir, 'train.txt'), 'w') as f:
            f.write('sample1\nsample2\nsample3\n')
        
        with open(os.path.join(self.alt_temp_dir, 'train.txt'), 'w') as f:
            f.write('1/sample1\n1/sample2\n')
        
        # Create mock val.txt files
        with open(os.path.join(self.temp_dir, 'val.txt'), 'w') as f:
            f.write('val_sample1\nval_sample2\n')
        
        with open(os.path.join(self.alt_temp_dir, 'val.txt'), 'w') as f:
            f.write('1/val_sample1\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.alt_temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization_wc(self):
        """Test pipeline initialization for world coordinates."""
        pipeline = TFDataPipeline(
            root=self.temp_dir,
            task_type='wc',
            img_size=self.img_size
        )
        
        self.assertEqual(pipeline.root, self.temp_dir)
        self.assertEqual(pipeline.task_type, 'wc')
        self.assertEqual(pipeline.img_size, self.img_size)
    
    def test_pipeline_initialization_bm(self):
        """Test pipeline initialization for backward mapping."""
        pipeline = TFDataPipeline(
            root=self.temp_dir,
            task_type='bm',
            img_size=self.img_size,
            altroot=self.alt_temp_dir
        )
        
        self.assertEqual(pipeline.root, self.temp_dir)
        self.assertEqual(pipeline.task_type, 'bm')
        self.assertEqual(pipeline.img_size, self.img_size)
        self.assertEqual(pipeline.altroot, self.alt_temp_dir)
    
    def test_invalid_task_type(self):
        """Test that invalid task type raises error."""
        with self.assertRaises(ValueError):
            TFDataPipeline(
                root=self.temp_dir,
                task_type='invalid',
                img_size=self.img_size
            )
    
    @patch('cv2.imread')
    def test_create_wc_dataset(self, mock_imread):
        """Test world coordinate dataset creation."""
        # Mock image and EXR loading
        mock_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mock_wc = np.random.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.png'):
                return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
            elif path.endswith('.exr'):
                return mock_wc
            return None
        
        mock_imread.side_effect = imread_side_effect
        
        pipeline = TFDataPipeline(
            root=self.temp_dir,
            task_type='wc',
            img_size=self.img_size
        )
        
        dataset = pipeline.create_dataset(
            split='train',
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Check dataset properties
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Test iteration
        for batch in dataset.take(1):
            imgs, lbls = batch
            self.assertEqual(imgs.shape[0], self.batch_size)
            self.assertEqual(imgs.shape[1:], (*self.img_size, 3))
            self.assertEqual(lbls.shape[1:], (*self.img_size, 3))
    
    @patch('cv2.imread')
    @patch('scipy.io.loadmat')
    def test_create_bm_dataset(self, mock_loadmat, mock_imread):
        """Test backward mapping dataset creation."""
        # Mock data
        mock_wc = np.random.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
        mock_alb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mock_bm = np.random.uniform(0, 448, (448, 448, 2)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.exr'):
                return mock_wc
            elif path.endswith('.png'):
                return cv2.cvtColor(mock_alb, cv2.COLOR_RGB2BGR)
            return None
        
        mock_imread.side_effect = imread_side_effect
        mock_loadmat.return_value = {'bm': mock_bm}
        
        pipeline = TFDataPipeline(
            root=self.temp_dir,
            task_type='bm',
            img_size=self.img_size,
            altroot=self.alt_temp_dir
        )
        
        dataset = pipeline.create_dataset(
            split='train',
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Check dataset properties
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Test iteration
        for batch in dataset.take(1):
            imgs, lbls = batch
            self.assertEqual(imgs.shape[0], self.batch_size)
            self.assertEqual(imgs.shape[1:], (*self.img_size, 6))  # albedo + wc
            self.assertEqual(lbls.shape[1:], (*self.img_size, 2))  # backward mapping
    
    @patch('cv2.imread')
    def test_dataset_optimizations(self, mock_imread):
        """Test dataset optimization features."""
        # Mock image and EXR loading
        mock_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mock_wc = np.random.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.png'):
                return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
            elif path.endswith('.exr'):
                return mock_wc
            return None
        
        mock_imread.side_effect = imread_side_effect
        
        pipeline = TFDataPipeline(
            root=self.temp_dir,
            task_type='wc',
            img_size=self.img_size
        )
        
        # Test with various optimization options
        dataset = pipeline.create_dataset(
            split='train',
            batch_size=self.batch_size,
            shuffle=True,
            cache=True,
            repeat=True,
            drop_remainder=True,
            prefetch_buffer=2
        )
        
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Test that we can iterate (at least a few batches)
        batch_count = 0
        for batch in dataset.take(3):
            batch_count += 1
            imgs, lbls = batch
            self.assertEqual(imgs.shape[0], self.batch_size)
        
        self.assertEqual(batch_count, 3)
    
    @patch('cv2.imread')
    def test_get_dataset_info(self, mock_imread):
        """Test dataset info retrieval."""
        # Mock image and EXR loading
        mock_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mock_wc = np.random.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.png'):
                return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
            elif path.endswith('.exr'):
                return mock_wc
            return None
        
        mock_imread.side_effect = imread_side_effect
        
        pipeline = TFDataPipeline(
            root=self.temp_dir,
            task_type='wc',
            img_size=self.img_size
        )
        
        info = pipeline.get_dataset_info(split='train')
        
        self.assertEqual(info['task_type'], 'wc')
        self.assertEqual(info['split'], 'train')
        self.assertEqual(info['num_samples'], 3)
        self.assertEqual(info['img_size'], self.img_size)
        self.assertEqual(info['root'], self.temp_dir)


class TestTrainingPipeline(unittest.TestCase):
    """Test cases for training pipeline creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.img_size = (128, 128)
        
        # Create mock directory structure
        os.makedirs(os.path.join(self.temp_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'wc'), exist_ok=True)
        
        # Create mock train.txt and val.txt files
        with open(os.path.join(self.temp_dir, 'train.txt'), 'w') as f:
            f.write('sample1\nsample2\nsample3\n')
        
        with open(os.path.join(self.temp_dir, 'val.txt'), 'w') as f:
            f.write('val_sample1\nval_sample2\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cv2.imread')
    def test_create_training_pipeline(self, mock_imread):
        """Test training pipeline creation."""
        # Mock image and EXR loading
        mock_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mock_wc = np.random.uniform(-1, 1, (128, 128, 3)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.png'):
                return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
            elif path.endswith('.exr'):
                return mock_wc
            return None
        
        mock_imread.side_effect = imread_side_effect
        
        train_dataset, val_dataset = create_training_pipeline(
            root=self.temp_dir,
            task_type='wc',
            img_size=self.img_size,
            batch_size=2,
            augmentations=True
        )
        
        # Check that both datasets are created
        self.assertIsInstance(train_dataset, tf.data.Dataset)
        self.assertIsInstance(val_dataset, tf.data.Dataset)
        
        # Test train dataset
        for batch in train_dataset.take(1):
            imgs, lbls = batch
            self.assertEqual(imgs.shape[0], 2)  # batch size
            self.assertEqual(imgs.shape[1:], (*self.img_size, 3))
        
        # Test val dataset
        for batch in val_dataset.take(1):
            imgs, lbls = batch
            self.assertEqual(imgs.shape[1:], (*self.img_size, 3))


class TestDatasetUtilities(unittest.TestCase):
    """Test cases for dataset utility functions."""
    
    def test_benchmark_dataset_performance(self):
        """Test dataset performance benchmarking."""
        # Create a simple mock dataset
        def generator():
            for i in range(50):
                img = tf.random.normal((128, 128, 3))
                lbl = tf.random.normal((128, 128, 3))
                yield img, lbl
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32)
            )
        ).batch(4)
        
        # Benchmark performance
        metrics = benchmark_dataset_performance(
            dataset,
            num_batches=10,
            warmup_batches=2
        )
        
        # Check that metrics are returned
        self.assertIn('total_time', metrics)
        self.assertIn('avg_batch_time', metrics)
        self.assertIn('batches_per_second', metrics)
        self.assertGreater(metrics['total_time'], 0)
        self.assertGreater(metrics['avg_batch_time'], 0)
        self.assertGreater(metrics['batches_per_second'], 0)
    
    def test_validate_dataset_integrity(self):
        """Test dataset integrity validation."""
        # Create a dataset with valid data
        def generator():
            for i in range(10):
                img = tf.random.uniform((128, 128, 3), 0, 1)  # Valid range [0, 1]
                lbl = tf.random.uniform((128, 128, 2), -1, 1)  # Valid range [-1, 1]
                yield img, lbl
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(128, 128, 2), dtype=tf.float32)
            )
        ).batch(2)
        
        # Validate integrity
        results = validate_dataset_integrity(dataset, num_samples=5)
        
        # Check results
        self.assertGreater(results['valid_samples'], 0)
        self.assertEqual(results['invalid_samples'], 0)
        self.assertEqual(len(results['errors']), 0)
        self.assertGreater(len(results['shapes']), 0)
        self.assertGreater(len(results['dtypes']), 0)
        self.assertGreater(len(results['value_ranges']), 0)
    
    def test_validate_dataset_with_invalid_data(self):
        """Test dataset validation with invalid data."""
        # Create a dataset with some invalid data
        def generator():
            for i in range(5):
                if i == 2:
                    # Create invalid data with NaN
                    img = tf.constant(float('nan')) * tf.ones((128, 128, 3))
                    lbl = tf.random.uniform((128, 128, 2), -1, 1)
                else:
                    img = tf.random.uniform((128, 128, 3), 0, 1)
                    lbl = tf.random.uniform((128, 128, 2), -1, 1)
                yield img, lbl
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(128, 128, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(128, 128, 2), dtype=tf.float32)
            )
        ).batch(1)
        
        # Validate integrity
        results = validate_dataset_integrity(dataset, num_samples=5)
        
        # Check that invalid samples are detected
        self.assertGreater(results['invalid_samples'], 0)
        self.assertGreater(len(results['errors']), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)