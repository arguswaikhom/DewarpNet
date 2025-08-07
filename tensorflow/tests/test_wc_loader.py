"""
Unit tests for World Coordinate Data Loader.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loaders.doc3d_wc_loader import Doc3DWCLoader, create_wc_dataset
from loaders.augmentations_tf import tight_crop_tf, data_aug_tf, color_jitter_tf


class TestDoc3DWCLoader(unittest.TestCase):
    """Test cases for Doc3DWCLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.img_size = (256, 256)
        self.batch_size = 2
        
        # Create mock directory structure
        os.makedirs(os.path.join(self.temp_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'wc'), exist_ok=True)
        
        # Create mock train.txt file
        with open(os.path.join(self.temp_dir, 'train.txt'), 'w') as f:
            f.write('sample1\nsample2\nsample3\n')
        
        # Create mock val.txt file
        with open(os.path.join(self.temp_dir, 'val.txt'), 'w') as f:
            f.write('val_sample1\nval_sample2\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_image(self, filename: str, size: tuple = (512, 512)):
        """Create a mock RGB image file."""
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img
    
    def create_mock_exr(self, filename: str, size: tuple = (512, 512)):
        """Create a mock EXR world coordinate file."""
        # Create mock world coordinates
        wc = np.random.uniform(-1, 1, (*size, 3)).astype(np.float32)
        # Set some pixels to zero (invalid coordinates)
        mask = np.random.random(size) > 0.1
        wc[~mask] = 0
        
        # Save as EXR (OpenCV can handle this)
        cv2.imwrite(filename, wc)
        return wc
    
    @patch('cv2.imread')
    def test_loader_initialization(self, mock_imread):
        """Test loader initialization."""
        loader = Doc3DWCLoader(
            root=self.temp_dir,
            split='train',
            img_size=self.img_size,
            augmentations=False
        )
        
        self.assertEqual(loader.root, self.temp_dir)
        self.assertEqual(loader.split, 'train')
        self.assertEqual(loader.img_size, self.img_size)
        self.assertEqual(len(loader.files['train']), 3)
        self.assertEqual(len(loader.files['val']), 2)
    
    @patch('cv2.imread')
    def test_loader_length(self, mock_imread):
        """Test loader length method."""
        loader = Doc3DWCLoader(root=self.temp_dir, split='train')
        self.assertEqual(len(loader), 3)
        
        loader_val = Doc3DWCLoader(root=self.temp_dir, split='val')
        self.assertEqual(len(loader_val), 2)
    
    @patch('cv2.imread')
    def test_getitem_basic(self, mock_imread):
        """Test basic __getitem__ functionality."""
        # Mock image and EXR loading
        mock_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mock_wc = np.random.uniform(-1, 1, (512, 512, 3)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.png'):
                return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
            elif path.endswith('.exr'):
                return mock_wc
            return None
        
        mock_imread.side_effect = imread_side_effect
        
        loader = Doc3DWCLoader(
            root=self.temp_dir,
            split='train',
            img_size=self.img_size,
            is_transform=True
        )
        
        img, lbl = loader[0]
        
        # Check output types
        self.assertIsInstance(img, tf.Tensor)
        self.assertIsInstance(lbl, tf.Tensor)
        
        # Check output shapes
        self.assertEqual(img.shape, (*self.img_size, 3))
        self.assertEqual(lbl.shape, (*self.img_size, 3))
        
        # Check data types
        self.assertEqual(img.dtype, tf.float32)
        self.assertEqual(lbl.dtype, tf.float32)
    
    def test_transform_method(self):
        """Test the transform method."""
        loader = Doc3DWCLoader(root=self.temp_dir, split='train', img_size=self.img_size)
        
        # Create test data
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        lbl = np.random.uniform(-1, 1, (512, 512, 3)).astype(np.float32)
        
        # Apply transform
        img_tf, lbl_tf = loader.transform(img, lbl)
        
        # Check output types and shapes
        self.assertIsInstance(img_tf, tf.Tensor)
        self.assertIsInstance(lbl_tf, tf.Tensor)
        self.assertEqual(img_tf.shape, (*self.img_size, 3))
        self.assertEqual(lbl_tf.shape, (*self.img_size, 3))
        
        # Check normalization (image should be in [0, 1])
        self.assertTrue(tf.reduce_all(img_tf >= 0.0))
        self.assertTrue(tf.reduce_all(img_tf <= 1.0))
    
    def test_coordinate_normalization(self):
        """Test world coordinate normalization."""
        loader = Doc3DWCLoader(root=self.temp_dir, split='train')
        
        # Create test coordinates with known values
        lbl = np.ones((100, 100, 3), dtype=np.float32)
        lbl[:, :, 0] = 0.5  # Z coordinate
        lbl[:, :, 1] = 0.5  # Y coordinate  
        lbl[:, :, 2] = 0.5  # X coordinate
        
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        _, lbl_normalized = loader.transform(img, lbl)
        
        # Check that normalization was applied
        # Values should be normalized using the dataset statistics
        expected_z = (0.5 - loader.zmn) / (loader.zmx - loader.zmn)
        expected_y = (0.5 - loader.ymn) / (loader.ymx - loader.ymn)
        expected_x = (0.5 - loader.xmn) / (loader.xmx - loader.xmn)
        
        # Allow for small floating point differences
        self.assertAlmostEqual(float(lbl_normalized[50, 50, 0]), expected_z, places=5)
        self.assertAlmostEqual(float(lbl_normalized[50, 50, 1]), expected_y, places=5)
        self.assertAlmostEqual(float(lbl_normalized[50, 50, 2]), expected_x, places=5)


class TestAugmentations(unittest.TestCase):
    """Test cases for augmentation functions."""
    
    def test_tight_crop_tf(self):
        """Test tight cropping function."""
        # Create test image and coordinates
        img = np.random.rand(200, 200, 3).astype(np.float32)
        coords = np.zeros((200, 200, 3), dtype=np.float32)
        
        # Set valid coordinates in a central region
        coords[50:150, 50:150, :] = np.random.rand(100, 100, 3)
        
        # Apply tight crop
        img_cropped, coords_cropped = tight_crop_tf(img, coords)
        
        # Check that output is smaller than input
        self.assertLessEqual(img_cropped.shape[0], img.shape[0])
        self.assertLessEqual(img_cropped.shape[1], img.shape[1])
        self.assertEqual(img_cropped.shape[2], img.shape[2])
        self.assertEqual(coords_cropped.shape[2], coords.shape[2])
    
    def test_color_jitter_tf(self):
        """Test color jittering function."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        
        # Apply color jitter
        img_jittered = color_jitter_tf(img, brightness=0.2, contrast=0.2)
        
        # Check output shape and range
        self.assertEqual(img_jittered.shape, img.shape)
        self.assertTrue(np.all(img_jittered >= 0.0))
        self.assertTrue(np.all(img_jittered <= 1.0))
    
    def test_data_aug_tf(self):
        """Test complete data augmentation pipeline."""
        # Create test data
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        coords = np.zeros((200, 200, 3), dtype=np.float32)
        coords[50:150, 50:150, :] = np.random.rand(100, 100, 3)
        bg = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Apply augmentation
        img_aug, coords_aug = data_aug_tf(img, coords, bg)
        
        # Check output properties
        self.assertEqual(len(img_aug.shape), 3)
        self.assertEqual(len(coords_aug.shape), 3)
        self.assertEqual(img_aug.shape[2], 3)
        self.assertEqual(coords_aug.shape[2], 3)
        
        # Check value ranges
        self.assertTrue(np.all(img_aug >= 0.0))
        self.assertTrue(np.all(img_aug <= 1.0))


class TestDatasetCreation(unittest.TestCase):
    """Test cases for dataset creation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock directory structure
        os.makedirs(os.path.join(self.temp_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'wc'), exist_ok=True)
        
        # Create mock train.txt file
        with open(os.path.join(self.temp_dir, 'train.txt'), 'w') as f:
            f.write('sample1\nsample2\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cv2.imread')
    def test_create_wc_dataset(self, mock_imread):
        """Test dataset creation function."""
        # Mock image and EXR loading
        mock_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mock_wc = np.random.uniform(-1, 1, (256, 256, 3)).astype(np.float32)
        
        def imread_side_effect(path, flags=None):
            if path.endswith('.png'):
                return cv2.cvtColor(mock_img, cv2.COLOR_RGB2BGR)
            elif path.endswith('.exr'):
                return mock_wc
            return None
        
        mock_imread.side_effect = imread_side_effect
        
        # Create dataset
        dataset = create_wc_dataset(
            root=self.temp_dir,
            split='train',
            batch_size=2,
            img_size=(256, 256),
            shuffle=False
        )
        
        # Check dataset properties
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Test iteration
        for batch in dataset.take(1):
            imgs, lbls = batch
            self.assertEqual(imgs.shape[0], 2)  # batch size
            self.assertEqual(imgs.shape[1:], (256, 256, 3))  # image shape
            self.assertEqual(lbls.shape[1:], (256, 256, 3))  # label shape


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)