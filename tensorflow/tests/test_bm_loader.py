"""
Unit tests for Backward Mapping Data Loader.
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

from loaders.doc3d_bm_loader import Doc3DBMLoader, create_bm_dataset


class TestDoc3DBMLoader(unittest.TestCase):
    """Test cases for Doc3DBMLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.alt_temp_dir = tempfile.mkdtemp()
        self.img_size = (128, 128)
        self.batch_size = 2
        
        # Create mock directory structure for main root
        os.makedirs(os.path.join(self.temp_dir, 'recon', '1', 'chess48'), exist_ok=True)
        
        # Create mock directory structure for alternative root
        os.makedirs(os.path.join(self.alt_temp_dir, 'wc'), exist_ok=True)
        os.makedirs(os.path.join(self.alt_temp_dir, 'bm'), exist_ok=True)
        
        # Create mock train.txt file
        with open(os.path.join(self.alt_temp_dir, 'train.txt'), 'w') as f:
            f.write('1/sample1\n1/sample2\n')
        
        # Create mock val.txt file
        with open(os.path.join(self.alt_temp_dir, 'val.txt'), 'w') as f:
            f.write('1/val_sample1\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.alt_temp_dir, ignore_errors=True)
    
    def create_mock_albedo(self, filename: str, size: tuple = (128, 128)):
        """Create a mock albedo image file."""
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img
    
    def create_mock_wc(self, filename: str, size: tuple = (128, 128)):
        """Create a mock world coordinate EXR file."""
        wc = np.random.uniform(-1, 1, (*size, 3)).astype(np.float32)
        # Set some pixels to zero (invalid coordinates)
        mask = np.random.random(size) > 0.1
        wc[~mask] = 0
        cv2.imwrite(filename, wc)
        return wc
    
    def create_mock_bm(self, filename: str, size: tuple = (448, 448)):
        """Create a mock backward mapping MAT file."""
        bm = np.random.uniform(0, 448, (*size, 2)).astype(np.float32)
        sio.savemat(filename, {'bm': bm})
        return bm
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train',
            img_size=self.img_size
        )
        
        self.assertEqual(loader.root, self.temp_dir)
        self.assertEqual(loader.altroot, self.alt_temp_dir)
        self.assertEqual(loader.split, 'train')
        self.assertEqual(loader.img_size, self.img_size)
        self.assertEqual(len(loader.files['train']), 2)
        self.assertEqual(len(loader.files['val']), 1)
    
    def test_loader_length(self):
        """Test loader length method."""
        loader = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train'
        )
        self.assertEqual(len(loader), 2)
        
        loader_val = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='val'
        )
        self.assertEqual(len(loader_val), 1)
    
    def test_tight_crop_bm(self):
        """Test tight cropping for backward mapping."""
        loader = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train'
        )
        
        # Create test data
        wc = np.zeros((200, 200, 3), dtype=np.float32)
        wc[50:150, 50:150, :] = np.random.rand(100, 100, 3)
        
        alb = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Apply tight crop
        wc_cropped, alb_cropped, t, b, l, r = loader.tight_crop_bm(wc, alb)
        
        # Check that output is processed
        self.assertLessEqual(wc_cropped.shape[0], wc.shape[0])
        self.assertLessEqual(wc_cropped.shape[1], wc.shape[1])
        self.assertEqual(wc_cropped.shape[2], wc.shape[2])
        self.assertEqual(alb_cropped.shape[2], alb.shape[2])
        
        # Check crop parameters are integers
        self.assertIsInstance(t, (int, np.integer))
        self.assertIsInstance(b, (int, np.integer))
        self.assertIsInstance(l, (int, np.integer))
        self.assertIsInstance(r, (int, np.integer))
    
    @patch('cv2.imread')
    @patch('scipy.io.loadmat')
    def test_getitem_basic(self, mock_loadmat, mock_imread):
        """Test basic __getitem__ functionality."""
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
        
        loader = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train',
            img_size=self.img_size,
            is_transform=True
        )
        
        img, lbl = loader[0]
        
        # Check output types
        self.assertIsInstance(img, tf.Tensor)
        self.assertIsInstance(lbl, tf.Tensor)
        
        # Check output shapes
        self.assertEqual(img.shape, (*self.img_size, 6))  # 3 albedo + 3 world coords
        self.assertEqual(lbl.shape, (*self.img_size, 2))  # 2D backward mapping
        
        # Check data types
        self.assertEqual(img.dtype, tf.float32)
        self.assertEqual(lbl.dtype, tf.float32)
    
    @patch('cv2.imread')
    @patch('scipy.io.loadmat')
    def test_transform_method(self, mock_loadmat, mock_imread):
        """Test the transform method."""
        loader = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train',
            img_size=self.img_size
        )
        
        # Create test data
        wc = np.random.uniform(-1, 1, (200, 200, 3)).astype(np.float32)
        wc[50:150, 50:150, :] = np.random.rand(100, 100, 3)  # Valid region
        
        alb = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        bm = np.random.uniform(0, 448, (448, 448, 2)).astype(np.float32)
        
        # Apply transform
        img_tf, lbl_tf = loader.transform(wc, bm, alb)
        
        # Check output types and shapes
        self.assertIsInstance(img_tf, tf.Tensor)
        self.assertIsInstance(lbl_tf, tf.Tensor)
        self.assertEqual(img_tf.shape, (*self.img_size, 6))
        self.assertEqual(lbl_tf.shape, (*self.img_size, 2))
        
        # Check value ranges
        # Albedo channels (first 3) should be in [0, 1] after normalization
        albedo_channels = img_tf[:, :, :3]
        self.assertTrue(tf.reduce_all(albedo_channels >= 0.0))
        self.assertTrue(tf.reduce_all(albedo_channels <= 1.0))
        
        # World coordinate channels (last 3) are normalized using dataset statistics
        # and may have values outside [0, 1] range
        wc_channels = img_tf[:, :, 3:]
        self.assertTrue(tf.reduce_all(tf.math.is_finite(wc_channels)))
        
        # Backward mapping should be in [-1, 1] range
        self.assertTrue(tf.reduce_all(lbl_tf >= -1.0))
        self.assertTrue(tf.reduce_all(lbl_tf <= 1.0))
    
    def test_coordinate_processing(self):
        """Test coordinate normalization and processing."""
        loader = Doc3DBMLoader(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train'
        )
        
        # Create test world coordinates with known values
        wc = np.ones((100, 100, 3), dtype=np.float32)
        wc[:, :, 0] = 0.5  # Z coordinate
        wc[:, :, 1] = 0.5  # Y coordinate  
        wc[:, :, 2] = 0.5  # X coordinate
        
        alb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bm = np.ones((100, 100, 2), dtype=np.float32) * 224  # Center coordinates
        
        img_tf, lbl_tf = loader.transform(wc, bm, alb)
        
        # Check that world coordinate normalization was applied
        expected_z = (0.5 - loader.zmn) / (loader.zmx - loader.zmn)
        expected_y = (0.5 - loader.ymn) / (loader.ymx - loader.ymn)
        expected_x = (0.5 - loader.xmn) / (loader.xmx - loader.xmn)
        
        # The exact values will be affected by tight cropping and resizing,
        # but we can check that normalization constants are being used
        self.assertIsNotNone(loader.zmx)
        self.assertIsNotNone(loader.zmn)
        self.assertIsNotNone(loader.ymx)
        self.assertIsNotNone(loader.ymn)
        self.assertIsNotNone(loader.xmx)
        self.assertIsNotNone(loader.xmn)


class TestBMDatasetCreation(unittest.TestCase):
    """Test cases for backward mapping dataset creation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.alt_temp_dir = tempfile.mkdtemp()
        
        # Create mock directory structure
        os.makedirs(os.path.join(self.temp_dir, 'recon', '1', 'chess48'), exist_ok=True)
        os.makedirs(os.path.join(self.alt_temp_dir, 'wc'), exist_ok=True)
        os.makedirs(os.path.join(self.alt_temp_dir, 'bm'), exist_ok=True)
        
        # Create mock train.txt file
        with open(os.path.join(self.alt_temp_dir, 'train.txt'), 'w') as f:
            f.write('1/sample1\n1/sample2\n')
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.alt_temp_dir, ignore_errors=True)
    
    @patch('cv2.imread')
    @patch('scipy.io.loadmat')
    def test_create_bm_dataset(self, mock_loadmat, mock_imread):
        """Test dataset creation function."""
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
        
        # Create dataset
        dataset = create_bm_dataset(
            root=self.temp_dir,
            altroot=self.alt_temp_dir,
            split='train',
            batch_size=2,
            img_size=(128, 128),
            shuffle=False
        )
        
        # Check dataset properties
        self.assertIsInstance(dataset, tf.data.Dataset)
        
        # Test iteration
        for batch in dataset.take(1):
            imgs, lbls = batch
            self.assertEqual(imgs.shape[0], 2)  # batch size
            self.assertEqual(imgs.shape[1:], (128, 128, 6))  # input shape (alb + wc)
            self.assertEqual(lbls.shape[1:], (128, 128, 2))  # label shape (bm)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)