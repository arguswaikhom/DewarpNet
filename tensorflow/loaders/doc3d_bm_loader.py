"""
TensorFlow implementation of Backward Mapping Data Loader for DewarpNet.
This module provides data loading functionality for backward mapping training.
"""

import os
import random
import collections
from os.path import join as pjoin
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import scipy.io as sio
import hdf5storage as h5

from .augmentations_tf import tight_crop_tf


class Doc3DBMLoader(Sequence):
    """
    TensorFlow data loader for backward mapping training data.
    
    This loader replicates the functionality of the PyTorch doc3dbmnoimgcLoader,
    loading albedo images, world coordinates, and backward mapping coordinates.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        is_transform: bool = True,
        img_size: Union[int, Tuple[int, int]] = 512,
        batch_size: int = 1,
        altroot: Optional[str] = None
    ):
        """
        Initialize the Backward Mapping Data Loader.
        
        Args:
            root: Root directory path containing the dataset
            split: Dataset split ('train' or 'val')
            is_transform: Whether to apply transformations
            img_size: Target image size (int or tuple)
            batch_size: Batch size for data loading
            altroot: Alternative root path for world coordinate and backward mapping data
        """
        self.root = os.path.expanduser(root)
        
        # Set alternative root path (for world coordinates and backward mapping)
        if altroot is None:
            # Default path structure from PyTorch implementation
            self.altroot = '/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/swat3d/'
        else:
            self.altroot = altroot
            
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.batch_size = batch_size
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        # World coordinate normalization constants (same as WC loader)
        self.xmx, self.xmn = 1.2539363, -1.2442188
        self.ymx, self.ymn = 1.2396319, -1.2289206
        self.zmx, self.zmn = 0.6436657, -0.67492497
        
        # Load file lists for each split
        for split_name in ['train', 'val']:
            path = pjoin(self.altroot, split_name + '.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    file_list = [line.rstrip() for line in f.readlines()]
                self.files[split_name] = file_list
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files[self.split])
    
    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (concatenated_input, backward_mapping) as TensorFlow tensors
        """
        im_name = self.files[self.split][index]  # e.g., "1/2Xec_Page_453X56X0001"
        
        # Parse folder and filename
        img_foldr, fname = im_name.split('/')
        recon_foldr = 'chess48'
        
        # Construct file paths
        wc_path = pjoin(self.altroot, 'wc', im_name + '.exr')
        bm_path = pjoin(self.altroot, 'bm', im_name + '.mat')
        alb_path = pjoin(self.root, 'recon', img_foldr, recon_foldr, 
                        fname[:-4] + recon_foldr + '0001.png')
        
        # Load data
        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        wc = np.array(wc, dtype=np.float32)
        
        # Load backward mapping coordinates
        try:
            bm_data = h5.loadmat(bm_path)
            bm = bm_data['bm']
        except:
            # Fallback to scipy.io if hdf5storage fails
            bm_data = sio.loadmat(bm_path)
            bm = bm_data['bm']
        
        bm = np.array(bm, dtype=np.float32)
        
        # Load albedo image
        alb = cv2.imread(alb_path, cv2.IMREAD_COLOR)
        alb = cv2.cvtColor(alb, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        alb = np.array(alb, dtype=np.uint8)
        
        # Apply transformations
        if self.is_transform:
            im, lbl = self.transform(wc, bm, alb)
        else:
            # Convert to tensors without transformation
            im = tf.convert_to_tensor(np.concatenate([alb, wc], axis=-1), dtype=tf.float32)
            lbl = tf.convert_to_tensor(bm, dtype=tf.float32)
        
        return im, lbl
    
    def tight_crop_bm(self, wc: np.ndarray, alb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
        """
        Apply tight cropping for backward mapping data.
        
        Args:
            wc: World coordinate array (H, W, 3)
            alb: Albedo image array (H, W, 3)
            
        Returns:
            Tuple of (cropped_wc, cropped_alb, top, bottom, left, right)
        """
        # Create mask for valid coordinates
        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8)
        size = msk.shape
        
        # Find bounding box
        y_coords, x_coords = np.nonzero(msk)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            # If no valid coordinates, return original with zero padding info
            return wc, alb, 0, 0, 0, 0
        
        minx, maxx = min(x_coords), max(x_coords)
        miny, maxy = min(y_coords), max(y_coords)
        
        # Crop to bounding box
        wc = wc[miny:maxy + 1, minx:maxx + 1, :]
        alb = alb[miny:maxy + 1, minx:maxx + 1, :]
        
        # Add padding for random cropping
        s = 20
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), 'constant')
        
        # Random crop
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1
        
        wc = wc[cy1:-cy2, cx1:-cx2, :]
        alb = alb[cy1:-cy2, cx1:-cx2, :]
        
        # Calculate crop parameters for backward mapping coordinate adjustment
        t = miny - s + cy1  # top
        b = size[0] - maxy - s + cy2  # bottom
        l = minx - s + cx1  # left
        r = size[1] - maxx - s + cx2  # right
        
        return wc, alb, t, b, l, r
    
    def transform(self, wc: np.ndarray, bm: np.ndarray, alb: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply transformations to world coordinates, backward mapping, and albedo.
        
        Args:
            wc: World coordinate array
            bm: Backward mapping coordinate array
            alb: Albedo image array
            
        Returns:
            Tuple of transformed (concatenated_input, backward_mapping) tensors
        """
        # Apply tight cropping
        wc, alb, t, b, l, r = self.tight_crop_bm(wc, alb)
        
        # Process albedo image
        alb = cv2.resize(alb, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Handle RGBA images by discarding alpha channel
        if alb.shape[-1] == 4:
            alb = alb[:, :, :3]
        
        # Normalize albedo to [0, 1]
        alb = alb.astype(np.float32) / 255.0
        
        # Process world coordinates
        wc = wc.astype(np.float32)
        
        # Create mask for valid coordinates
        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(np.uint8) * 255
        
        # Normalize world coordinates using dataset statistics
        wc[:, :, 0] = (wc[:, :, 0] - self.zmn) / (self.zmx - self.zmn)  # Z coordinate
        wc[:, :, 1] = (wc[:, :, 1] - self.ymn) / (self.ymx - self.ymn)  # Y coordinate
        wc[:, :, 2] = (wc[:, :, 2] - self.xmn) / (self.xmx - self.xmn)  # X coordinate
        
        # Apply mask to coordinates
        wc = cv2.bitwise_and(wc, wc, mask=msk)
        
        # Resize world coordinates
        wc = cv2.resize(wc, self.img_size, interpolation=cv2.INTER_NEAREST)
        wc = wc.astype(np.float32)
        
        # Process backward mapping coordinates
        bm = bm.astype(np.float32)
        
        # Adjust coordinates based on cropping
        bm[:, :, 1] = bm[:, :, 1] - t  # Adjust Y coordinates
        bm[:, :, 0] = bm[:, :, 0] - l  # Adjust X coordinates
        
        # Normalize to image dimensions after cropping
        bm = bm / np.array([448.0 - l - r, 448.0 - t - b])
        
        # Normalize to [-1, 1] range
        bm = (bm - 0.5) * 2
        
        # Resize backward mapping coordinates
        bm0 = cv2.resize(bm[:, :, 0], self.img_size, interpolation=cv2.INTER_LINEAR)
        bm1 = cv2.resize(bm[:, :, 1], self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Concatenate albedo and world coordinates for input
        img = np.concatenate([alb, wc], axis=-1)  # Shape: (H, W, 6)
        
        # Stack backward mapping coordinates for label
        lbl = np.stack([bm0, bm1], axis=-1)  # Shape: (H, W, 2)
        
        # Convert to TensorFlow tensors
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        lbl = tf.convert_to_tensor(lbl, dtype=tf.float32)
        
        return img, lbl
    
    def on_epoch_end(self):
        """Called at the end of each epoch (for compatibility with Keras)."""
        pass


def create_bm_dataset(
    root: str,
    split: str = 'train',
    batch_size: int = 4,
    img_size: Union[int, Tuple[int, int]] = 512,
    shuffle: bool = True,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    altroot: Optional[str] = None
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset for backward mapping training.
    
    Args:
        root: Root directory path containing the dataset
        split: Dataset split ('train' or 'val')
        batch_size: Batch size for training
        img_size: Target image size
        shuffle: Whether to shuffle the dataset
        prefetch_buffer: Prefetch buffer size
        num_parallel_calls: Number of parallel calls for data processing
        altroot: Alternative root path for world coordinate and backward mapping data
        
    Returns:
        tf.data.Dataset configured for backward mapping training
    """
    loader = Doc3DBMLoader(
        root=root,
        split=split,
        img_size=img_size,
        batch_size=batch_size,
        altroot=altroot
    )
    
    # Create dataset from generator
    def generator():
        for i in range(len(loader)):
            yield loader[i]
    
    # Determine output signature
    try:
        sample_img, sample_lbl = loader[0]
        output_signature = (
            tf.TensorSpec(shape=sample_img.shape, dtype=tf.float32),
            tf.TensorSpec(shape=sample_lbl.shape, dtype=tf.float32)
        )
    except (IndexError, FileNotFoundError):
        # Fallback signature if no data available
        output_signature = (
            tf.TensorSpec(shape=(img_size[0], img_size[1], 6), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size[0], img_size[1], 2), dtype=tf.float32)
        )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(1000, len(loader)))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    
    return dataset