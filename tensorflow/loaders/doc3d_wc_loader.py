"""
TensorFlow implementation of World Coordinate Data Loader for DewarpNet.
This module provides data loading functionality for world coordinate regression training.
"""

import os
import glob
import random
import collections
from os.path import join as pjoin
from typing import Tuple, List, Optional, Union

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .augmentations_tf import data_aug_tf, tight_crop_tf


class Doc3DWCLoader(Sequence):
    """
    TensorFlow data loader for world coordinate regression and RGB images.
    
    This loader replicates the functionality of the PyTorch doc3dwcLoader,
    providing identical preprocessing, normalization, and augmentation.
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        is_transform: bool = True,
        img_size: Union[int, Tuple[int, int]] = 512,
        augmentations: bool = False,
        batch_size: int = 1
    ):
        """
        Initialize the World Coordinate Data Loader.
        
        Args:
            root: Root directory path containing the dataset
            split: Dataset split ('train' or 'val')
            is_transform: Whether to apply transformations
            img_size: Target image size (int or tuple)
            augmentations: Whether to apply data augmentations
            batch_size: Batch size for data loading
        """
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 3
        self.batch_size = batch_size
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        # World coordinate normalization constants (from PyTorch implementation)
        self.xmx, self.xmn = 1.2539363, -1.2442188
        self.ymx, self.ymn = 1.2396319, -1.2289206
        self.zmx, self.zmn = 0.6436657, -0.67492497
        
        # Load file lists for each split
        for split_name in ['train', 'val']:
            path = pjoin(self.root, split_name + '.txt')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    file_list = [line.rstrip() for line in f.readlines()]
                self.files[split_name] = file_list
        
        # Load texture paths for augmentation
        if self.augmentations:
            self.txpths = []
            aug_tex_path = os.path.join(self.root[:-7], 'augtexnames.txt')
            if os.path.exists(aug_tex_path):
                with open(aug_tex_path, 'r') as f:
                    for line in f:
                        txpth = line.strip()
                        self.txpths.append(txpth)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files[self.split])
    
    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (image, world_coordinates) as TensorFlow tensors
        """
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, 'img', im_name + '.png')
        lbl_path = pjoin(self.root, 'wc', im_name + '.exr')
        
        # Load image and world coordinates
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        im = np.array(im, dtype=np.uint8)
        
        lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lbl = np.array(lbl, dtype=np.float32)
        
        # Apply tight cropping for validation
        if 'val' in self.split:
            im, lbl = tight_crop_tf(im / 255.0, lbl)
        
        # Apply augmentations for training
        if self.augmentations and len(self.txpths) > 0:
            tex_id = random.randint(0, len(self.txpths) - 1)
            txpth = self.txpths[tex_id]
            tex_path = os.path.join(self.root[:-7], txpth)
            
            if os.path.exists(tex_path):
                tex = cv2.imread(tex_path)
                tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
                tex = tex.astype(np.uint8)
                bg = cv2.resize(tex, self.img_size, interpolation=cv2.INTER_NEAREST)
                im, lbl = data_aug_tf(im, lbl, bg)
        
        # Apply transformations
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        
        return im, lbl
    
    def transform(self, img: np.ndarray, lbl: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply transformations to image and label.
        
        Args:
            img: Input image array
            lbl: World coordinate label array
            
        Returns:
            Tuple of transformed (image, label) tensors
        """
        # Resize image
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Handle RGBA images by discarding alpha channel
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        
        # Normalize image to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Process world coordinates
        lbl = lbl.astype(np.float32)
        
        # Create mask for valid coordinates
        msk = ((lbl[:, :, 0] != 0) & (lbl[:, :, 1] != 0) & (lbl[:, :, 2] != 0)).astype(np.uint8) * 255
        
        # Normalize world coordinates using dataset statistics
        lbl[:, :, 0] = (lbl[:, :, 0] - self.zmn) / (self.zmx - self.zmn)  # Z coordinate
        lbl[:, :, 1] = (lbl[:, :, 1] - self.ymn) / (self.ymx - self.ymn)  # Y coordinate
        lbl[:, :, 2] = (lbl[:, :, 2] - self.xmn) / (self.xmx - self.xmn)  # X coordinate
        
        # Apply mask to coordinates
        lbl = cv2.bitwise_and(lbl, lbl, mask=msk)
        
        # Resize label to match image size
        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to TensorFlow tensors
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        lbl = tf.convert_to_tensor(lbl, dtype=tf.float32)
        
        return img, lbl
    
    def on_epoch_end(self):
        """Called at the end of each epoch (for compatibility with Keras)."""
        pass


def create_wc_dataset(
    root: str,
    split: str = 'train',
    batch_size: int = 4,
    img_size: Union[int, Tuple[int, int]] = 512,
    augmentations: bool = False,
    shuffle: bool = True,
    prefetch_buffer: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """
    Create a TensorFlow Dataset for world coordinate training.
    
    Args:
        root: Root directory path containing the dataset
        split: Dataset split ('train' or 'val')
        batch_size: Batch size for training
        img_size: Target image size
        augmentations: Whether to apply data augmentations
        shuffle: Whether to shuffle the dataset
        prefetch_buffer: Prefetch buffer size
        num_parallel_calls: Number of parallel calls for data processing
        
    Returns:
        tf.data.Dataset configured for world coordinate training
    """
    loader = Doc3DWCLoader(
        root=root,
        split=split,
        img_size=img_size,
        augmentations=augmentations,
        batch_size=batch_size
    )
    
    # Create dataset from generator
    def generator():
        for i in range(len(loader)):
            yield loader[i]
    
    # Determine output signature
    sample_img, sample_lbl = loader[0]
    output_signature = (
        tf.TensorSpec(shape=sample_img.shape, dtype=tf.float32),
        tf.TensorSpec(shape=sample_lbl.shape, dtype=tf.float32)
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