"""
TensorFlow implementation of data augmentations for DewarpNet.
This module provides augmentation functions that replicate the PyTorch implementation.
"""

import random
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf


def tight_crop_tf(im: np.ndarray, fm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply tight cropping to image and world coordinates.
    
    This function replicates the tight_crop function from the PyTorch implementation,
    cropping the image to the valid coordinate region with random padding.
    
    Args:
        im: Input image array (H, W, C)
        fm: World coordinate array (H, W, 3)
        
    Returns:
        Tuple of cropped (image, coordinates)
    """
    # Create mask for valid coordinates
    msk = ((fm[:, :, 0] != 0) & (fm[:, :, 1] != 0) & (fm[:, :, 2] != 0)).astype(np.uint8)
    
    # Find bounding box of valid coordinates
    y_coords, x_coords = np.nonzero(msk)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        # If no valid coordinates, return original
        return im, fm
    
    minx, maxx = min(x_coords), max(x_coords)
    miny, maxy = min(y_coords), max(y_coords)
    
    # Crop to bounding box
    im = im[miny:maxy + 1, minx:maxx + 1, :]
    fm = fm[miny:maxy + 1, minx:maxx + 1, :]
    
    # Add padding for random cropping
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    fm = np.pad(fm, ((s, s), (s, s), (0, 0)), 'constant')
    
    # Random crop
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1
    
    im = im[cy1:-cy2, cx1:-cx2, :]
    fm = fm[cy1:-cy2, cx1:-cx2, :]
    
    return im, fm


def tight_crop_d_tf(im: np.ndarray, dm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply tight cropping for depth/displacement maps.
    
    Args:
        im: Input image array (H, W, C)
        dm: Depth/displacement map (H, W)
        
    Returns:
        Tuple of cropped (image, depth_map)
    """
    # Create mask for valid depth values
    msk = (dm != 0).astype(np.uint8)
    
    # Find bounding box
    y_coords, x_coords = np.nonzero(msk)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return im, dm
    
    minx, maxx = min(x_coords), max(x_coords)
    miny, maxy = min(y_coords), max(y_coords)
    
    # Crop to bounding box
    im = im[miny:maxy + 1, minx:maxx + 1, :]
    dm = dm[miny:maxy + 1, minx:maxx + 1]
    
    # Add padding
    s = 20
    im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
    dm = np.pad(dm, ((s, s), (s, s)), 'constant')
    
    # Random crop
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1
    
    im = im[cy1:-cy2, cx1:-cx2, :]
    dm = dm[cy1:-cy2, cx1:-cx2]
    
    return im, dm


def color_jitter_tf(
    im: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0
) -> np.ndarray:
    """
    Apply color jittering to image.
    
    Args:
        im: Input image array (H, W, C) in range [0, 1]
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range (unused in original)
        hue: Hue jitter range (unused in original)
        
    Returns:
        Color jittered image
    """
    # Apply contrast jitter
    if contrast > 0:
        f = random.uniform(1 - contrast, 1 + contrast)
        im = np.clip(im * f, 0.0, 1.0)
    
    # Apply brightness jitter
    if brightness > 0:
        f = random.uniform(-brightness, brightness)
        im = np.clip(im + f, 0.0, 1.0).astype(np.float32)
    
    return im


def change_intensity_tf(img: np.ndarray) -> np.ndarray:
    """
    Change image intensity (brightness) in HSV space.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Image with modified intensity
    """
    chance = random.uniform(0, 1)
    
    # Convert to HSV
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if chance > 0.3:
        inc = random.randint(15, 50)
        # Increase value (brightness) channel
        v = nimg[:, :, 2]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 2] = v
    
    # Convert back to BGR
    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    return nimg


def change_hue_sat_tf(img: np.ndarray) -> np.ndarray:
    """
    Change image hue and saturation.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Image with modified hue and saturation
    """
    chance = random.uniform(0, 1)
    
    # Convert to HSV
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Modify hue
    if chance > 0.3:
        inc = random.randint(5, 15)
        v = nimg[:, :, 0]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 0] = v
    
    # Modify saturation
    if chance > 0.3:
        inc = random.randint(5, 15)
        v = nimg[:, :, 1]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 1] = v
    
    # Convert back to BGR
    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    return nimg


def data_aug_tf(
    im: np.ndarray,
    fm: Union[np.ndarray, np.ndarray],
    bg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation pipeline.
    
    This function replicates the data_aug function from the PyTorch implementation,
    including background replacement, tight cropping, and color jittering.
    
    Args:
        im: Input image array
        fm: World coordinates or depth map
        bg: Background texture image
        
    Returns:
        Tuple of augmented (image, coordinates/depth)
    """
    # Normalize input image
    im = im / 255.0
    bg = bg / 255.0
    
    # Apply tight cropping based on coordinate/depth map dimensions
    if len(fm.shape) == 3 and fm.shape[-1] == 3:
        # World coordinates (3 channels)
        im, fm = tight_crop_tf(im, fm)
        # Create mask for valid coordinates
        msk = ((fm[:, :, 0] != 0) & (fm[:, :, 1] != 0) & (fm[:, :, 2] != 0)).astype(np.uint8)
    else:
        # Depth map (single channel)
        im, fm = tight_crop_d_tf(im, fm)
        # Create mask for valid depth values
        msk = (fm != 0).astype(np.uint8)
    
    # Expand mask dimensions for broadcasting
    msk = np.expand_dims(msk, axis=2)
    
    # Get image dimensions
    fh, fw, _ = im.shape
    
    # Background replacement logic
    chance = random.random()
    
    if chance > 0.3:
        # Tile background texture
        bg = cv2.resize(bg, (200, 200))
        bg = np.tile(bg, (3, 3, 1))
        bg = bg[:fh, :fw, :]
    elif 0.2 < chance <= 0.3:
        # Random solid color background
        c = np.array([random.random(), random.random(), random.random()])
        bg = np.ones((fh, fw, 3)) * c
    else:
        # Black background, keep original image
        bg = np.zeros((fh, fw, 3))
        msk = np.ones((fh, fw, 3))
    
    # Composite image with background
    im = bg * (1 - msk) + im * msk
    
    # Apply color jittering
    im = color_jitter_tf(im, brightness=0.2, contrast=0.2, saturation=0.6, hue=0.6)
    
    return im, fm


# TensorFlow-specific augmentation functions using tf.image
def tf_color_jitter(
    image: tf.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.6,
    hue: float = 0.6
) -> tf.Tensor:
    """
    Apply color jittering using TensorFlow operations.
    
    Args:
        image: Input image tensor
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range
        hue: Hue jitter range
        
    Returns:
        Color jittered image tensor
    """
    # Apply random brightness
    if brightness > 0:
        image = tf.image.random_brightness(image, max_delta=brightness)
    
    # Apply random contrast
    if contrast > 0:
        image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    
    # Apply random saturation
    if saturation > 0:
        image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    
    # Apply random hue
    if hue > 0:
        image = tf.image.random_hue(image, max_delta=hue)
    
    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def tf_random_crop_and_resize(
    image: tf.Tensor,
    label: tf.Tensor,
    target_size: Tuple[int, int],
    crop_ratio_range: Tuple[float, float] = (0.8, 1.0)
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply random crop and resize using TensorFlow operations.
    
    Args:
        image: Input image tensor
        label: Input label tensor
        target_size: Target output size (height, width)
        crop_ratio_range: Range of crop ratios
        
    Returns:
        Tuple of cropped and resized (image, label)
    """
    # Get image shape
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    
    # Random crop ratio
    crop_ratio = tf.random.uniform([], crop_ratio_range[0], crop_ratio_range[1])
    
    # Calculate crop size
    crop_height = tf.cast(tf.cast(height, tf.float32) * crop_ratio, tf.int32)
    crop_width = tf.cast(tf.cast(width, tf.float32) * crop_ratio, tf.int32)
    
    # Random crop offset
    offset_height = tf.random.uniform([], 0, height - crop_height + 1, dtype=tf.int32)
    offset_width = tf.random.uniform([], 0, width - crop_width + 1, dtype=tf.int32)
    
    # Crop image and label
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    label = tf.image.crop_to_bounding_box(label, offset_height, offset_width, crop_height, crop_width)
    
    # Resize to target size
    image = tf.image.resize(image, target_size, method='bilinear')
    label = tf.image.resize(label, target_size, method='nearest')
    
    return image, label