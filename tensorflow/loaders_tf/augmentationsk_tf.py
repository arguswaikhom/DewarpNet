"""
TensorFlow implementation of data augmentations for DewarpNet.
Equivalent to PyTorch augmentationsk.py
"""

import numpy as np
import cv2
import random
from typing import Tuple


def tight_crop_tf(im: np.ndarray, fm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    TensorFlow version of tight cropping.
    
    Args:
        im: Input image
        fm: Flow map or world coordinates
        
    Returns:
        Cropped image and flow map
    """
    # Create mask for non-zero coordinates
    if fm.shape[-1] == 3:
        msk = ((fm[:, :, 0] != 0) & (fm[:, :, 1] != 0) & (fm[:, :, 2] != 0)).astype(np.uint8)
    else:
        msk = (fm != 0).astype(np.uint8)
    
    # Find bounding box
    y_coords, x_coords = np.nonzero(msk)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return im, fm
        
    minx, maxx = min(x_coords), max(x_coords)
    miny, maxy = min(y_coords), max(y_coords)
    
    # Crop to bounding box
    im = im[miny:maxy + 1, minx:maxx + 1, :]
    if fm.ndim == 3:
        fm = fm[miny:maxy + 1, minx:maxx + 1, :]
    else:
        fm = fm[miny:maxy + 1, minx:maxx + 1]
    
    # Add padding
    s = 20
    if fm.ndim == 3:
        im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        fm = np.pad(fm, ((s, s), (s, s), (0, 0)), 'constant')
    else:
        im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        fm = np.pad(fm, ((s, s), (s, s)), 'constant')
    
    # Random crop
    cx1 = random.randint(0, s - 5)
    cx2 = random.randint(0, s - 5) + 1
    cy1 = random.randint(0, s - 5)
    cy2 = random.randint(0, s - 5) + 1

    im = im[cy1:-cy2, cx1:-cx2, :]
    if fm.ndim == 3:
        fm = fm[cy1:-cy2, cx1:-cx2, :]
    else:
        fm = fm[cy1:-cy2, cx1:-cx2]
    
    return im, fm


def color_jitter_tf(im: np.ndarray, brightness: float = 0, contrast: float = 0, 
                   saturation: float = 0, hue: float = 0) -> np.ndarray:
    """
    Apply color jittering to image.
    
    Args:
        im: Input image (0-1 range)
        brightness: Brightness variation range
        contrast: Contrast variation range
        saturation: Saturation variation range (not used in original)
        hue: Hue variation range (not used in original)
        
    Returns:
        Color jittered image
    """
    # Apply contrast
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0.0, 1.0)
    
    # Apply brightness
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0.0, 1.0).astype(np.float32)
    
    return im


def change_intensity_tf(img: np.ndarray) -> np.ndarray:
    """
    Change image intensity (HSV value channel).
    
    Args:
        img: Input image in BGR format (0-255 range)
        
    Returns:
        Image with modified intensity
    """
    chance = random.uniform(0, 1)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if chance > 0.3:
        inc = random.randint(15, 50)
        # Increase value channel
        v = nimg[:, :, 2]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 2] = v

    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    return nimg


def change_hue_sat_tf(img: np.ndarray) -> np.ndarray:
    """
    Change image hue and saturation.
    
    Args:
        img: Input image in BGR format (0-255 range)
        
    Returns:
        Image with modified hue and saturation
    """
    chance = random.uniform(0, 1)
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if chance > 0.3:
        inc = random.randint(5, 15)
        # Increase hue channel
        v = nimg[:, :, 0]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 0] = v

    if chance > 0.3:
        inc = random.randint(5, 15)
        # Increase saturation channel
        v = nimg[:, :, 1]
        v = np.where(v <= 255 - inc, v + inc, 255)
        nimg[:, :, 1] = v
    
    nimg = cv2.cvtColor(nimg, cv2.COLOR_HSV2BGR)
    return nimg


def data_aug_tf(im: np.ndarray, fm: np.ndarray, bg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation with background replacement and color jittering.
    
    Args:
        im: Input image (0-255 range)
        fm: Flow map or world coordinates
        bg: Background texture image (0-255 range)
        
    Returns:
        Augmented image and flow map
    """
    # Normalize images to 0-1 range
    im = im / 255.0
    bg = bg / 255.0
    
    # Apply tight crop
    if fm.shape[-1] == 3:
        im, fm = tight_crop_tf(im, fm)
    else:
        im, fm = tight_crop_tf(im, fm)
    
    # Create mask for foreground
    if fm.shape[-1] == 3:
        msk = ((fm[:, :, 0] != 0) & (fm[:, :, 1] != 0) & (fm[:, :, 2] != 0)).astype(np.uint8)
    else:
        msk = (fm != 0).astype(np.uint8)
    
    msk = np.expand_dims(msk, axis=2)
    
    # Get image dimensions
    fh, fw, _ = im.shape
    chance = random.random()
    
    # Prepare background
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
        # Black background / no background change
        bg = np.zeros((fh, fw, 3))
        msk = np.ones((fh, fw, 3))
    
    # Composite foreground with background
    im = bg * (1 - msk) + im * msk
    
    # Apply color jittering
    im = color_jitter_tf(im, brightness=0.2, contrast=0.2, saturation=0.6, hue=0.6)
    
    return im, fm


if __name__ == "__main__":
    # Test augmentation functions
    print("Testing TensorFlow augmentations...")
    
    # Create dummy data
    height, width = 128, 128
    im = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    fm = np.random.rand(height, width, 3).astype(np.float32)
    bg = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Test tight crop
    im_crop, fm_crop = tight_crop_tf(im / 255.0, fm)
    print(f"Original shape: {im.shape}, Cropped shape: {im_crop.shape}")
    
    # Test color jitter
    im_jitter = color_jitter_tf(im / 255.0, brightness=0.2, contrast=0.2)
    print(f"Color jitter shape: {im_jitter.shape}")
    
    # Test data augmentation
    im_aug, fm_aug = data_aug_tf(im, fm, bg)
    print(f"Augmented image shape: {im_aug.shape}")
    print(f"Augmented flow map shape: {fm_aug.shape}")
    
    print("✓ TensorFlow augmentations test passed!")