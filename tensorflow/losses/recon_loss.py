"""TensorFlow implementation of reconstruction loss function."""

import tensorflow as tf
import numpy as np
from .ssim_loss import SSIMLoss


def unwarp(img, bm):
    """Unwarp image using backward mapping coordinates.
    
    Args:
        img (tf.Tensor): Input image [batch, height, width, channels]
        bm (tf.Tensor): Backward mapping coordinates [batch, height, width, 2]
        
    Returns:
        tf.Tensor: Unwarped image [batch, height, width, channels]
    """
    n, h, w, c = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], tf.shape(img)[3]
    
    # Get backward mapping dimensions
    bm_h, bm_w = tf.shape(bm)[1], tf.shape(bm)[2]
    
    # Resize backward mapping to match image size if needed
    if bm_h != h or bm_w != w:
        bm = tf.image.resize(bm, [h, w], method='bilinear')
    
    # Convert to float64 for precision (matching PyTorch implementation)
    img = tf.cast(img, tf.float64)
    bm = tf.cast(bm, tf.float64)
    
    # TensorFlow's grid_sample expects coordinates in [-1, 1] range
    # The backward mapping coordinates should already be in this range
    
    # Use tf.raw_ops.ImageProjectiveTransformV3 or implement custom grid sampling
    # For now, we'll use a simpler approach with tf.gather_nd
    
    # Normalize coordinates to [0, height-1] and [0, width-1] for indexing
    # Assuming bm is already in [-1, 1] range, convert to pixel coordinates
    bm_normalized = bm * 0.5 + 0.5  # Convert from [-1, 1] to [0, 1]
    
    # Scale to pixel coordinates
    x_coords = bm_normalized[:, :, :, 0] * tf.cast(w - 1, tf.float64)
    y_coords = bm_normalized[:, :, :, 1] * tf.cast(h - 1, tf.float64)
    
    # Use bilinear interpolation for grid sampling
    res = bilinear_sample(img, x_coords, y_coords)
    
    return res


def bilinear_sample(img, x, y):
    """Perform bilinear sampling on image.
    
    Args:
        img (tf.Tensor): Input image [batch, height, width, channels]
        x (tf.Tensor): X coordinates [batch, height, width]
        y (tf.Tensor): Y coordinates [batch, height, width]
        
    Returns:
        tf.Tensor: Sampled image [batch, height, width, channels]
    """
    batch_size = tf.shape(img)[0]
    height = tf.shape(img)[1]
    width = tf.shape(img)[2]
    channels = tf.shape(img)[3]
    
    # Clamp coordinates to valid range
    x = tf.clip_by_value(x, 0.0, tf.cast(width - 1, tf.float64))
    y = tf.clip_by_value(y, 0.0, tf.cast(height - 1, tf.float64))
    
    # Get integer coordinates
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = tf.minimum(x0 + 1, width - 1)
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = tf.minimum(y0 + 1, height - 1)
    
    # Get fractional parts
    x_frac = x - tf.cast(x0, tf.float64)
    y_frac = y - tf.cast(y0, tf.float64)
    
    # Create batch indices with correct shape
    batch_idx = tf.range(batch_size, dtype=tf.int32)
    batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, tf.shape(x)[1], tf.shape(x)[2]])
    
    # Gather pixel values at four corners
    def gather_pixel(y_idx, x_idx):
        indices = tf.stack([batch_idx, y_idx, x_idx], axis=-1)
        return tf.gather_nd(img, indices)
    
    # Get pixel values at four corners
    I00 = gather_pixel(y0, x0)  # Top-left
    I01 = gather_pixel(y0, x1)  # Top-right
    I10 = gather_pixel(y1, x0)  # Bottom-left
    I11 = gather_pixel(y1, x1)  # Bottom-right
    
    # Expand fractional coordinates for broadcasting
    x_frac = tf.expand_dims(x_frac, axis=-1)
    y_frac = tf.expand_dims(y_frac, axis=-1)
    
    # Bilinear interpolation
    I0 = I00 * (1.0 - x_frac) + I01 * x_frac
    I1 = I10 * (1.0 - x_frac) + I11 * x_frac
    result = I0 * (1.0 - y_frac) + I1 * y_frac
    
    return result


class ReconLoss:
    """Reconstruction loss function using image unwarping."""
    
    def __init__(self, use_ssim=True, ssim_weight=1.0, mse_weight=1.0, name='recon_loss'):
        """Initialize reconstruction loss.
        
        Args:
            use_ssim (bool): Whether to include SSIM loss (default: True)
            ssim_weight (float): Weight for SSIM loss component (default: 1.0)
            mse_weight (float): Weight for MSE loss component (default: 1.0)
            name (str): Name of the loss function
        """
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight
        self.name = name
        
        if self.use_ssim:
            self.ssim_loss_fn = SSIMLoss()
    
    def __call__(self, inputs, y_true, y_pred):
        """Compute reconstruction loss.
        
        Args:
            inputs (tf.Tensor): Input images [batch, height, width, channels]
                               First 3 channels should be RGB image
            y_true (tf.Tensor): Ground truth backward mapping [batch, height, width, 2]
            y_pred (tf.Tensor): Predicted backward mapping [batch, height, width, 2]
            
        Returns:
            dict: Dictionary containing loss components
        """
        # Extract RGB image (first 3 channels)
        inp_img = inputs[:, :, :, :3]
        
        # Convert to float64 for precision
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        
        # Unwarp images using predicted and ground truth mappings
        uwpred = unwarp(inp_img, y_pred)
        uworg = unwarp(inp_img, y_true)
        
        # Convert back to float32 for loss computation
        uwpred = tf.cast(uwpred, tf.float32)
        uworg = tf.cast(uworg, tf.float32)
        
        # Compute MSE loss
        mse_loss = tf.reduce_mean(tf.square(uwpred - uworg))
        
        # Compute SSIM loss if enabled
        ssim_loss = 0.0
        if self.use_ssim:
            ssim_loss = self.ssim_loss_fn(uworg, uwpred)
        
        # Combine losses
        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'ssim_loss': ssim_loss,
            'uworg': uworg,
            'uwpred': uwpred
        }
    
    def get_config(self):
        """Get configuration for serialization."""
        return {
            'use_ssim': self.use_ssim,
            'ssim_weight': self.ssim_weight,
            'mse_weight': self.mse_weight,
            'name': self.name
        }


class UnwarpLoss:
    """Unwarp loss matching the PyTorch implementation exactly."""
    
    def __init__(self, name='unwarp_loss'):
        """Initialize unwarp loss.
        
        Args:
            name (str): Name of the loss function
        """
        self.name = name
        # Normalization parameters (set to 0 as in PyTorch implementation)
        self.xmx, self.xmn, self.ymx, self.ymn = 0.0, 0.0, 0.0, 0.0
        self.ssim_loss_fn = SSIMLoss()
    
    def __call__(self, inp, pred, label):
        """Compute unwarp loss matching PyTorch implementation.
        
        Args:
            inp (tf.Tensor): Input tensor [batch, height, width, channels]
                            First 3 channels should be RGB image
            pred (tf.Tensor): Predicted backward mapping [batch, height, width, 2]
            label (tf.Tensor): Ground truth backward mapping [batch, height, width, 2]
            
        Returns:
            tuple: (mse_loss, ssim_loss, uworg, uwpred)
        """
        # Extract RGB image (first 3 channels)
        inp_img = inp[:, :, :, :3]
        
        # Convert to float64 for precision (matching PyTorch .double())
        pred = tf.cast(pred, tf.float64)
        label = tf.cast(label, tf.float64)
        
        # Unwarp images
        uwpred = unwarp(inp_img, pred)
        uworg = unwarp(inp_img, label)
        
        # Convert back to float32 for loss computation
        uwpred_f32 = tf.cast(uwpred, tf.float32)
        uworg_f32 = tf.cast(uworg, tf.float32)
        
        # Compute MSE loss
        mse_loss = tf.reduce_mean(tf.square(uwpred_f32 - uworg_f32))
        
        # Compute SSIM loss
        ssim_loss = self.ssim_loss_fn(uworg_f32, uwpred_f32)
        
        return mse_loss, ssim_loss, uworg_f32, uwpred_f32


# Functional interfaces for compatibility
def recon_loss(inputs, y_true, y_pred, use_ssim=True, ssim_weight=1.0, mse_weight=1.0):
    """Functional interface for reconstruction loss.
    
    Args:
        inputs (tf.Tensor): Input images
        y_true (tf.Tensor): Ground truth backward mapping
        y_pred (tf.Tensor): Predicted backward mapping
        use_ssim (bool): Whether to include SSIM loss
        ssim_weight (float): Weight for SSIM loss component
        mse_weight (float): Weight for MSE loss component
        
    Returns:
        dict: Dictionary containing loss components
    """
    loss_fn = ReconLoss(use_ssim=use_ssim, ssim_weight=ssim_weight, mse_weight=mse_weight)
    return loss_fn(inputs, y_true, y_pred)


def unwarp_loss(inp, pred, label):
    """Functional interface for unwarp loss matching PyTorch implementation.
    
    Args:
        inp (tf.Tensor): Input tensor
        pred (tf.Tensor): Predicted backward mapping
        label (tf.Tensor): Ground truth backward mapping
        
    Returns:
        tuple: (mse_loss, ssim_loss, uworg, uwpred)
    """
    loss_fn = UnwarpLoss()
    return loss_fn(inp, pred, label)