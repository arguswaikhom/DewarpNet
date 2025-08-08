"""TensorFlow implementation of SSIM loss function."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    """Create 1D Gaussian kernel.
    
    Args:
        window_size (int): Size of the Gaussian window
        sigma (float): Standard deviation of the Gaussian
        
    Returns:
        tf.Tensor: 1D Gaussian kernel
    """
    gauss = tf.constant([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
        for x in range(window_size)
    ], dtype=tf.float32)
    return gauss / tf.reduce_sum(gauss)


def create_window(window_size, channel):
    """Create 2D Gaussian window for SSIM computation.
    
    Args:
        window_size (int): Size of the window
        channel (int): Number of channels
        
    Returns:
        tf.Tensor: 2D Gaussian window [window_size, window_size, channel, 1]
    """
    _1D_window = tf.expand_dims(gaussian(window_size, 1.5), axis=1)
    _2D_window = tf.matmul(_1D_window, tf.transpose(_1D_window))
    _2D_window = tf.expand_dims(tf.expand_dims(_2D_window, axis=-1), axis=-1)
    
    # Expand for multiple channels: [window_size, window_size, channel, 1]
    window = tf.tile(_2D_window, [1, 1, channel, 1])
    
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Compute SSIM between two images.
    
    Args:
        img1 (tf.Tensor): First image [batch, height, width, channels]
        img2 (tf.Tensor): Second image [batch, height, width, channels]
        window (tf.Tensor): Gaussian window for convolution
        window_size (int): Size of the window
        channel (int): Number of channels
        size_average (bool): Whether to average over spatial dimensions
        
    Returns:
        tf.Tensor: SSIM value
    """
    padding = window_size // 2
    
    # Compute means
    mu1 = tf.nn.depthwise_conv2d(img1, window, strides=[1, 1, 1, 1], padding='SAME')
    mu2 = tf.nn.depthwise_conv2d(img2, window, strides=[1, 1, 1, 1], padding='SAME')
    
    mu1_sq = tf.square(mu1)
    mu2_sq = tf.square(mu2)
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = tf.nn.depthwise_conv2d(tf.square(img1), window, strides=[1, 1, 1, 1], padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.depthwise_conv2d(tf.square(img2), window, strides=[1, 1, 1, 1], padding='SAME') - mu2_sq
    sigma12 = tf.nn.depthwise_conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='SAME') - mu1_mu2
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return tf.reduce_mean(ssim_map)
    else:
        return tf.reduce_mean(ssim_map, axis=[1, 2, 3])


class SSIMLoss(keras.losses.Loss):
    """SSIM loss function."""
    
    def __init__(self, window_size=11, size_average=True, channels=3, name='ssim_loss'):
        """Initialize SSIM loss.
        
        Args:
            window_size (int): Size of the Gaussian window (default: 11)
            size_average (bool): Whether to average over spatial dimensions (default: True)
            channels (int): Number of channels (default: 3)
            name (str): Name of the loss function
        """
        super(SSIMLoss, self).__init__(name=name)
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channels

    
    def call(self, y_true, y_pred):
        """Compute SSIM loss between predicted and true values.
        
        Args:
            y_true (tf.Tensor): Ground truth tensor [batch, height, width, channels]
            y_pred (tf.Tensor): Predicted tensor [batch, height, width, channels]
            
        Returns:
            tf.Tensor: SSIM loss value (1 - SSIM)
        """
        # Get channel dimension
        channel = tf.shape(y_pred)[-1]
        
        # Always create window fresh to avoid graph scope issues
        window = create_window(self.window_size, channel)
        
        # Compute SSIM
        ssim_value = _ssim(y_pred, y_true, window, self.window_size, channel, self.size_average)
        
        # Return 1 - SSIM as loss (higher SSIM = lower loss)
        return 1.0 - ssim_value
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super(SSIMLoss, self).get_config()
        config.update({
            'window_size': self.window_size,
            'size_average': self.size_average,
            'channels': self.channel
        })
        return config


# Functional interface for compatibility
def ssim_loss(y_true, y_pred, window_size=11, size_average=True):
    """Functional interface for SSIM loss.
    
    Args:
        y_true (tf.Tensor): Ground truth tensor
        y_pred (tf.Tensor): Predicted tensor
        window_size (int): Size of the Gaussian window
        size_average (bool): Whether to average over spatial dimensions
        
    Returns:
        tf.Tensor: SSIM loss value
    """
    loss_fn = SSIMLoss(window_size=window_size, size_average=size_average)
    return loss_fn(y_true, y_pred)


def ssim(y_true, y_pred, window_size=11, size_average=True):
    """Compute SSIM value (not loss) between two images.
    
    Args:
        y_true (tf.Tensor): Ground truth tensor
        y_pred (tf.Tensor): Predicted tensor
        window_size (int): Size of the Gaussian window
        size_average (bool): Whether to average over spatial dimensions
        
    Returns:
        tf.Tensor: SSIM value
    """
    channel = tf.shape(y_pred)[-1]
    window = create_window(window_size, channel)
    return _ssim(y_pred, y_true, window, window_size, channel, size_average)