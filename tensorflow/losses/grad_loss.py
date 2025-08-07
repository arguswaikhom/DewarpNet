"""TensorFlow implementation of gradient loss function."""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def sobel(window_size):
    """Create Sobel filter kernels for gradient computation.
    
    Args:
        window_size (int): Size of the Sobel kernel (must be odd)
        
    Returns:
        tuple: (matx, maty) - Sobel kernels for x and y gradients
    """
    assert window_size % 2 != 0, "Window size must be odd"
    
    ind = window_size // 2
    matx = []
    maty = []
    
    # Create x-gradient kernel
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    
    # Create y-gradient kernel
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)
    
    # Apply scaling factor based on window size
    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780
    else:
        mult = 1
    
    matx = np.array(matx) * mult
    maty = np.array(maty) * mult
    
    return tf.constant(matx, dtype=tf.float32), tf.constant(maty, dtype=tf.float32)


def create_window(window_size, channel):
    """Create convolution kernels for gradient computation.
    
    Args:
        window_size (int): Size of the Sobel kernel
        channel (int): Number of input channels
        
    Returns:
        tuple: (windowx, windowy) - Convolution kernels for x and y gradients
    """
    windowx, windowy = sobel(window_size)
    
    # Reshape for TensorFlow convolution: [height, width, in_channels, out_channels]
    windowx = tf.reshape(windowx, [window_size, window_size, 1, 1])
    windowy = tf.reshape(windowy, [window_size, window_size, 1, 1])
    
    # Expand for multiple channels
    windowx = tf.tile(windowx, [1, 1, channel, 1])
    windowy = tf.tile(windowy, [1, 1, channel, 1])
    
    return windowx, windowy


def gradient(img, windowx, windowy, window_size, padding, channel):
    """Compute gradients using Sobel filters.
    
    Args:
        img (tf.Tensor): Input image tensor [batch, height, width, channels]
        windowx (tf.Tensor): X-gradient convolution kernel
        windowy (tf.Tensor): Y-gradient convolution kernel
        window_size (int): Size of the Sobel kernel
        padding (str): Padding type for convolution
        channel (int): Number of channels
        
    Returns:
        tuple: (gradx, grady) - X and Y gradients
    """
    if channel > 1:
        # Process each channel separately
        gradx_list = []
        grady_list = []
        
        for i in range(channel):
            # Extract single channel and add channel dimension
            single_channel = tf.expand_dims(img[:, :, :, i], axis=-1)
            
            # Create single-channel kernel
            single_windowx = windowx[:, :, 0:1, :]
            single_windowy = windowy[:, :, 0:1, :]
            
            # Compute gradients for this channel
            gradx_i = tf.nn.conv2d(single_channel, single_windowx, 
                                 strides=[1, 1, 1, 1], padding=padding)
            grady_i = tf.nn.conv2d(single_channel, single_windowy, 
                                 strides=[1, 1, 1, 1], padding=padding)
            
            gradx_list.append(gradx_i)
            grady_list.append(grady_i)
        
        gradx = tf.concat(gradx_list, axis=-1)
        grady = tf.concat(grady_list, axis=-1)
    else:
        gradx = tf.nn.conv2d(img, windowx, strides=[1, 1, 1, 1], padding=padding)
        grady = tf.nn.conv2d(img, windowy, strides=[1, 1, 1, 1], padding=padding)
    
    return gradx, grady


class GradLoss(keras.losses.Loss):
    """Gradient loss function using Sobel filters."""
    
    def __init__(self, window_size=3, padding='SAME', name='grad_loss'):
        """Initialize gradient loss.
        
        Args:
            window_size (int): Size of the Sobel kernel (default: 3)
            padding (str): Padding type for convolution (default: 'SAME')
            name (str): Name of the loss function
        """
        super(GradLoss, self).__init__(name=name)
        self.window_size = window_size
        self.padding = padding
        self.channel = 1  # Will be updated based on input
        
        # Initialize kernels (will be updated in call)
        self.windowx = None
        self.windowy = None
    
    def call(self, y_true, y_pred):
        """Compute gradient loss between predicted and true values.
        
        Args:
            y_true (tf.Tensor): Ground truth tensor [batch, height, width, channels]
            y_pred (tf.Tensor): Predicted tensor [batch, height, width, channels]
            
        Returns:
            tf.Tensor: Gradient loss value
        """
        # Get input shape
        batch_size, height, width, channel = tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], tf.shape(y_pred)[3]
        
        # Create or update kernels if needed
        if self.windowx is None or tf.shape(self.windowx)[2] != channel:
            self.windowx, self.windowy = create_window(self.window_size, channel)
        
        # Compute gradients for predicted values
        pred_gradx, pred_grady = gradient(y_pred, self.windowx, self.windowy, 
                                        self.window_size, self.padding, channel)
        
        # Compute gradients for ground truth values
        label_gradx, label_grady = gradient(y_true, self.windowx, self.windowy, 
                                          self.window_size, self.padding, channel)
        
        # Compute L1 loss between gradients
        l1_loss_x = tf.reduce_mean(tf.abs(pred_gradx - label_gradx))
        l1_loss_y = tf.reduce_mean(tf.abs(pred_grady - label_grady))
        
        grad_loss = l1_loss_x + l1_loss_y
        
        return grad_loss
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super(GradLoss, self).get_config()
        config.update({
            'window_size': self.window_size,
            'padding': self.padding
        })
        return config


# Functional interface for compatibility
def grad_loss(y_true, y_pred, window_size=3, padding='SAME'):
    """Functional interface for gradient loss.
    
    Args:
        y_true (tf.Tensor): Ground truth tensor
        y_pred (tf.Tensor): Predicted tensor
        window_size (int): Size of the Sobel kernel
        padding (str): Padding type for convolution
        
    Returns:
        tf.Tensor: Gradient loss value
    """
    loss_fn = GradLoss(window_size=window_size, padding=padding)
    return loss_fn(y_true, y_pred)