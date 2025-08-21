"""
TensorFlow implementation of gradient loss for DewarpNet.
Equivalent to PyTorch grad_loss.py
"""

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Please install TensorFlow 2.x")
    import sys
    sys.exit(1)

import numpy as np


def sobel(window_size: int):
    """
    Create Sobel kernels for gradient computation.
    Exact replica of PyTorch implementation.
    
    Args:
        window_size: Size of the Sobel kernel (must be odd)
        
    Returns:
        Tuple of (sobel_x, sobel_y) kernels
    """
    assert window_size % 2 != 0, "Window size must be odd"
    
    ind = window_size // 2
    matx = []
    maty = []
    
    # Create x-direction Sobel kernel
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gx_ij = 0
            else:
                gx_ij = i / float(i * i + j * j)
            row.append(gx_ij)
        matx.append(row)
    
    # Create y-direction Sobel kernel
    for j in range(-ind, ind + 1):
        row = []
        for i in range(-ind, ind + 1):
            if (i * i + j * j) == 0:
                gy_ij = 0
            else:
                gy_ij = j / float(i * i + j * j)
            row.append(gy_ij)
        maty.append(row)
    
    # Apply multipliers based on window size (exact same as PyTorch)
    if window_size == 3:
        mult = 2
    elif window_size == 5:
        mult = 20
    elif window_size == 7:
        mult = 780
    else:
        mult = 1  # Default multiplier
    
    matx = np.array(matx) * mult
    maty = np.array(maty) * mult
    
    return tf.constant(matx, dtype=tf.float32), tf.constant(maty, dtype=tf.float32)


def create_window(window_size: int, channel: int):
    """
    Create Sobel windows for convolution.
    
    Args:
        window_size: Size of the Sobel kernel
        channel: Number of channels
        
    Returns:
        Tuple of (windowx, windowy) for convolution
    """
    windowx, windowy = sobel(window_size)
    
    # Expand dimensions for convolution: [height, width, in_channels, out_channels]
    windowx = tf.expand_dims(tf.expand_dims(windowx, -1), -1)
    windowy = tf.expand_dims(tf.expand_dims(windowy, -1), -1)
    
    # Tile for multiple channels
    windowx = tf.tile(windowx, [1, 1, channel, 1])
    windowy = tf.tile(windowy, [1, 1, channel, 1])
    
    return windowx, windowy


def gradient(img, windowx, windowy, window_size: int, padding: str, channel: int):
    """
    Compute gradients using Sobel convolution.
    
    Args:
        img: Input image tensor (NHWC format)
        windowx: Sobel kernel for x-direction
        windowy: Sobel kernel for y-direction  
        window_size: Size of the Sobel kernel
        padding: Padding type ('SAME' or 'VALID')
        channel: Number of channels
        
    Returns:
        Tuple of (gradx, grady) gradients
    """
    if channel > 1:
        # Process each channel separately
        gradx_list = []
        grady_list = []
        
        for i in range(channel):
            # Extract single channel
            img_channel = tf.expand_dims(img[:, :, :, i], -1)  # (N, H, W, 1)
            
            # Apply convolution 
            gradx_channel = tf.nn.conv2d(
                img_channel, 
                windowx[:, :, 0:1, :],  # Use only first channel of kernel
                strides=[1, 1, 1, 1], 
                padding=padding
            )
            grady_channel = tf.nn.conv2d(
                img_channel,
                windowy[:, :, 0:1, :],  # Use only first channel of kernel
                strides=[1, 1, 1, 1],
                padding=padding
            )
            
            gradx_list.append(gradx_channel)
            grady_list.append(grady_channel)
        
        # Concatenate channels back
        gradx = tf.concat(gradx_list, axis=-1)
        grady = tf.concat(grady_list, axis=-1)
    else:
        # Single channel case
        gradx = tf.nn.conv2d(img, windowx, strides=[1, 1, 1, 1], padding=padding)
        grady = tf.nn.conv2d(img, windowy, strides=[1, 1, 1, 1], padding=padding)
    
    return gradx, grady


class GradLoss(tf.keras.losses.Loss):
    """
    TensorFlow implementation of gradient loss.
    Equivalent to PyTorch Gradloss class.
    """
    
    def __init__(self, window_size: int = 3, padding: str = 'SAME', name: str = "grad_loss"):
        super(GradLoss, self).__init__(name=name)
        
        self.window_size = window_size
        self.padding = padding
        self.channel = 1  # Will be set dynamically based on input
        
        # Create Sobel windows
        self.windowx, self.windowy = create_window(window_size, self.channel)
    
    def call(self, y_true, y_pred):
        """
        Compute gradient loss between predictions and labels.
        
        Args:
            y_true: Ground truth tensor (NHWC format)
            y_pred: Predicted tensor (NHWC format)
            
        Returns:
            Gradient loss value
        """
        # Get number of channels from input
        channel = tf.shape(y_pred)[-1]
        
        # Recreate windows if channel count has changed
        if channel != self.channel:
            self.channel = channel
            self.windowx, self.windowy = create_window(self.window_size, self.channel)
        
        # Convert to float32 if needed
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Compute gradients
        pred_gradx, pred_grady = gradient(
            y_pred, self.windowx, self.windowy, 
            self.window_size, self.padding, channel
        )
        
        label_gradx, label_grady = gradient(
            y_true, self.windowx, self.windowy,
            self.window_size, self.padding, channel
        )
        
        # Compute L1 loss on gradients
        grad_loss_x = tf.reduce_mean(tf.abs(pred_gradx - label_gradx))
        grad_loss_y = tf.reduce_mean(tf.abs(pred_grady - label_grady))
        
        return grad_loss_x + grad_loss_y


# Functional interface
def gradient_loss(y_true, y_pred, window_size: int = 5, padding: str = 'SAME'):
    """
    Functional interface for gradient loss.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        window_size: Size of Sobel kernel  
        padding: Padding type
        
    Returns:
        Gradient loss value
    """
    loss_fn = GradLoss(window_size=window_size, padding=padding)
    return loss_fn(y_true, y_pred)


if __name__ == "__main__":
    # Test gradient loss implementation
    print("Testing Gradient Loss TensorFlow implementation...")
    
    # Create dummy data
    batch_size, height, width, channels = 2, 64, 64, 3
    y_true = tf.random.normal((batch_size, height, width, channels))
    y_pred = tf.random.normal((batch_size, height, width, channels))
    
    # Test functional interface
    loss_value = gradient_loss(y_true, y_pred, window_size=5)
    print(f"Gradient loss (functional): {loss_value.numpy():.6f}")
    
    # Test class interface
    grad_loss_fn = GradLoss(window_size=5)
    loss_value_class = grad_loss_fn(y_true, y_pred)
    print(f"Gradient loss (class): {loss_value_class.numpy():.6f}")
    
    # Test that identical inputs give zero loss
    loss_identical = gradient_loss(y_true, y_true, window_size=5)
    print(f"Gradient loss (identical): {loss_identical.numpy():.6f}")
    
    # Test different window sizes
    for ws in [3, 5, 7]:
        loss_ws = gradient_loss(y_true, y_pred, window_size=ws)
        print(f"Gradient loss (window_size={ws}): {loss_ws.numpy():.6f}")
    
    print("✓ Gradient Loss TensorFlow implementation test passed!")