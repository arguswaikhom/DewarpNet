"""TensorFlow implementation of DenseNet Architecture for DewarpNet."""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Optional, List


def add_coordconv_channels(t):
    """
    Add coordinate channels to input tensor for spatial awareness.
    
    Args:
        t: Input tensor of shape (batch, height, width, channels)
        
    Returns:
        Tensor with coordinate channels appended: (batch, height, width, channels+2)
    """
    batch_size, height, width, channels = tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3]
    
    # Create coordinate grids
    xx_channel = tf.ones((height, width), dtype=tf.float32)
    xx_range = tf.range(height, dtype=tf.float32)
    xx_range = tf.expand_dims(xx_range, -1)
    xx_coord = xx_channel * xx_range
    yy_coord = tf.transpose(xx_coord)
    
    # Normalize coordinates to [-1, 1]
    xx_coord = xx_coord / tf.cast(height - 1, tf.float32)
    yy_coord = yy_coord / tf.cast(width - 1, tf.float32)
    xx_coord = xx_coord * 2.0 - 1.0
    yy_coord = yy_coord * 2.0 - 1.0
    
    # Expand dimensions to match batch size
    xx_coord = tf.expand_dims(tf.expand_dims(xx_coord, 0), -1)
    yy_coord = tf.expand_dims(tf.expand_dims(yy_coord, 0), -1)
    xx_coord = tf.tile(xx_coord, [batch_size, 1, 1, 1])
    yy_coord = tf.tile(yy_coord, [batch_size, 1, 1, 1])
    
    # Concatenate coordinate channels
    t_cc = tf.concat([t, xx_coord, yy_coord], axis=-1)
    
    return t_cc


class DenseBlockEncoder(tf.keras.Model):
    """Dense block for encoder with dense connections."""
    
    def __init__(self, n_channels: int, n_convs: int, 
                 activation=layers.ReLU, activation_args: List = None,
                 name: str = "dense_block_encoder", **kwargs):
        super(DenseBlockEncoder, self).__init__(name=name, **kwargs)
        
        assert n_convs > 0, "Number of convolutions must be positive"
        
        self.n_channels = n_channels
        self.n_convs = n_convs
        
        if activation_args is None:
            activation_args = []
        
        # Create layers for each convolution
        self.conv_layers = []
        for i in range(n_convs):
            layer_sequence = [
                layers.BatchNormalization(name=f"{name}_bn_{i}"),
                activation(*activation_args, name=f"{name}_activation_{i}"),
                layers.Conv2D(
                    n_channels, kernel_size=3, strides=1, padding='same', 
                    use_bias=False, name=f"{name}_conv_{i}"
                )
            ]
            self.conv_layers.append(layer_sequence)
    
    def call(self, inputs, training=None):
        outputs = []
        
        for i, layer_sequence in enumerate(self.conv_layers):
            if i > 0:
                # Sum all previous outputs (dense connections)
                next_input = tf.add_n(outputs)
            else:
                next_input = inputs
            
            # Apply the layer sequence
            x = next_input
            for layer in layer_sequence:
                x = layer(x, training=training)
            
            outputs.append(x)
        
        return outputs[-1]


class DenseBlockDecoder(tf.keras.Model):
    """Dense block for decoder with dense connections."""
    
    def __init__(self, n_channels: int, n_convs: int, 
                 activation=layers.ReLU, activation_args: List = None,
                 name: str = "dense_block_decoder", **kwargs):
        super(DenseBlockDecoder, self).__init__(name=name, **kwargs)
        
        assert n_convs > 0, "Number of convolutions must be positive"
        
        self.n_channels = n_channels
        self.n_convs = n_convs
        
        if activation_args is None:
            activation_args = []
        
        # Create layers for each transposed convolution
        self.conv_layers = []
        for i in range(n_convs):
            layer_sequence = [
                layers.BatchNormalization(name=f"{name}_bn_{i}"),
                activation(*activation_args, name=f"{name}_activation_{i}"),
                layers.Conv2DTranspose(
                    n_channels, kernel_size=3, strides=1, padding='same', 
                    use_bias=False, name=f"{name}_conv_transpose_{i}"
                )
            ]
            self.conv_layers.append(layer_sequence)
    
    def call(self, inputs, training=None):
        outputs = []
        
        for i, layer_sequence in enumerate(self.conv_layers):
            if i > 0:
                # Sum all previous outputs (dense connections)
                next_input = tf.add_n(outputs)
            else:
                next_input = inputs
            
            # Apply the layer sequence
            x = next_input
            for layer in layer_sequence:
                x = layer(x, training=training)
            
            outputs.append(x)
        
        return outputs[-1]


class DenseTransitionBlockEncoder(tf.keras.Model):
    """Transition block for encoder with downsampling."""
    
    def __init__(self, n_channels_in: int, n_channels_out: int, mp: int,
                 activation=layers.ReLU, activation_args: List = None,
                 name: str = "dense_transition_encoder", **kwargs):
        super(DenseTransitionBlockEncoder, self).__init__(name=name, **kwargs)
        
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp = mp
        
        if activation_args is None:
            activation_args = []
        
        self.batch_norm = layers.BatchNormalization(name=f"{name}_bn")
        self.activation = activation(*activation_args, name=f"{name}_activation")
        self.conv = layers.Conv2D(
            n_channels_out, kernel_size=1, strides=1, padding='valid', 
            use_bias=False, name=f"{name}_conv"
        )
        self.max_pool = layers.MaxPool2D(pool_size=mp, name=f"{name}_maxpool")
    
    def call(self, inputs, training=None):
        x = self.batch_norm(inputs, training=training)
        x = self.activation(x)
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class DenseTransitionBlockDecoder(tf.keras.Model):
    """Transition block for decoder with upsampling."""
    
    def __init__(self, n_channels_in: int, n_channels_out: int,
                 activation=layers.ReLU, activation_args: List = None,
                 name: str = "dense_transition_decoder", **kwargs):
        super(DenseTransitionBlockDecoder, self).__init__(name=name, **kwargs)
        
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        
        if activation_args is None:
            activation_args = []
        
        self.batch_norm = layers.BatchNormalization(name=f"{name}_bn")
        self.activation = activation(*activation_args, name=f"{name}_activation")
        self.conv_transpose = layers.Conv2DTranspose(
            n_channels_out, kernel_size=4, strides=2, padding='same', 
            use_bias=False, name=f"{name}_conv_transpose"
        )
    
    def call(self, inputs, training=None):
        x = self.batch_norm(inputs, training=training)
        x = self.activation(x)
        x = self.conv_transpose(x)
        return x


class WaspDenseEncoder128(tf.keras.Model):
    """Dense encoder for 128x128 images with coordconv channels."""
    
    def __init__(self, nc: int = 1, ndf: int = 32, ndim: int = 128,
                 activation=layers.LeakyReLU, activation_args: List = None,
                 f_activation=layers.Activation, f_activation_args: List = None,
                 name: str = "wasp_dense_encoder_128", **kwargs):
        super(WaspDenseEncoder128, self).__init__(name=name, **kwargs)
        
        self.ndim = ndim
        
        if activation_args is None:
            activation_args = [0.2]
        if f_activation_args is None:
            f_activation_args = ['tanh']
        
        # Initial layers
        self.initial_bn = layers.BatchNormalization(name=f"{name}_initial_bn")
        self.initial_relu = layers.ReLU(name=f"{name}_initial_relu")
        self.initial_conv = layers.Conv2D(
            ndf, kernel_size=4, strides=2, padding='same', name=f"{name}_initial_conv"
        )
        
        # Dense blocks and transitions
        self.dense_block_1 = DenseBlockEncoder(ndf, 6, activation, activation_args, f"{name}_dense_1")
        self.transition_1 = DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation, activation_args, f"{name}_trans_1")
        
        self.dense_block_2 = DenseBlockEncoder(ndf*2, 12, activation, activation_args, f"{name}_dense_2")
        self.transition_2 = DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation, activation_args, f"{name}_trans_2")
        
        self.dense_block_3 = DenseBlockEncoder(ndf*4, 16, activation, activation_args, f"{name}_dense_3")
        self.transition_3 = DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation, activation_args, f"{name}_trans_3")
        
        self.dense_block_4 = DenseBlockEncoder(ndf*8, 16, activation, activation_args, f"{name}_dense_4")
        self.transition_4 = DenseTransitionBlockEncoder(ndf*8, ndf*8, 2, activation, activation_args, f"{name}_trans_4")
        
        self.dense_block_5 = DenseBlockEncoder(ndf*8, 16, activation, activation_args, f"{name}_dense_5")
        self.transition_5 = DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation, activation_args, f"{name}_trans_5")
        
        self.final_activation = f_activation(*f_activation_args, name=f"{name}_final_activation")
        
        # Flatten layer
        self.flatten = layers.Flatten(name=f"{name}_flatten")
    
    def call(self, inputs, training=None):
        # Add coordinate channels
        x = add_coordconv_channels(inputs)
        
        # Initial processing
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)
        x = self.initial_conv(x)
        
        # Dense blocks with transitions
        x = self.dense_block_1(x, training=training)
        x = self.transition_1(x, training=training)
        
        x = self.dense_block_2(x, training=training)
        x = self.transition_2(x, training=training)
        
        x = self.dense_block_3(x, training=training)
        x = self.transition_3(x, training=training)
        
        x = self.dense_block_4(x, training=training)
        x = self.transition_4(x, training=training)
        
        x = self.dense_block_5(x, training=training)
        x = self.transition_5(x, training=training)
        
        x = self.final_activation(x)
        
        # Flatten to vector
        x = self.flatten(x)
        
        return x


class WaspDenseDecoder128(tf.keras.Model):
    """Dense decoder for 128x128 images."""
    
    def __init__(self, nz: int = 128, nc: int = 1, ngf: int = 32,
                 lb: float = 0, ub: float = 1,
                 activation=layers.ReLU, activation_args: List = None,
                 f_activation=layers.Activation, f_activation_args: List = None,
                 name: str = "wasp_dense_decoder_128", **kwargs):
        super(WaspDenseDecoder128, self).__init__(name=name, **kwargs)
        
        if activation_args is None:
            activation_args = []
        if f_activation_args is None:
            f_activation_args = ['hard_sigmoid']  # Approximation of Hardtanh
        
        # Initial layers
        self.initial_bn = layers.BatchNormalization(name=f"{name}_initial_bn")
        self.initial_activation = activation(*activation_args, name=f"{name}_initial_activation")
        self.initial_conv_transpose = layers.Conv2DTranspose(
            ngf * 8, kernel_size=4, strides=1, padding='valid', 
            use_bias=False, name=f"{name}_initial_conv_transpose"
        )
        
        # Dense blocks and transitions
        self.dense_block_1 = DenseBlockDecoder(ngf*8, 16, activation, activation_args, f"{name}_dense_1")
        self.transition_1 = DenseTransitionBlockDecoder(ngf*8, ngf*8, activation, activation_args, f"{name}_trans_1")
        
        self.dense_block_2 = DenseBlockDecoder(ngf*8, 16, activation, activation_args, f"{name}_dense_2")
        self.transition_2 = DenseTransitionBlockDecoder(ngf*8, ngf*4, activation, activation_args, f"{name}_trans_2")
        
        self.dense_block_3 = DenseBlockDecoder(ngf*4, 12, activation, activation_args, f"{name}_dense_3")
        self.transition_3 = DenseTransitionBlockDecoder(ngf*4, ngf*2, activation, activation_args, f"{name}_trans_3")
        
        self.dense_block_4 = DenseBlockDecoder(ngf*2, 6, activation, activation_args, f"{name}_dense_4")
        self.transition_4 = DenseTransitionBlockDecoder(ngf*2, ngf, activation, activation_args, f"{name}_trans_4")
        
        self.dense_block_5 = DenseBlockDecoder(ngf, 6, activation, activation_args, f"{name}_dense_5")
        self.transition_5 = DenseTransitionBlockDecoder(ngf, ngf, activation, activation_args, f"{name}_trans_5")
        
        # Final layers
        self.final_bn = layers.BatchNormalization(name=f"{name}_final_bn")
        self.final_activation = activation(*activation_args, name=f"{name}_final_activation")
        self.final_conv_transpose = layers.Conv2DTranspose(
            nc, kernel_size=3, strides=1, padding='same', 
            use_bias=False, name=f"{name}_final_conv_transpose"
        )
        self.output_activation = f_activation(*f_activation_args, name=f"{name}_output_activation")
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Initial processing
        x = self.initial_bn(x, training=training)
        x = self.initial_activation(x)
        x = self.initial_conv_transpose(x)
        
        # Dense blocks with transitions
        x = self.dense_block_1(x, training=training)
        x = self.transition_1(x, training=training)
        
        x = self.dense_block_2(x, training=training)
        x = self.transition_2(x, training=training)
        
        x = self.dense_block_3(x, training=training)
        x = self.transition_3(x, training=training)
        
        x = self.dense_block_4(x, training=training)
        x = self.transition_4(x, training=training)
        
        x = self.dense_block_5(x, training=training)
        x = self.transition_5(x, training=training)
        
        # Final processing
        x = self.final_bn(x, training=training)
        x = self.final_activation(x)
        x = self.final_conv_transpose(x)
        x = self.output_activation(x)
        
        return x


class DNetCCNL(tf.keras.Model):
    """
    Main DenseNet model combining encoder and decoder.
    
    Args:
        img_size: Input image size (default: 128)
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 2 for optical flow)
        filters: Number of filters in first layer (default: 32)
        fc_units: Number of fully connected units (currently unused, default: 100)
    """
    
    def __init__(self, img_size: int = 128, in_channels: int = 1, 
                 out_channels: int = 2, filters: int = 32, fc_units: int = 100,
                 name: str = "dnet_ccnl", **kwargs):
        super(DNetCCNL, self).__init__(name=name, **kwargs)
        
        self.nc = in_channels
        self.nf = filters
        self.ndim = img_size
        self.oc = out_channels
        self.fcu = fc_units
        
        # Encoder expects input channels + 2 coordinate channels
        self.encoder = WaspDenseEncoder128(
            nc=self.nc, ndf=self.nf, ndim=self.ndim, name=f"{name}_encoder"
        )
        
        self.decoder = WaspDenseDecoder128(
            nz=self.ndim, nc=self.oc, ngf=self.nf, name=f"{name}_decoder"
        )
        
        # Reshape layer to convert flattened encoder output to 4D tensor
        self.reshape = layers.Reshape((1, 1, self.ndim), name=f"{name}_reshape")
    
    def call(self, inputs, training=None):
        # Encode input to latent vector
        encoded = self.encoder(inputs, training=training)
        
        # Reshape to 4D tensor for decoder
        encoded = self.reshape(encoded)
        
        # Decode to output
        decoded = self.decoder(encoded, training=training)
        
        return decoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'img_size': self.ndim,
            'in_channels': self.nc,
            'out_channels': self.oc,
            'filters': self.nf,
            'fc_units': self.fcu,
        })
        return config


def create_dnet_ccnl(img_size: int = 128, in_channels: int = 1, 
                     out_channels: int = 2, filters: int = 32) -> DNetCCNL:
    """
    Factory function to create a DNetCCNL model with standard parameters.
    
    Args:
        img_size: Input image size (default: 128)
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 2)
        filters: Number of filters in first layer (default: 32)
    
    Returns:
        DNetCCNL model instance
    """
    return DNetCCNL(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        filters=filters
    )