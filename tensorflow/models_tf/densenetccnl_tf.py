try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Please install it.")
    import sys
    sys.exit(1)

import numpy as np
from typing import Optional


def add_coordConv_channels(t):
    """
    Add coordinate convolution channels to input tensor.
    Converts input tensor from (N, H, W, C) to (N, H, W, C+2) with coordinate channels.
    """
    n, h, w, c = tf.shape(t)[0], tf.shape(t)[1], tf.shape(t)[2], tf.shape(t)[3]
    
    # Create coordinate meshgrid
    xx_range = tf.range(h, dtype=tf.float32)
    yy_range = tf.range(w, dtype=tf.float32)
    xx_coord, yy_coord = tf.meshgrid(xx_range, yy_range, indexing='ij')
    
    # Normalize coordinates to [-1, 1]
    xx_coord = xx_coord / tf.cast(h - 1, tf.float32) * 2.0 - 1.0
    yy_coord = yy_coord / tf.cast(w - 1, tf.float32) * 2.0 - 1.0
    
    # Expand dimensions and tile for batch
    xx_coord = tf.expand_dims(tf.expand_dims(xx_coord, 0), -1)  # (1, H, W, 1)
    yy_coord = tf.expand_dims(tf.expand_dims(yy_coord, 0), -1)  # (1, H, W, 1)
    
    xx_coord = tf.tile(xx_coord, [n, 1, 1, 1])
    yy_coord = tf.tile(yy_coord, [n, 1, 1, 1])
    
    # Concatenate coordinate channels
    t_cc = tf.concat([t, xx_coord, yy_coord], axis=-1)
    
    return t_cc


class DenseBlockEncoder(tf.keras.layers.Layer):
    """Dense block for encoder with residual connections."""
    
    def __init__(self, n_channels: int, n_convs: int, name: str = "dense_block_encoder", **kwargs):
        super(DenseBlockEncoder, self).__init__(name=name, **kwargs)
        assert n_convs > 0
        
        self.n_channels = n_channels
        self.n_convs = n_convs
        
        # Build sequence of dense layers
        self.layers_list = []
        for i in range(n_convs):
            layer_seq = tf.keras.Sequential([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(n_channels, 3, strides=1, padding='same', use_bias=False)
            ], name=f"dense_layer_{i}")
            self.layers_list.append(layer_seq)
    
    def call(self, inputs, training=None):
        outputs = []
        
        for i, layer in enumerate(self.layers_list):
            if i > 0:
                # Sum all previous outputs
                next_output = tf.add_n(outputs)
                outputs.append(layer(next_output, training=training))
            else:
                outputs.append(layer(inputs, training=training))
        
        return outputs[-1]


class DenseBlockDecoder(tf.keras.layers.Layer):
    """Dense block for decoder with residual connections."""
    
    def __init__(self, n_channels: int, n_convs: int, name: str = "dense_block_decoder", **kwargs):
        super(DenseBlockDecoder, self).__init__(name=name, **kwargs)
        assert n_convs > 0
        
        self.n_channels = n_channels
        self.n_convs = n_convs
        
        # Build sequence of dense layers using transpose convolution
        self.layers_list = []
        for i in range(n_convs):
            layer_seq = tf.keras.Sequential([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(n_channels, 3, strides=1, padding='same', use_bias=False)
            ], name=f"dense_layer_{i}")
            self.layers_list.append(layer_seq)
    
    def call(self, inputs, training=None):
        outputs = []
        
        for i, layer in enumerate(self.layers_list):
            if i > 0:
                # Sum all previous outputs
                next_output = tf.add_n(outputs)
                outputs.append(layer(next_output, training=training))
            else:
                outputs.append(layer(inputs, training=training))
        
        return outputs[-1]


class DenseTransitionBlockEncoder(tf.keras.layers.Layer):
    """Transition block for encoder with pooling."""
    
    def __init__(self, n_channels_in: int, n_channels_out: int, mp: int, 
                 name: str = "dense_transition_encoder", **kwargs):
        super(DenseTransitionBlockEncoder, self).__init__(name=name, **kwargs)
        
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp = mp
        
        self.main = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv2D(n_channels_out, 1, strides=1, padding='same', use_bias=False),
            tf.keras.layers.MaxPool2D(pool_size=mp, strides=mp)
        ])
    
    def call(self, inputs, training=None):
        return self.main(inputs, training=training)


class DenseTransitionBlockDecoder(tf.keras.layers.Layer):
    """Transition block for decoder with upsampling."""
    
    def __init__(self, n_channels_in: int, n_channels_out: int, 
                 name: str = "dense_transition_decoder", **kwargs):
        super(DenseTransitionBlockDecoder, self).__init__(name=name, **kwargs)
        
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        
        self.main = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(n_channels_out, 4, strides=2, padding='same', use_bias=False)
        ])
    
    def call(self, inputs, training=None):
        return self.main(inputs, training=training)


class WaspDenseEncoder128(tf.keras.layers.Layer):
    """Dense encoder for 128x128 images."""
    
    def __init__(self, nc: int = 1, ndf: int = 32, ndim: int = 128, 
                 name: str = "wasp_dense_encoder", **kwargs):
        super(WaspDenseEncoder128, self).__init__(name=name, **kwargs)
        
        self.ndim = ndim
        self.nc = nc
        
        # Initial layers - input will have nc+2 channels after coordinate addition
        self.initial = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(ndf, 4, strides=2, padding='same')  # 128 -> 64
        ])
        
        # Dense blocks and transitions
        # Stage 1: 64x64
        self.dense1 = DenseBlockEncoder(ndf, 6, name="dense_encoder_1")
        self.trans1 = DenseTransitionBlockEncoder(ndf, ndf*2, 2, name="trans_encoder_1")  # 64 -> 32
        
        # Stage 2: 32x32  
        self.dense2 = DenseBlockEncoder(ndf*2, 12, name="dense_encoder_2")
        self.trans2 = DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, name="trans_encoder_2")  # 32 -> 16
        
        # Stage 3: 16x16
        self.dense3 = DenseBlockEncoder(ndf*4, 16, name="dense_encoder_3")
        self.trans3 = DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, name="trans_encoder_3")  # 16 -> 8
        
        # Stage 4: 8x8
        self.dense4 = DenseBlockEncoder(ndf*8, 16, name="dense_encoder_4")
        self.trans4 = DenseTransitionBlockEncoder(ndf*8, ndf*8, 2, name="trans_encoder_4")  # 8 -> 4
        
        # Stage 5: 4x4
        self.dense5 = DenseBlockEncoder(ndf*8, 16, name="dense_encoder_5")
        self.trans5 = DenseTransitionBlockEncoder(ndf*8, ndim, 4, name="trans_encoder_5")  # 4 -> 1
        
        # Final activation
        self.final_activation = tf.keras.layers.Activation('tanh')
        
        # Global pooling to get vector output
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
    
    def call(self, inputs, training=None):
        # Add coordinate channels
        x = add_coordConv_channels(inputs)
        
        # Apply network
        x = self.initial(x, training=training)
        
        x = self.dense1(x, training=training)
        x = self.trans1(x, training=training)
        
        x = self.dense2(x, training=training)
        x = self.trans2(x, training=training)
        
        x = self.dense3(x, training=training)
        x = self.trans3(x, training=training)
        
        x = self.dense4(x, training=training)
        x = self.trans4(x, training=training)
        
        x = self.dense5(x, training=training)
        x = self.trans5(x, training=training)
        
        x = self.final_activation(x)
        
        # Flatten to vector
        x = self.global_pool(x)
        
        return x


class WaspDenseDecoder128(tf.keras.layers.Layer):
    """Dense decoder for 128x128 images."""
    
    def __init__(self, nz: int = 128, nc: int = 1, ngf: int = 32, 
                 lb: float = 0, ub: float = 1, name: str = "wasp_dense_decoder", **kwargs):
        super(WaspDenseDecoder128, self).__init__(name=name, **kwargs)
        
        # Initial projection from vector to 4x4 feature map
        self.initial = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(ngf * 8, 4, strides=1, padding='valid', use_bias=False)  # 1x1 -> 4x4
        ])
        
        # Dense blocks and transitions
        # Stage 1: 4x4
        self.dense1 = DenseBlockDecoder(ngf*8, 16, name="dense_decoder_1")
        self.trans1 = DenseTransitionBlockDecoder(ngf*8, ngf*8, name="trans_decoder_1")  # 4 -> 8
        
        # Stage 2: 8x8
        self.dense2 = DenseBlockDecoder(ngf*8, 16, name="dense_decoder_2")
        self.trans2 = DenseTransitionBlockDecoder(ngf*8, ngf*4, name="trans_decoder_2")  # 8 -> 16
        
        # Stage 3: 16x16
        self.dense3 = DenseBlockDecoder(ngf*4, 12, name="dense_decoder_3")
        self.trans3 = DenseTransitionBlockDecoder(ngf*4, ngf*2, name="trans_decoder_3")  # 16 -> 32
        
        # Stage 4: 32x32
        self.dense4 = DenseBlockDecoder(ngf*2, 6, name="dense_decoder_4")
        self.trans4 = DenseTransitionBlockDecoder(ngf*2, ngf, name="trans_decoder_4")  # 32 -> 64
        
        # Stage 5: 64x64
        self.dense5 = DenseBlockDecoder(ngf, 6, name="dense_decoder_5")
        self.trans5 = DenseTransitionBlockDecoder(ngf, ngf, name="trans_decoder_5")  # 64 -> 128
        
        # Final layers
        self.final = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(nc, 3, strides=1, padding='same', use_bias=False),
            tf.keras.layers.Activation('hard_sigmoid')  # Equivalent to Hardtanh(0,1)
        ])
    
    def call(self, inputs, training=None):
        # Reshape vector input to 4D tensor for convolution
        x = tf.expand_dims(tf.expand_dims(inputs, 1), 1)  # (batch, 1, 1, nz)
        
        x = self.initial(x, training=training)
        
        x = self.dense1(x, training=training)
        x = self.trans1(x, training=training)
        
        x = self.dense2(x, training=training)
        x = self.trans2(x, training=training)
        
        x = self.dense3(x, training=training)
        x = self.trans3(x, training=training)
        
        x = self.dense4(x, training=training)
        x = self.trans4(x, training=training)
        
        x = self.dense5(x, training=training)
        x = self.trans5(x, training=training)
        
        x = self.final(x, training=training)
        
        return x


class DnetCCNL(tf.keras.Model):
    """
    TensorFlow implementation of DenseNet with coordinate convolution for backward mapping.
    """
    
    def __init__(self, img_size: int = 128, in_channels: int = 1, out_channels: int = 2, 
                 filters: int = 32, fc_units: int = 100, name: str = "dnetccnl", **kwargs):
        super(DnetCCNL, self).__init__(name=name, **kwargs)
        
        self.nc = in_channels
        self.nf = filters
        self.ndim = img_size
        self.oc = out_channels
        self.fcu = fc_units
        
        # Build encoder and decoder
        self.encoder = WaspDenseEncoder128(nc=self.nc, ndf=self.nf, ndim=self.ndim, name="encoder")
        self.decoder = WaspDenseDecoder128(nz=self.ndim, nc=self.oc, ngf=self.nf, name="decoder")
    
    def call(self, inputs, training=None):
        # Encode to latent vector
        encoded = self.encoder(inputs, training=training)
        
        # Reshape for decoder (decoder expects vector input)
        # encoded is already a vector from global pooling
        
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


def create_dnetccnl_model(img_size: int = 128, in_channels: int = 1, out_channels: int = 2,
                          filters: int = 32, input_shape: tuple = (128, 128, 1)) -> tf.keras.Model:
    """
    Factory function to create DenseNet model with proper input shape.
    
    Args:
        img_size: Input image size
        in_channels: Number of input channels (coordinate channels added automatically)
        out_channels: Number of output channels (2 for backward mapping)
        filters: Number of base filters
        input_shape: Input tensor shape (H, W, C)
    
    Returns:
        Compiled Keras model
    """
    # Create input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_coordinates")
    
    # Create DenseNet
    dnet = DnetCCNL(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        filters=filters,
        name="dnetccnl_tf"
    )
    
    # Apply DenseNet to inputs
    outputs = dnet(inputs)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dnetccnl_tf")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing DenseNet TensorFlow implementation...")
    
    # Create model for backward mapping (1 input channel -> 2 output channels)
    model = create_dnetccnl_model(img_size=128, in_channels=1, out_channels=2, filters=32, input_shape=(128, 128, 1))
    
    # Print model summary
    model.summary()
    
    # Test with dummy input
    dummy_input = tf.random.normal((1, 128, 128, 1))
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test that output has correct dimensions
    assert output.shape[1:3] == dummy_input.shape[1:3], "Output spatial dimensions should match input"
    assert output.shape[-1] == 2, "Output should have 2 channels for backward mapping"
    print("✓ DenseNet TensorFlow implementation test passed!")