try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Please install TensorFlow 2.x")
    import sys
    sys.exit(1)

import numpy as np
from typing import Optional


class UnetSkipConnectionBlock(tf.keras.layers.Layer):
    """
    Defines the submodule with skip connection.
    X -------------------identity---------------------- X
      |-- downsampling -- |submodule| -- upsampling --|
    """
    
    def __init__(self, outer_nc: int, inner_nc: int, input_nc: Optional[int] = None,
                 submodule: Optional['UnetSkipConnectionBlock'] = None, outermost: bool = False, 
                 innermost: bool = False, use_batch_norm: bool = True, use_dropout: bool = False,
                 name: str = "unet_skip_connection_block", **kwargs):
        super(UnetSkipConnectionBlock, self).__init__(name=name, **kwargs)
        
        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.submodule = submodule
        
        if input_nc is None:
            input_nc = outer_nc
        
        # Build the layers based on position in UNet
        if outermost:
            # Outermost layer: no norm on input, Tanh output
            self.downconv = tf.keras.layers.Conv2D(
                inner_nc, kernel_size=4, strides=2, padding='same', use_bias=True
            )
            self.upconv = tf.keras.layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same', use_bias=True
            )
            self.uprelu = tf.keras.layers.ReLU()
            self.final_activation = tf.keras.layers.Activation('tanh')
            
        elif innermost:
            # Innermost layer: no skip connection, just down-up
            self.downrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.downconv = tf.keras.layers.Conv2D(
                inner_nc, kernel_size=4, strides=2, padding='same', use_bias=not use_batch_norm
            )
            self.uprelu = tf.keras.layers.ReLU()
            self.upconv = tf.keras.layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same', use_bias=not use_batch_norm
            )
            if use_batch_norm:
                self.upnorm = tf.keras.layers.BatchNormalization()
                
        else:
            # Middle layers: full down-norm-up structure with skip connections
            self.downrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.downconv = tf.keras.layers.Conv2D(
                inner_nc, kernel_size=4, strides=2, padding='same', use_bias=not use_batch_norm
            )
            self.uprelu = tf.keras.layers.ReLU()
            self.upconv = tf.keras.layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same', use_bias=not use_batch_norm
            )
            
            if use_batch_norm:
                self.downnorm = tf.keras.layers.BatchNormalization()
                self.upnorm = tf.keras.layers.BatchNormalization()
                
            if use_dropout:
                self.dropout = tf.keras.layers.Dropout(0.5)
                
        # Concatenation layer for skip connections
        if not outermost and not innermost:
            self.concat = tf.keras.layers.Concatenate(axis=-1)  # Channel axis for NHWC
    
    def call(self, x, training=None):
        if self.outermost:
            # Outermost block
            down = self.downconv(x)
            if self.submodule is not None:
                sub_out = self.submodule(down, training=training)
            else:
                sub_out = down
            up = self.uprelu(sub_out)
            up = self.upconv(up)
            return self.final_activation(up)
            
        elif self.innermost:
            # Innermost block
            down = self.downrelu(x)
            down = self.downconv(down)
            up = self.uprelu(down)
            up = self.upconv(up)
            if hasattr(self, 'upnorm'):
                up = self.upnorm(up, training=training)
            return up
            
        else:
            # Middle blocks with skip connections
            down = self.downrelu(x)
            down = self.downconv(down)
            if hasattr(self, 'downnorm'):
                down = self.downnorm(down, training=training)
                
            if self.submodule is not None:
                sub_out = self.submodule(down, training=training)
            else:
                sub_out = down
                
            up = self.uprelu(sub_out)
            up = self.upconv(up)
            if hasattr(self, 'upnorm'):
                up = self.upnorm(up, training=training)
                
            if self.use_dropout:
                up = self.dropout(up, training=training)
                
            # Skip connection - concatenate input with upsampled output
            return self.concat([x, up])


class UnetGenerator(tf.keras.Model):
    """
    TensorFlow implementation of UNet Generator.
    Defines the Unet generator with exact architecture as PyTorch version.
    |num_downs|: number of downsamplings in UNet. For example,
    if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck
    """
    
    def __init__(self, input_nc: int, output_nc: int, num_downs: int, ngf: int = 64,
                 use_batch_norm: bool = True, use_dropout: bool = False, name: str = "unet_generator", **kwargs):
        super(UnetGenerator, self).__init__(name=name, **kwargs)
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_downs = num_downs
        self.ngf = ngf
        
        # Construct unet structure from inside out
        # Start with the innermost layer
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, 
            use_batch_norm=use_batch_norm, innermost=True, name="innermost"
        )
        
        # Add intermediate layers
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                use_batch_norm=use_batch_norm, use_dropout=use_dropout, 
                name=f"intermediate_{i}"
            )
            
        # Add the remaining layers with decreasing channel counts
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
            use_batch_norm=use_batch_norm, name="down_4"
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
            use_batch_norm=use_batch_norm, name="down_3"
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block,
            use_batch_norm=use_batch_norm, name="down_2"
        )
        
        # Outermost layer
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, use_batch_norm=use_batch_norm, name="outermost"
        )
    
    def call(self, inputs, training=None):
        return self.model(inputs, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_nc': self.input_nc,
            'output_nc': self.output_nc,
            'num_downs': self.num_downs,
            'ngf': self.ngf,
        })
        return config


def create_unet_model(input_nc: int = 3, output_nc: int = 3, num_downs: int = 7, 
                      ngf: int = 64, input_shape: tuple = (256, 256, 3)) -> tf.keras.Model:
    """
    Factory function to create UNet model with proper input shape.
    
    Args:
        input_nc: Number of input channels
        output_nc: Number of output channels  
        num_downs: Number of downsampling layers (7 for 256x256 input)
        ngf: Number of generator filters in first conv layer
        input_shape: Input tensor shape (H, W, C)
    
    Returns:
        Compiled Keras model
    """
    # Create input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_rgb")
    
    # Create UNet generator
    unet = UnetGenerator(
        input_nc=input_nc, 
        output_nc=output_nc, 
        num_downs=num_downs, 
        ngf=ngf,
        name="unet_generator"
    )
    
    # Apply UNet to inputs
    outputs = unet(inputs)
    
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="unetnc_tf")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing UNet TensorFlow implementation...")
    
    # Create model for world coordinate regression (3 input channels -> 3 output channels)
    model = create_unet_model(input_nc=3, output_nc=3, num_downs=7, ngf=64, input_shape=(256, 256, 3))
    
    # Print model summary
    model.summary()
    
    # Test with dummy input
    dummy_input = tf.random.normal((1, 256, 256, 3))
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test that output has same spatial dimensions as input
    assert output.shape[1:3] == dummy_input.shape[1:3], "Output spatial dimensions should match input"
    print("✓ UNet TensorFlow implementation test passed!")