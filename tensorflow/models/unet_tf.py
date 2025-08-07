"""TensorFlow implementation of UNet Generator for DewarpNet."""

import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional


class UnetSkipConnectionBlock(tf.keras.Model):
    """
    Defines the submodule with skip connection.
    X -------------------identity---------------------- X
      |-- downsampling -- |submodule| -- upsampling --|
    """
    
    def __init__(self, outer_nc: int, inner_nc: int, input_nc: Optional[int] = None,
                 submodule: Optional['UnetSkipConnectionBlock'] = None, 
                 outermost: bool = False, innermost: bool = False, 
                 norm_layer=layers.BatchNormalization, use_dropout: bool = False,
                 name: str = "unet_skip_connection_block", **kwargs):
        super(UnetSkipConnectionBlock, self).__init__(name=name, **kwargs)
        
        self.outermost = outermost
        self.innermost = innermost
        self.submodule = submodule
        
        if input_nc is None:
            input_nc = outer_nc
            
        # Determine bias usage (similar to PyTorch logic)
        use_bias = norm_layer != layers.BatchNormalization
        
        # Downsampling layers
        self.downconv = layers.Conv2D(
            inner_nc, kernel_size=4, strides=2, padding='same', 
            use_bias=use_bias, name=f"{name}_downconv"
        )
        
        if not outermost:
            self.downrelu = layers.LeakyReLU(0.2, name=f"{name}_downrelu")
        else:
            self.downrelu = None
        
        if not innermost and not outermost:
            self.downnorm = norm_layer(name=f"{name}_downnorm")
        else:
            self.downnorm = None
        
        # Upsampling layers
        self.uprelu = layers.ReLU(name=f"{name}_uprelu")
        
        if outermost:
            self.upconv = layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same',
                name=f"{name}_upconv"
            )
            self.tanh = layers.Activation('tanh', name=f"{name}_tanh")
            self.upnorm = None
        elif innermost:
            self.upconv = layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same',
                use_bias=use_bias, name=f"{name}_upconv"
            )
            self.upnorm = norm_layer(name=f"{name}_upnorm")
        else:
            self.upconv = layers.Conv2DTranspose(
                outer_nc, kernel_size=4, strides=2, padding='same',
                use_bias=use_bias, name=f"{name}_upconv"
            )
            self.upnorm = norm_layer(name=f"{name}_upnorm")
            
        if use_dropout and not outermost and not innermost:
            self.dropout = layers.Dropout(0.5, name=f"{name}_dropout")
        else:
            self.dropout = None
    
    def call(self, x, training=None):
        # Downsampling path
        if self.outermost:
            down = self.downconv(x)
        else:
            down = self.downrelu(x)
            down = self.downconv(down)
            if self.downnorm is not None:
                down = self.downnorm(down, training=training)
        
        # Apply submodule if exists
        if self.submodule is not None:
            down = self.submodule(down, training=training)
        
        # Upsampling path
        up = self.uprelu(down)
        up = self.upconv(up)
        
        if self.outermost:
            up = self.tanh(up)
            return up
        elif self.innermost:
            if self.upnorm is not None:
                up = self.upnorm(up, training=training)
            # For innermost block, return upsampled result without concatenation
            return up
        else:
            if self.upnorm is not None:
                up = self.upnorm(up, training=training)
            if self.dropout is not None:
                up = self.dropout(up, training=training)
        
        # Skip connection (concatenate input with upsampled output)
        return layers.Concatenate(axis=-1)([x, up])


class UnetGenerator(tf.keras.Model):
    """
    Defines the Unet generator.
    |num_downs|: number of downsamplings in UNet. For example,
    if |num_downs| == 7, image of size 128x128 will become of size 1x1
    at the bottleneck
    """
    
    def __init__(self, input_nc: int, output_nc: int, num_downs: int, 
                 ngf: int = 64, norm_layer=layers.BatchNormalization, 
                 use_dropout: bool = False, name: str = "unet_generator", **kwargs):
        super(UnetGenerator, self).__init__(name=name, **kwargs)
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_downs = num_downs
        self.ngf = ngf
        
        # Construct unet structure (innermost block first)
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, 
            norm_layer=norm_layer, innermost=True, 
            name=f"{name}_innermost"
        )
        
        # Add intermediate blocks
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                norm_layer=norm_layer, use_dropout=use_dropout,
                name=f"{name}_intermediate_{i}"
            )
        
        # Add remaining blocks with decreasing channel counts
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, 
            norm_layer=norm_layer, name=f"{name}_block_4"
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, 
            norm_layer=norm_layer, name=f"{name}_block_2"
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, 
            norm_layer=norm_layer, name=f"{name}_block_1"
        )
        
        # Outermost block
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, 
            outermost=True, norm_layer=norm_layer, name=f"{name}_outermost"
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


def create_unet_generator(input_nc: int = 3, output_nc: int = 3, num_downs: int = 7, 
                         ngf: int = 64, use_dropout: bool = False) -> UnetGenerator:
    """
    Factory function to create a UNet generator with standard parameters.
    
    Args:
        input_nc: Number of input channels (default: 3 for RGB)
        output_nc: Number of output channels (default: 3 for world coordinates)
        num_downs: Number of downsampling layers (default: 7)
        ngf: Number of generator filters in first conv layer (default: 64)
        use_dropout: Whether to use dropout in intermediate layers (default: False)
    
    Returns:
        UnetGenerator model instance
    """
    return UnetGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        num_downs=num_downs,
        ngf=ngf,
        use_dropout=use_dropout
    )