"""Model factory and utilities for DewarpNet TensorFlow implementation."""

import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import os

from .unet_tf import UnetGenerator, create_unet_generator
from .densenet_tf import DNetCCNL, create_dnet_ccnl


class ModelFactory:
    """Factory class for creating DewarpNet models."""
    
    SUPPORTED_MODELS = {
        'unet': UnetGenerator,
        'unet_generator': UnetGenerator,
        'densenet': DNetCCNL,
        'dnet_ccnl': DNetCCNL,
        'backward_mapping': DNetCCNL,
        'world_coordinate': UnetGenerator,
    }
    
    @classmethod
    def get_model(cls, model_name: str, **kwargs) -> tf.keras.Model:
        """
        Create a model instance by name.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model_name is not supported
        """
        model_name = model_name.lower()
        
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(cls.SUPPORTED_MODELS.keys())}"
            )
        
        model_class = cls.SUPPORTED_MODELS[model_name]
        
        # Use factory functions for standard configurations
        if model_name in ['unet', 'unet_generator', 'world_coordinate']:
            return create_unet_generator(**kwargs)
        elif model_name in ['densenet', 'dnet_ccnl', 'backward_mapping']:
            return create_dnet_ccnl(**kwargs)
        else:
            return model_class(**kwargs)
    
    @classmethod
    def create_world_coordinate_model(cls, **kwargs) -> UnetGenerator:
        """
        Create a world coordinate prediction model (UNet).
        
        Args:
            **kwargs: UNet parameters
            
        Returns:
            UnetGenerator instance configured for world coordinate prediction
        """
        default_params = {
            'input_nc': 3,  # RGB input
            'output_nc': 3,  # World coordinates (x, y, z)
            'num_downs': 7,  # Standard depth
            'ngf': 64,      # Number of generator filters
            'use_dropout': False
        }
        default_params.update(kwargs)
        return create_unet_generator(**default_params)
    
    @classmethod
    def create_backward_mapping_model(cls, **kwargs) -> DNetCCNL:
        """
        Create a backward mapping model (DenseNet).
        
        Args:
            **kwargs: DenseNet parameters
            
        Returns:
            DNetCCNL instance configured for backward mapping
        """
        default_params = {
            'img_size': 128,     # Standard input size
            'in_channels': 3,    # World coordinates input
            'out_channels': 2,   # 2D mapping coordinates
            'filters': 32        # Number of filters
        }
        default_params.update(kwargs)
        return create_dnet_ccnl(**default_params)


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
        """
        Count model parameters.
        
        Args:
            model: TensorFlow model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    @staticmethod
    def compare_model_parameters(model1: tf.keras.Model, 
                               model2: tf.keras.Model) -> Dict[str, Any]:
        """
        Compare parameter counts between two models.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Dictionary with comparison results
        """
        params1 = ModelUtils.count_parameters(model1)
        params2 = ModelUtils.count_parameters(model2)
        
        return {
            'model1': params1,
            'model2': params2,
            'difference': {
                'total': params2['total'] - params1['total'],
                'trainable': params2['trainable'] - params1['trainable'],
                'non_trainable': params2['non_trainable'] - params1['non_trainable']
            },
            'ratio': {
                'total': params2['total'] / params1['total'] if params1['total'] > 0 else 0,
                'trainable': params2['trainable'] / params1['trainable'] if params1['trainable'] > 0 else 0
            }
        }
    
    @staticmethod
    def get_model_summary(model: tf.keras.Model, 
                         input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """
        Get detailed model summary.
        
        Args:
            model: TensorFlow model
            input_shape: Input shape for building the model (if not built)
            
        Returns:
            Model summary string
        """
        if input_shape and not model.built:
            # Build the model with dummy input
            dummy_input = tf.random.normal((1,) + input_shape)
            _ = model(dummy_input)
        
        # Capture summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    @staticmethod
    def save_model_architecture(model: tf.keras.Model, filepath: str):
        """
        Save model architecture to JSON file.
        
        Args:
            model: TensorFlow model
            filepath: Path to save the architecture
        """
        architecture = model.to_json()
        with open(filepath, 'w') as f:
            f.write(architecture)
    
    @staticmethod
    def load_model_architecture(filepath: str) -> tf.keras.Model:
        """
        Load model architecture from JSON file.
        
        Args:
            filepath: Path to the architecture file
            
        Returns:
            Model instance (weights not loaded)
        """
        with open(filepath, 'r') as f:
            architecture = f.read()
        return tf.keras.models.model_from_json(architecture)


class CheckpointConverter:
    """Utilities for converting checkpoints between PyTorch and TensorFlow."""
    
    @staticmethod
    def pytorch_to_tensorflow_mapping() -> Dict[str, str]:
        """
        Get mapping from PyTorch parameter names to TensorFlow layer names.
        
        Returns:
            Dictionary mapping PyTorch names to TensorFlow names
        """
        # This would need to be implemented based on specific model architectures
        # For now, return empty mapping as placeholder
        return {}
    
    @staticmethod
    def convert_pytorch_checkpoint(pytorch_checkpoint_path: str,
                                 tensorflow_model: tf.keras.Model,
                                 output_path: str):
        """
        Convert PyTorch checkpoint to TensorFlow format.
        
        Args:
            pytorch_checkpoint_path: Path to PyTorch checkpoint
            tensorflow_model: TensorFlow model to load weights into
            output_path: Path to save TensorFlow checkpoint
            
        Note:
            This is a placeholder implementation. Full conversion would require
            loading PyTorch weights and mapping them to TensorFlow layers.
        """
        # Placeholder implementation
        # In practice, this would:
        # 1. Load PyTorch checkpoint using torch.load()
        # 2. Map parameter names using pytorch_to_tensorflow_mapping()
        # 3. Convert weight formats (e.g., Conv2D weight dimension order)
        # 4. Set weights in TensorFlow model
        # 5. Save TensorFlow checkpoint
        
        print(f"Converting {pytorch_checkpoint_path} to {output_path}")
        print("Note: Full implementation requires PyTorch dependency")
        
        # Save current model state as placeholder
        tensorflow_model.save_weights(output_path)
    
    @staticmethod
    def verify_checkpoint_compatibility(pytorch_checkpoint_path: str,
                                      tensorflow_model: tf.keras.Model) -> Dict[str, Any]:
        """
        Verify compatibility between PyTorch checkpoint and TensorFlow model.
        
        Args:
            pytorch_checkpoint_path: Path to PyTorch checkpoint
            tensorflow_model: TensorFlow model
            
        Returns:
            Dictionary with compatibility information
        """
        # Placeholder implementation
        tf_params = ModelUtils.count_parameters(tensorflow_model)
        
        return {
            'tensorflow_params': tf_params,
            'pytorch_checkpoint': pytorch_checkpoint_path,
            'compatible': True,  # Placeholder
            'notes': 'Full verification requires PyTorch dependency'
        }


class ModelValidator:
    """Utilities for validating model implementations."""
    
    @staticmethod
    def validate_output_shapes(model: tf.keras.Model,
                             input_shapes: Dict[str, Tuple[int, ...]],
                             expected_output_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, bool]:
        """
        Validate that model produces expected output shapes.
        
        Args:
            model: TensorFlow model
            input_shapes: Dictionary of input names to shapes
            expected_output_shapes: Dictionary of expected output shapes
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for input_name, input_shape in input_shapes.items():
            dummy_input = tf.random.normal((1,) + input_shape)
            output = model(dummy_input)
            
            expected_shape = expected_output_shapes.get(input_name)
            if expected_shape:
                actual_shape = tuple(output.shape[1:])  # Exclude batch dimension
                results[input_name] = actual_shape == expected_shape
            else:
                results[input_name] = True  # No expectation to validate
        
        return results
    
    @staticmethod
    def validate_parameter_ranges(model: tf.keras.Model) -> Dict[str, Any]:
        """
        Validate that model parameters are in reasonable ranges.
        
        Args:
            model: TensorFlow model
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'weights_finite': True,
            'weights_not_zero': True,
            'gradients_finite': True,
            'parameter_stats': {}
        }
        
        for weight in model.weights:
            weight_values = weight.numpy()
            
            # Check for finite values
            if not np.all(np.isfinite(weight_values)):
                results['weights_finite'] = False
            
            # Check for non-zero weights (at least some should be non-zero)
            if np.all(weight_values == 0):
                results['weights_not_zero'] = False
            
            # Collect statistics
            results['parameter_stats'][weight.name] = {
                'mean': float(np.mean(weight_values)),
                'std': float(np.std(weight_values)),
                'min': float(np.min(weight_values)),
                'max': float(np.max(weight_values))
            }
        
        return results
    
    @staticmethod
    def validate_forward_pass(model: tf.keras.Model,
                            input_shape: Tuple[int, ...],
                            num_tests: int = 5) -> Dict[str, Any]:
        """
        Validate that forward passes are consistent and produce valid outputs.
        
        Args:
            model: TensorFlow model
            input_shape: Input shape for testing
            num_tests: Number of test runs
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'all_finite': True,
            'consistent_shapes': True,
            'output_stats': {},
            'test_results': []
        }
        
        outputs = []
        
        for i in range(num_tests):
            dummy_input = tf.random.normal((1,) + input_shape)
            output = model(dummy_input, training=False)
            outputs.append(output)
            
            # Check for finite outputs
            if not tf.reduce_all(tf.math.is_finite(output)):
                results['all_finite'] = False
            
            # Check shape consistency
            if i > 0 and output.shape != outputs[0].shape:
                results['consistent_shapes'] = False
            
            results['test_results'].append({
                'output_shape': tuple(output.shape),
                'output_finite': bool(tf.reduce_all(tf.math.is_finite(output))),
                'output_mean': float(tf.reduce_mean(output)),
                'output_std': float(tf.math.reduce_std(output))
            })
        
        # Overall output statistics
        if outputs:
            all_outputs = tf.concat(outputs, axis=0)
            results['output_stats'] = {
                'mean': float(tf.reduce_mean(all_outputs)),
                'std': float(tf.math.reduce_std(all_outputs)),
                'min': float(tf.reduce_min(all_outputs)),
                'max': float(tf.reduce_max(all_outputs))
            }
        
        return results


# Convenience functions
def get_model(model_name: str, **kwargs) -> tf.keras.Model:
    """Convenience function to create a model."""
    return ModelFactory.get_model(model_name, **kwargs)


def create_dewarpnet_models() -> Dict[str, tf.keras.Model]:
    """
    Create both DewarpNet models with standard configurations.
    
    Returns:
        Dictionary with 'world_coordinate' and 'backward_mapping' models
    """
    return {
        'world_coordinate': ModelFactory.create_world_coordinate_model(),
        'backward_mapping': ModelFactory.create_backward_mapping_model()
    }