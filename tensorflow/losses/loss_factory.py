"""Loss function factory for TensorFlow DewarpNet implementation."""

import tensorflow as tf
from typing import Dict, Any, Union, Optional
import logging

from .grad_loss import GradLoss, grad_loss
from .recon_loss import ReconLoss, UnwarpLoss, recon_loss, unwarp_loss
from .ssim_loss import SSIMLoss, ssim_loss


class LossFactory:
    """Factory class for creating and managing loss functions."""
    
    # Registry of available loss functions
    LOSS_REGISTRY = {
        'grad_loss': GradLoss,
        'gradient_loss': GradLoss,
        'recon_loss': ReconLoss,
        'reconstruction_loss': ReconLoss,
        'unwarp_loss': UnwarpLoss,
        'ssim_loss': SSIMLoss,
        'ssim': SSIMLoss,
    }
    
    # Default configurations for each loss type
    DEFAULT_CONFIGS = {
        'grad_loss': {
            'window_size': 3,
            'padding': 'SAME'
        },
        'recon_loss': {
            'use_ssim': True,
            'ssim_weight': 1.0,
            'mse_weight': 1.0
        },
        'unwarp_loss': {},
        'ssim_loss': {
            'window_size': 11,
            'size_average': True,
            'channels': 3
        }
    }
    
    def __init__(self):
        """Initialize loss factory."""
        self.logger = logging.getLogger(__name__)
        self._loss_instances = {}
    
    @classmethod
    def create_loss(cls, loss_type: str, **kwargs) -> Union[GradLoss, ReconLoss, UnwarpLoss, SSIMLoss]:
        """Create a loss function instance.
        
        Args:
            loss_type (str): Type of loss function to create
            **kwargs: Additional arguments for loss function initialization
            
        Returns:
            Loss function instance
            
        Raises:
            ValueError: If loss_type is not supported
        """
        if loss_type not in cls.LOSS_REGISTRY:
            available_losses = list(cls.LOSS_REGISTRY.keys())
            raise ValueError(f"Unsupported loss type: {loss_type}. "
                           f"Available losses: {available_losses}")
        
        # Get default configuration
        default_config = cls.DEFAULT_CONFIGS.get(loss_type, {})
        
        # Merge with provided kwargs
        config = {**default_config, **kwargs}
        
        # Create loss instance
        loss_class = cls.LOSS_REGISTRY[loss_type]
        return loss_class(**config)
    
    def get_loss(self, loss_type: str, **kwargs) -> Union[GradLoss, ReconLoss, UnwarpLoss, SSIMLoss]:
        """Get or create a loss function instance with caching.
        
        Args:
            loss_type (str): Type of loss function to get
            **kwargs: Additional arguments for loss function initialization
            
        Returns:
            Loss function instance
        """
        # Create cache key from loss type and kwargs
        cache_key = (loss_type, tuple(sorted(kwargs.items())))
        
        if cache_key not in self._loss_instances:
            self._loss_instances[cache_key] = self.create_loss(loss_type, **kwargs)
            self.logger.info(f"Created new loss instance: {loss_type}")
        
        return self._loss_instances[cache_key]
    
    @classmethod
    def list_available_losses(cls) -> Dict[str, str]:
        """List all available loss functions.
        
        Returns:
            Dictionary mapping loss names to their descriptions
        """
        descriptions = {
            'grad_loss': 'Gradient loss using Sobel filters',
            'gradient_loss': 'Alias for grad_loss',
            'recon_loss': 'Reconstruction loss with MSE and optional SSIM',
            'reconstruction_loss': 'Alias for recon_loss',
            'unwarp_loss': 'Unwarp loss matching PyTorch implementation',
            'ssim_loss': 'Structural Similarity Index loss',
            'ssim': 'Alias for ssim_loss'
        }
        return descriptions
    
    def clear_cache(self):
        """Clear cached loss instances."""
        self._loss_instances.clear()
        self.logger.info("Cleared loss function cache")


class CombinedLoss:
    """Combined loss function that can weight and combine multiple loss components."""
    
    def __init__(self, loss_configs: Dict[str, Dict[str, Any]], weights: Optional[Dict[str, float]] = None):
        """Initialize combined loss.
        
        Args:
            loss_configs (Dict[str, Dict[str, Any]]): Configuration for each loss component
                Format: {'loss_name': {'type': 'loss_type', 'config': {...}}}
            weights (Optional[Dict[str, float]]): Weights for each loss component
                If None, all weights default to 1.0
        """
        self.loss_configs = loss_configs
        self.weights = weights or {}
        self.factory = LossFactory()
        self.loss_functions = {}
        
        # Initialize loss functions
        for loss_name, loss_config in loss_configs.items():
            loss_type = loss_config['type']
            config = loss_config.get('config', {})
            self.loss_functions[loss_name] = self.factory.create_loss(loss_type, **config)
            
            # Set default weight if not provided
            if loss_name not in self.weights:
                self.weights[loss_name] = 1.0
    
    def __call__(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        """Compute combined loss.
        
        Args:
            *args: Arguments to pass to loss functions
            **kwargs: Keyword arguments to pass to loss functions
            
        Returns:
            Dictionary containing individual and total loss values
        """
        results = {}
        total_loss = 0.0
        
        for loss_name, loss_fn in self.loss_functions.items():
            weight = self.weights[loss_name]
            
            # Compute individual loss
            loss_result = loss_fn(*args, **kwargs)
            
            if isinstance(loss_result, dict):
                # Reconstruction loss returns a dictionary
                individual_loss = loss_result['total_loss']
                results[f'{loss_name}_components'] = loss_result
            else:
                individual_loss = loss_result
            
            # Store weighted loss
            weighted_loss = weight * individual_loss
            results[loss_name] = individual_loss
            results[f'{loss_name}_weighted'] = weighted_loss
            
            # Add to total
            total_loss += weighted_loss
        
        results['total_loss'] = total_loss
        results['weights'] = self.weights
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            'loss_configs': self.loss_configs,
            'weights': self.weights
        }


class LossLogger:
    """Utility class for logging and monitoring loss values."""
    
    def __init__(self, log_frequency: int = 10):
        """Initialize loss logger.
        
        Args:
            log_frequency (int): Frequency of logging (every N steps)
        """
        self.log_frequency = log_frequency
        self.step_count = 0
        self.loss_history = {}
        self.logger = logging.getLogger(__name__)
    
    def log_losses(self, losses: Dict[str, tf.Tensor], step: Optional[int] = None):
        """Log loss values.
        
        Args:
            losses (Dict[str, tf.Tensor]): Dictionary of loss values
            step (Optional[int]): Current training step
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        # Store in history
        if step not in self.loss_history:
            self.loss_history[step] = {}
        
        # Process and store loss values
        log_msg = f"Step {step}: "
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, tf.Tensor):
                value = loss_value.numpy()
                self.loss_history[step][loss_name] = value
                if step % self.log_frequency == 0:
                    log_msg += f"{loss_name}={value:.6f} "
            elif isinstance(loss_value, (int, float)):
                self.loss_history[step][loss_name] = loss_value
                if step % self.log_frequency == 0:
                    log_msg += f"{loss_name}={loss_value:.6f} "
            else:
                # Skip non-numeric values for logging
                continue
        
        # Log if at specified frequency
        if step % self.log_frequency == 0:
            self.logger.info(log_msg.strip())
    
    def get_loss_history(self) -> Dict[int, Dict[str, float]]:
        """Get complete loss history.
        
        Returns:
            Dictionary mapping steps to loss values
        """
        return self.loss_history
    
    def get_average_losses(self, last_n_steps: Optional[int] = None) -> Dict[str, float]:
        """Get average loss values over recent steps.
        
        Args:
            last_n_steps (Optional[int]): Number of recent steps to average over
                If None, averages over all steps
                
        Returns:
            Dictionary of average loss values
        """
        if not self.loss_history:
            return {}
        
        steps = sorted(self.loss_history.keys())
        if last_n_steps is not None:
            steps = steps[-last_n_steps:]
        
        # Collect all loss names
        all_loss_names = set()
        for step_losses in self.loss_history.values():
            all_loss_names.update(step_losses.keys())
        
        # Compute averages
        averages = {}
        for loss_name in all_loss_names:
            values = []
            for step in steps:
                if loss_name in self.loss_history[step]:
                    value = self.loss_history[step][loss_name]
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            if values:
                averages[loss_name] = sum(values) / len(values)
        
        return averages


# Convenience functions for creating common loss combinations
def create_dewarpnet_loss(grad_weight: float = 1.0, recon_weight: float = 1.0, 
                         ssim_weight: float = 1.0, **kwargs) -> CombinedLoss:
    """Create combined loss for DewarpNet training.
    
    Args:
        grad_weight (float): Weight for gradient loss
        recon_weight (float): Weight for reconstruction loss
        ssim_weight (float): Weight for SSIM component in reconstruction loss
        **kwargs: Additional configuration options
        
    Returns:
        CombinedLoss instance configured for DewarpNet
    """
    loss_configs = {
        'gradient': {
            'type': 'grad_loss',
            'config': kwargs.get('grad_config', {})
        },
        'reconstruction': {
            'type': 'recon_loss',
            'config': {
                'use_ssim': True,
                'ssim_weight': ssim_weight,
                'mse_weight': 1.0,
                **kwargs.get('recon_config', {})
            }
        }
    }
    
    weights = {
        'gradient': grad_weight,
        'reconstruction': recon_weight
    }
    
    return CombinedLoss(loss_configs, weights)


def create_world_coordinate_loss(**kwargs) -> GradLoss:
    """Create loss function for world coordinate training.
    
    Args:
        **kwargs: Configuration options for gradient loss
        
    Returns:
        GradLoss instance
    """
    return LossFactory.create_loss('grad_loss', **kwargs)


def create_backward_mapping_loss(**kwargs) -> CombinedLoss:
    """Create loss function for backward mapping training.
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        CombinedLoss instance with reconstruction and gradient losses
    """
    return create_dewarpnet_loss(**kwargs)