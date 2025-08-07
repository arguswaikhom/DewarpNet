"""
Training configuration management for TensorFlow DewarpNet implementation.
Provides configuration classes and utilities for managing training parameters.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    arch: str = 'unetnc'  # Model architecture
    input_nc: int = 3     # Input channels
    output_nc: int = 3    # Output channels
    num_downs: int = 7    # Number of downsampling layers (for UNet)
    ngf: int = 64         # Number of generator filters
    use_dropout: bool = False
    img_size: int = 256   # Input image size
    filters: int = 32     # Number of filters (for DenseNet)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str = ''
    img_rows: int = 256
    img_cols: int = 256
    batch_size: int = 1
    num_workers: int = 8
    augmentations: bool = True
    shuffle: bool = True
    prefetch_buffer_size: int = 2
    
    # Dataset splits
    train_split: str = 'train'
    val_split: str = 'val'
    test_split: str = 'test'


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    optimizer_type: str = 'adam'
    learning_rate: float = 1e-5
    weight_decay: float = 5e-4
    amsgrad: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # Learning rate scheduler
    scheduler_type: str = 'reduce_on_plateau'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-8
    scheduler_mode: str = 'min'


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # World coordinate training
    wc_l1_weight: float = 1.0
    wc_grad_weight: float = 0.2
    wc_use_grad_loss: bool = False  # Disabled in PyTorch version
    
    # Backward mapping training
    bm_l1_weight: float = 10.0
    bm_recon_weight: float = 0.5
    bm_ssim_weight: float = 0.3
    bm_use_ssim: bool = False  # Disabled in PyTorch version
    
    # Gradient loss parameters
    grad_window_size: int = 5
    grad_padding: str = 'SAME'
    
    # SSIM parameters
    ssim_window_size: int = 11
    ssim_channels: int = 3


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    n_epoch: int = 100
    start_epoch: int = 0
    log_frequency: int = 50
    save_frequency: int = 10
    validation_frequency: int = 1
    
    # Checkpointing
    save_best_only: bool = False
    max_checkpoints_to_keep: int = 5
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    
    # Mixed precision
    use_mixed_precision: bool = True
    
    # Reproducibility
    random_seed: Optional[int] = 42


@dataclass
class LoggingConfig:
    """Configuration for logging and visualization."""
    logdir: str = './checkpoints/'
    experiment_name: str = ''
    
    # TensorBoard
    use_tensorboard: bool = False
    tensorboard_log_dir: str = './runs/'
    tensorboard_update_freq: int = 20
    tensorboard_max_images: int = 8
    
    # File logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


@dataclass
class TrainingPipelineConfig:
    """Complete training pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Resume training
    resume_from: Optional[str] = None
    
    # Task type
    task_type: str = 'world_coordinate'  # 'world_coordinate' or 'backward_mapping'


class ConfigManager:
    """Manager for training configurations."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def create_world_coordinate_config(**kwargs) -> TrainingPipelineConfig:
        """Create configuration for world coordinate training.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            TrainingPipelineConfig for world coordinate training
        """
        config = TrainingPipelineConfig()
        config.task_type = 'world_coordinate'
        
        # Model configuration for world coordinate
        config.model.arch = 'unetnc'
        config.model.input_nc = 3
        config.model.output_nc = 3
        config.model.img_size = 256
        
        # Data configuration
        config.data.img_rows = 256
        config.data.img_cols = 256
        config.data.augmentations = True
        
        # Loss configuration
        config.loss.wc_use_grad_loss = False  # Matching PyTorch
        
        # Logging
        config.logging.logdir = './checkpoints-wc/'
        config.logging.experiment_name = 'htan_doc3d_l1grad_bghsaugk_scratch'
        
        # Apply overrides
        ConfigManager._apply_overrides(config, kwargs)
        
        return config
    
    @staticmethod
    def create_backward_mapping_config(**kwargs) -> TrainingPipelineConfig:
        """Create configuration for backward mapping training.
        
        Args:
            **kwargs: Override parameters
            
        Returns:
            TrainingPipelineConfig for backward mapping training
        """
        config = TrainingPipelineConfig()
        config.task_type = 'backward_mapping'
        
        # Model configuration for backward mapping
        config.model.arch = 'dnetccnl'
        config.model.input_nc = 3
        config.model.output_nc = 2
        config.model.img_size = 128
        
        # Data configuration
        config.data.img_rows = 128
        config.data.img_cols = 128
        config.data.augmentations = False  # No augmentations for BM
        
        # Optimizer configuration
        config.optimizer.scheduler_patience = 3  # Different patience for BM
        
        # Loss configuration
        config.loss.bm_use_ssim = False  # Matching PyTorch
        
        # Logging
        config.logging.logdir = './checkpoints-bm/'
        config.logging.experiment_name = 'dnetccnl_htan_swat3dmini1kbm_l1_noaug_scratch'
        
        # Apply overrides
        ConfigManager._apply_overrides(config, kwargs)
        
        return config
    
    @staticmethod
    def _apply_overrides(config: TrainingPipelineConfig, overrides: Dict[str, Any]):
        """Apply override parameters to configuration.
        
        Args:
            config: Configuration to modify
            overrides: Dictionary of override parameters
        """
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested attributes (e.g., 'model.arch')
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                # Handle top-level attributes
                if hasattr(config, key):
                    setattr(config, key, value)
    
    def save_config(self, config: TrainingPipelineConfig, filepath: str, 
                   format: str = 'json'):
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            filepath: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Configuration saved to: {filepath}")
    
    def load_config(self, filepath: str) -> TrainingPipelineConfig:
        """Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Loaded TrainingPipelineConfig
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Determine format from extension
        ext = Path(filepath).suffix.lower()
        
        if ext == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif ext in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Convert back to dataclass
        config = self._dict_to_config(config_dict)
        
        self.logger.info(f"Configuration loaded from: {filepath}")
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> TrainingPipelineConfig:
        """Convert dictionary to TrainingPipelineConfig.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            TrainingPipelineConfig instance
        """
        # Create nested configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Create main configuration
        config = TrainingPipelineConfig(
            model=model_config,
            data=data_config,
            optimizer=optimizer_config,
            loss=loss_config,
            training=training_config,
            logging=logging_config,
            resume_from=config_dict.get('resume_from'),
            task_type=config_dict.get('task_type', 'world_coordinate')
        )
        
        return config
    
    def validate_config(self, config: TrainingPipelineConfig) -> List[str]:
        """Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Validate data path
        if not config.data.data_path:
            issues.append("Data path is required")
        elif not os.path.exists(config.data.data_path):
            issues.append(f"Data path does not exist: {config.data.data_path}")
        
        # Validate model architecture
        valid_archs = ['unetnc', 'dnetccnl']
        if config.model.arch not in valid_archs:
            issues.append(f"Invalid architecture: {config.model.arch}. "
                         f"Valid options: {valid_archs}")
        
        # Validate task type
        valid_tasks = ['world_coordinate', 'backward_mapping']
        if config.task_type not in valid_tasks:
            issues.append(f"Invalid task type: {config.task_type}. "
                         f"Valid options: {valid_tasks}")
        
        # Validate consistency between task and model
        if config.task_type == 'world_coordinate' and config.model.arch != 'unetnc':
            issues.append("World coordinate task should use 'unetnc' architecture")
        
        if config.task_type == 'backward_mapping' and config.model.arch != 'dnetccnl':
            issues.append("Backward mapping task should use 'dnetccnl' architecture")
        
        # Validate image sizes
        if config.data.img_rows <= 0 or config.data.img_cols <= 0:
            issues.append("Image dimensions must be positive")
        
        # Validate batch size
        if config.data.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        # Validate learning rate
        if config.optimizer.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        # Validate epochs
        if config.training.n_epoch <= 0:
            issues.append("Number of epochs must be positive")
        
        return issues
    
    def print_config(self, config: TrainingPipelineConfig):
        """Print configuration in a readable format.
        
        Args:
            config: Configuration to print
        """
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        
        print(f"\nTask Type: {config.task_type}")
        
        print(f"\nModel Configuration:")
        print(f"  Architecture: {config.model.arch}")
        print(f"  Input channels: {config.model.input_nc}")
        print(f"  Output channels: {config.model.output_nc}")
        print(f"  Image size: {config.model.img_size}")
        
        print(f"\nData Configuration:")
        print(f"  Data path: {config.data.data_path}")
        print(f"  Image size: {config.data.img_rows}x{config.data.img_cols}")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Augmentations: {config.data.augmentations}")
        
        print(f"\nOptimizer Configuration:")
        print(f"  Type: {config.optimizer.optimizer_type}")
        print(f"  Learning rate: {config.optimizer.learning_rate}")
        print(f"  Weight decay: {config.optimizer.weight_decay}")
        print(f"  Scheduler: {config.optimizer.scheduler_type}")
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {config.training.n_epoch}")
        print(f"  Mixed precision: {config.training.use_mixed_precision}")
        print(f"  Early stopping: {config.training.early_stopping}")
        
        print(f"\nLogging Configuration:")
        print(f"  Log directory: {config.logging.logdir}")
        print(f"  Experiment name: {config.logging.experiment_name}")
        print(f"  TensorBoard: {config.logging.use_tensorboard}")
        
        if config.resume_from:
            print(f"\nResume from: {config.resume_from}")
        
        print("=" * 60)


def create_config_from_args(args) -> TrainingPipelineConfig:
    """Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        TrainingPipelineConfig instance
    """
    # Determine task type from architecture or explicit parameter
    task_type = getattr(args, 'task_type', None)
    if task_type is None:
        if args.arch == 'unetnc':
            task_type = 'world_coordinate'
        elif args.arch == 'dnetccnl':
            task_type = 'backward_mapping'
        else:
            task_type = 'world_coordinate'  # Default
    
    # Create base configuration
    if task_type == 'world_coordinate':
        config = ConfigManager.create_world_coordinate_config()
    else:
        config = ConfigManager.create_backward_mapping_config()
    
    # Override with command line arguments
    overrides = {}
    
    # Model overrides
    if hasattr(args, 'arch'):
        overrides['model.arch'] = args.arch
    
    # Data overrides
    if hasattr(args, 'data_path'):
        overrides['data.data_path'] = args.data_path
    if hasattr(args, 'img_rows'):
        overrides['data.img_rows'] = args.img_rows
        overrides['model.img_size'] = args.img_rows
    if hasattr(args, 'img_cols'):
        overrides['data.img_cols'] = args.img_cols
    if hasattr(args, 'batch_size'):
        overrides['data.batch_size'] = args.batch_size
    
    # Training overrides
    if hasattr(args, 'n_epoch'):
        overrides['training.n_epoch'] = args.n_epoch
    if hasattr(args, 'l_rate'):
        overrides['optimizer.learning_rate'] = args.l_rate
    
    # Logging overrides
    if hasattr(args, 'logdir'):
        overrides['logging.logdir'] = args.logdir
    if hasattr(args, 'tboard'):
        overrides['logging.use_tensorboard'] = args.tboard
    
    # Resume override
    if hasattr(args, 'resume'):
        overrides['resume_from'] = args.resume
    
    # Apply overrides
    ConfigManager._apply_overrides(config, overrides)
    
    return config


# Convenience functions
def get_world_coordinate_config(**kwargs) -> TrainingPipelineConfig:
    """Get world coordinate training configuration."""
    return ConfigManager.create_world_coordinate_config(**kwargs)


def get_backward_mapping_config(**kwargs) -> TrainingPipelineConfig:
    """Get backward mapping training configuration."""
    return ConfigManager.create_backward_mapping_config(**kwargs)


def save_config(config: TrainingPipelineConfig, filepath: str, format: str = 'json'):
    """Save configuration to file."""
    manager = ConfigManager()
    manager.save_config(config, filepath, format)


def load_config(filepath: str) -> TrainingPipelineConfig:
    """Load configuration from file."""
    manager = ConfigManager()
    return manager.load_config(filepath)


def validate_config(config: TrainingPipelineConfig) -> List[str]:
    """Validate configuration."""
    manager = ConfigManager()
    return manager.validate_config(config)


def print_config(config: TrainingPipelineConfig):
    """Print configuration."""
    manager = ConfigManager()
    manager.print_config(config)