"""
TensorFlow training scripts and utilities for DewarpNet.
This package provides training scripts, configuration management, metrics computation,
and a high-level training manager for world coordinate and backward mapping models.
"""

from .train_wc import WorldCoordinateTrainer
from .train_bm import BackwardMappingTrainer
from .config import (
    TrainingPipelineConfig, ModelConfig, DataConfig, OptimizerConfig,
    LossConfig, TrainingConfig, LoggingConfig, ConfigManager,
    create_config_from_args, get_world_coordinate_config, get_backward_mapping_config,
    save_config, load_config, validate_config, print_config
)
from .metrics import (
    MetricsComputer, TrainingMonitor, PerformanceProfiler, ValidationMetrics,
    MetricsAggregator, compute_metrics, create_metrics_computer,
    create_training_monitor, create_performance_profiler
)
from .manager import (
    TrainingManager, create_training_manager,
    train_world_coordinate_model, train_backward_mapping_model
)

__all__ = [
    # Training scripts
    'WorldCoordinateTrainer',
    'BackwardMappingTrainer',
    
    # Configuration management
    'TrainingPipelineConfig',
    'ModelConfig',
    'DataConfig', 
    'OptimizerConfig',
    'LossConfig',
    'TrainingConfig',
    'LoggingConfig',
    'ConfigManager',
    'create_config_from_args',
    'get_world_coordinate_config',
    'get_backward_mapping_config',
    'save_config',
    'load_config',
    'validate_config',
    'print_config',
    
    # Metrics and monitoring
    'MetricsComputer',
    'TrainingMonitor',
    'PerformanceProfiler',
    'ValidationMetrics',
    'MetricsAggregator',
    'compute_metrics',
    'create_metrics_computer',
    'create_training_monitor',
    'create_performance_profiler',
    
    # Training manager
    'TrainingManager',
    'create_training_manager',
    'train_world_coordinate_model',
    'train_backward_mapping_model'
]