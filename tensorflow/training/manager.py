"""
Training manager for TensorFlow DewarpNet implementation.
Provides a high-level interface for managing the complete training pipeline.
"""

import os
import sys
import logging
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import json
import time
from datetime import datetime

# Import DewarpNet components
from .config import TrainingPipelineConfig, ConfigManager
from .metrics import MetricsComputer, TrainingMonitor, PerformanceProfiler, ValidationMetrics
from ..models.model_factory import ModelFactory
from ..loaders import Doc3DWCLoader, Doc3DBMLoader, create_wc_dataset, create_bm_dataset
from ..losses.loss_factory import LossFactory
from ..utils.gpu_utils import setup_gpu
from ..utils.training_utils import (
    TrainingLogger, CheckpointManager, TensorBoardLogger, EarlyStopping
)


class TrainingManager:
    """High-level training manager for DewarpNet models."""
    
    def __init__(self, config: TrainingPipelineConfig):
        """Initialize training manager.
        
        Args:
            config: Training pipeline configuration
        """
        self.config = config
        self.setup_logging()
        self.setup_gpu()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.loss_functions = {}
        self.datasets = {}
        self.metrics_computer = MetricsComputer()
        self.training_monitor = TrainingMonitor()
        self.performance_profiler = PerformanceProfiler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        self.training_history = []
        
        # Managers and loggers
        self.checkpoint_manager = None
        self.tensorboard_logger = None
        self.training_logger = None
        self.early_stopping = None
        
        # Setup training pipeline
        self.setup_directories()
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_losses()
        self.setup_logging_and_monitoring()
        
        # Load checkpoint if resuming
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level),
            format=self.config.logging.log_format
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing training manager for task: {self.config.task_type}")
    
    def setup_gpu(self):
        """Setup GPU configuration."""
        self.gpu_info = setup_gpu()
        self.logger.info(f"GPU setup: {self.gpu_info}")
        
        # Enable mixed precision if configured
        if self.config.training.use_mixed_precision and self.gpu_info['gpu_available']:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            self.logger.info("Enabled mixed precision training")
    
    def setup_directories(self):
        """Setup output directories."""
        # Create main directories
        os.makedirs(self.config.logging.logdir, exist_ok=True)
        os.makedirs(self.config.logging.tensorboard_log_dir, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = os.path.join(self.config.logging.logdir, 'checkpoints')
        self.logs_dir = os.path.join(self.config.logging.logdir, 'logs')
        self.outputs_dir = os.path.join(self.config.logging.logdir, 'outputs')
        
        for directory in [self.checkpoint_dir, self.logs_dir, self.outputs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Created output directories in: {self.config.logging.logdir}")
    
    def setup_model(self):
        """Setup the model."""
        self.logger.info("Setting up model...")
        
        if self.config.task_type == 'world_coordinate':
            self.model = ModelFactory.create_world_coordinate_model(
                input_nc=self.config.model.input_nc,
                output_nc=self.config.model.output_nc,
                num_downs=self.config.model.num_downs,
                ngf=self.config.model.ngf,
                use_dropout=self.config.model.use_dropout
            )
        elif self.config.task_type == 'backward_mapping':
            self.model = ModelFactory.create_backward_mapping_model(
                img_size=self.config.model.img_size,
                in_channels=self.config.model.input_nc,
                out_channels=self.config.model.output_nc,
                filters=self.config.model.filters
            )
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
        
        # Build model
        dummy_input_shape = (
            self.config.data.img_rows,
            self.config.data.img_cols,
            self.config.model.input_nc
        )
        dummy_input = tf.random.normal((1,) + dummy_input_shape)
        _ = self.model(dummy_input)
        
        self.logger.info(f"Model created with {self.model.count_params()} parameters")
    
    def setup_data(self):
        """Setup data loaders and datasets."""
        self.logger.info("Setting up data loaders...")
        
        if self.config.task_type == 'world_coordinate':
            # World coordinate data loaders
            train_loader = Doc3DWCLoader(
                root_path=self.config.data.data_path,
                split=self.config.data.train_split,
                img_size=(self.config.data.img_rows, self.config.data.img_cols),
                augmentations=self.config.data.augmentations
            )
            
            val_loader = Doc3DWCLoader(
                root_path=self.config.data.data_path,
                split=self.config.data.val_split,
                img_size=(self.config.data.img_rows, self.config.data.img_cols),
                augmentations=False
            )
            
            # Create TensorFlow datasets
            self.datasets['train'] = create_wc_dataset(
                train_loader,
                batch_size=self.config.data.batch_size,
                shuffle=self.config.data.shuffle,
                num_parallel_calls=self.config.data.num_workers,
                prefetch_buffer_size=self.config.data.prefetch_buffer_size
            )
            
            self.datasets['val'] = create_wc_dataset(
                val_loader,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_parallel_calls=self.config.data.num_workers,
                prefetch_buffer_size=self.config.data.prefetch_buffer_size
            )
            
        elif self.config.task_type == 'backward_mapping':
            # Backward mapping data loaders
            train_loader = Doc3DBMLoader(
                root_path=self.config.data.data_path,
                split=self.config.data.train_split,
                img_size=(self.config.data.img_rows, self.config.data.img_cols),
                augmentations=self.config.data.augmentations
            )
            
            val_loader = Doc3DBMLoader(
                root_path=self.config.data.data_path,
                split=self.config.data.val_split,
                img_size=(self.config.data.img_rows, self.config.data.img_cols),
                augmentations=False
            )
            
            # Create TensorFlow datasets
            self.datasets['train'] = create_bm_dataset(
                train_loader,
                batch_size=self.config.data.batch_size,
                shuffle=self.config.data.shuffle,
                num_parallel_calls=self.config.data.num_workers,
                prefetch_buffer_size=self.config.data.prefetch_buffer_size
            )
            
            self.datasets['val'] = create_bm_dataset(
                val_loader,
                batch_size=self.config.data.batch_size,
                shuffle=False,
                num_parallel_calls=self.config.data.num_workers,
                prefetch_buffer_size=self.config.data.prefetch_buffer_size
            )
        
        self.logger.info(f"Training samples: {len(train_loader)}")
        self.logger.info(f"Validation samples: {len(val_loader)}")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.logger.info("Setting up optimizer...")
        
        if self.config.optimizer.optimizer_type.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
                amsgrad=self.config.optimizer.amsgrad,
                beta_1=self.config.optimizer.beta1,
                beta_2=self.config.optimizer.beta2,
                epsilon=self.config.optimizer.epsilon
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.optimizer_type}")
        
        # Setup learning rate scheduler
        if self.config.optimizer.scheduler_type == 'reduce_on_plateau':
            self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.optimizer.scheduler_factor,
                patience=self.config.optimizer.scheduler_patience,
                verbose=1,
                mode=self.config.optimizer.scheduler_mode,
                min_lr=self.config.optimizer.scheduler_min_lr
            )
        else:
            self.lr_scheduler = None
    
    def setup_losses(self):
        """Setup loss functions."""
        self.logger.info("Setting up loss functions...")
        
        factory = LossFactory()
        
        if self.config.task_type == 'world_coordinate':
            # World coordinate losses
            self.loss_functions['l1'] = tf.keras.losses.MeanAbsoluteError()
            self.loss_functions['mse'] = tf.keras.losses.MeanSquaredError()
            
            if self.config.loss.wc_use_grad_loss:
                self.loss_functions['gradient'] = factory.create_loss(
                    'grad_loss',
                    window_size=self.config.loss.grad_window_size,
                    padding=self.config.loss.grad_padding
                )
        
        elif self.config.task_type == 'backward_mapping':
            # Backward mapping losses
            self.loss_functions['l1'] = tf.keras.losses.MeanAbsoluteError()
            self.loss_functions['mse'] = tf.keras.losses.MeanSquaredError()
            self.loss_functions['reconstruction'] = factory.create_loss('unwarp_loss')
            
            if self.config.loss.bm_use_ssim:
                self.loss_functions['ssim'] = factory.create_loss(
                    'ssim_loss',
                    window_size=self.config.loss.ssim_window_size,
                    channels=self.config.loss.ssim_channels
                )
    
    def setup_logging_and_monitoring(self):
        """Setup logging and monitoring components."""
        # Training logger
        self.training_logger = TrainingLogger(
            self.logs_dir,
            self.config.logging.experiment_name
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_dir,
            max_to_keep=self.config.training.max_checkpoints_to_keep
        )
        
        # TensorBoard logger
        if self.config.logging.use_tensorboard:
            tb_dir = os.path.join(
                self.config.logging.tensorboard_log_dir,
                self.config.logging.experiment_name
            )
            self.tensorboard_logger = TensorBoardLogger(tb_dir)
        
        # Early stopping
        if self.config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.early_stopping_min_delta,
                mode='min',
                restore_best_weights=True
            )
        
        # Validation metrics
        self.validation_metrics = ValidationMetrics(
            save_dir=os.path.join(self.outputs_dir, 'validation')
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            checkpoint = tf.train.Checkpoint(
                optimizer=self.optimizer,
                model=self.model
            )
            
            status = checkpoint.restore(checkpoint_path)
            status.expect_partial()
            
            # Try to extract epoch from filename
            filename = os.path.basename(checkpoint_path)
            if '_epoch_' in filename:
                try:
                    epoch_str = filename.split('_epoch_')[1].split('_')[0]
                    self.current_epoch = int(epoch_str)
                except (IndexError, ValueError):
                    pass
            
            self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch: {self.current_epoch}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False) -> str:
        """Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model
        )
        
        # Create filename
        suffix = 'best' if is_best else f'epoch_{epoch:04d}'
        filename = f"{self.config.model.arch}_{suffix}.ckpt"
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint, filename, is_best
        )
        
        # Log checkpoint
        self.training_logger.log_checkpoint(epoch, checkpoint_path, is_best)
        
        return checkpoint_path
    
    @tf.function
    def train_step(self, inputs: tf.Tensor, targets: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single training step.
        
        Args:
            inputs: Input batch
            targets: Target batch
            
        Returns:
            Dictionary of loss values and predictions
        """
        with tf.GradientTape() as tape:
            # Forward pass
            if self.config.task_type == 'world_coordinate':
                predictions = self.model(inputs, training=True)
                # Apply hardtanh activation for world coordinates
                predictions = tf.clip_by_value(predictions, 0.0, 1.0)
            else:  # backward_mapping
                # Use world coordinates only (channels 3:)
                world_coords = inputs[:, :, :, 3:]
                predictions = self.model(world_coords, training=True)
                # Apply hardtanh activation for backward mapping
                predictions = tf.clip_by_value(predictions, -1.0, 1.0)
            
            # Compute losses
            losses = self._compute_losses(inputs, predictions, targets)
            total_loss = losses['total_loss']
            
            # Scale for mixed precision
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss
        
        # Compute gradients
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Add predictions to results
        losses['predictions'] = predictions
        losses['gradients'] = gradients
        
        return losses
    
    @tf.function
    def val_step(self, inputs: tf.Tensor, targets: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single validation step.
        
        Args:
            inputs: Input batch
            targets: Target batch
            
        Returns:
            Dictionary of loss values and predictions
        """
        # Forward pass
        if self.config.task_type == 'world_coordinate':
            predictions = self.model(inputs, training=False)
            predictions = tf.clip_by_value(predictions, 0.0, 1.0)
        else:  # backward_mapping
            world_coords = inputs[:, :, :, 3:]
            predictions = self.model(world_coords, training=False)
            predictions = tf.clip_by_value(predictions, -1.0, 1.0)
        
        # Compute losses
        losses = self._compute_losses(inputs, predictions, targets)
        losses['predictions'] = predictions
        
        return losses
    
    def _compute_losses(self, inputs: tf.Tensor, predictions: tf.Tensor, 
                       targets: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute all relevant losses.
        
        Args:
            inputs: Input batch
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Basic losses
        losses['l1_loss'] = self.loss_functions['l1'](targets, predictions)
        losses['mse_loss'] = self.loss_functions['mse'](targets, predictions)
        
        if self.config.task_type == 'world_coordinate':
            # World coordinate specific losses
            total_loss = losses['l1_loss']
            
            if self.config.loss.wc_use_grad_loss:
                grad_loss = self.loss_functions['gradient'](predictions, targets)
                losses['gradient_loss'] = grad_loss
                total_loss += self.config.loss.wc_grad_weight * grad_loss
        
        elif self.config.task_type == 'backward_mapping':
            # Backward mapping specific losses
            rgb_wc = inputs[:, :, :, :-1]  # RGB + world coordinates
            recon_results = self.loss_functions['reconstruction'](
                rgb_wc, predictions, targets
            )
            
            losses['recon_loss'] = recon_results['recon_loss']
            losses['ssim_loss'] = recon_results.get('ssim_loss', tf.constant(0.0))
            
            # Combined loss (matching PyTorch implementation)
            total_loss = (self.config.loss.bm_l1_weight * losses['l1_loss'] + 
                         self.config.loss.bm_recon_weight * losses['recon_loss'])
            
            if self.config.loss.bm_use_ssim:
                total_loss += self.config.loss.bm_ssim_weight * losses['ssim_loss']
        
        losses['total_loss'] = total_loss
        return losses
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.logger.info(f"Training epoch {epoch + 1}/{self.config.training.n_epoch}")
        
        # Initialize metrics tracking
        from .metrics import MetricsAggregator
        metrics_agg = MetricsAggregator()
        
        # Training loop
        self.performance_profiler.start_timer('epoch_train')
        
        for batch_idx, (inputs, targets) in enumerate(self.datasets['train']):
            self.performance_profiler.start_timer('train_step')
            
            # Training step
            step_results = self.train_step(inputs, targets)
            
            self.performance_profiler.end_timer('train_step')
            
            # Update metrics
            batch_size = tf.shape(inputs)[0]
            step_metrics = {k: v for k, v in step_results.items() 
                           if k not in ['predictions', 'gradients']}
            metrics_agg.update(step_metrics, weight=float(batch_size))
            
            # Update monitoring
            step_losses = {k: float(v) for k, v in step_metrics.items()}
            self.training_monitor.update(
                step_losses, 
                gradients=step_results.get('gradients')
            )
            
            self.global_step += 1
            
            # Logging
            if (batch_idx + 1) % self.config.training.log_frequency == 0:
                avg_metrics = metrics_agg.get_averages()
                self.logger.info(
                    f"Epoch[{epoch + 1}/{self.config.training.n_epoch}] "
                    f"Batch [{batch_idx + 1}] "
                    f"Loss: {avg_metrics.get('total_loss', 0.0):.4f}"
                )
            
            # TensorBoard logging
            if (self.tensorboard_logger and 
                (batch_idx + 1) % self.config.logging.tensorboard_update_freq == 0):
                
                # Log scalars
                scalar_metrics = {f'train/{k}': v for k, v in step_losses.items()}
                self.tensorboard_logger.log_scalars(scalar_metrics, self.global_step)
                
                # Log images (sample from batch)
                if 'predictions' in step_results:
                    self._log_images_to_tensorboard(
                        inputs, targets, step_results['predictions'], 
                        self.global_step, 'train'
                    )
        
        self.performance_profiler.end_timer('epoch_train')
        
        # Get epoch metrics
        epoch_metrics = metrics_agg.get_averages()
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.logger.info(f"Validating epoch {epoch + 1}")
        
        # Use validation metrics evaluator
        val_metrics = self.validation_metrics.evaluate_epoch(
            self.model, self.datasets['val'], self.metrics_computer
        )
        
        # TensorBoard logging
        if self.tensorboard_logger:
            scalar_metrics = {f'val/{k}': v for k, v in val_metrics.items()}
            self.tensorboard_logger.log_scalars(scalar_metrics, epoch + 1)
        
        return val_metrics
    
    def _log_images_to_tensorboard(self, inputs: tf.Tensor, targets: tf.Tensor,
                                  predictions: tf.Tensor, step: int, phase: str):
        """Log images to TensorBoard.
        
        Args:
            inputs: Input images
            targets: Target images
            predictions: Predicted images
            step: Current step
            phase: Training phase ('train' or 'val')
        """
        if not self.tensorboard_logger:
            return
        
        # Select random samples
        batch_size = tf.shape(inputs)[0]
        max_images = min(self.config.logging.tensorboard_max_images, batch_size)
        indices = tf.random.shuffle(tf.range(batch_size))[:max_images]
        
        sample_inputs = tf.gather(inputs, indices)
        sample_targets = tf.gather(targets, indices)
        sample_predictions = tf.gather(predictions, indices)
        
        # Log different image types based on task
        if self.config.task_type == 'world_coordinate':
            # For world coordinates: RGB inputs, world coordinate targets/predictions
            if sample_inputs.shape[-1] >= 3:
                self.tensorboard_logger.log_images(
                    sample_inputs[:, :, :, :3], f'{phase}/inputs', step, max_images
                )
            
            self.tensorboard_logger.log_images(
                sample_targets, f'{phase}/targets', step, max_images
            )
            self.tensorboard_logger.log_images(
                sample_predictions, f'{phase}/predictions', step, max_images
            )
        
        elif self.config.task_type == 'backward_mapping':
            # For backward mapping: show RGB, world coords, and mapping predictions
            if sample_inputs.shape[-1] >= 3:
                self.tensorboard_logger.log_images(
                    sample_inputs[:, :, :, :3], f'{phase}/rgb_inputs', step, max_images
                )
            
            if sample_inputs.shape[-1] >= 6:
                self.tensorboard_logger.log_images(
                    sample_inputs[:, :, :, 3:6], f'{phase}/wc_inputs', step, max_images
                )
            
            self.tensorboard_logger.log_images(
                sample_targets, f'{phase}/mapping_targets', step, max_images
            )
            self.tensorboard_logger.log_images(
                sample_predictions, f'{phase}/mapping_predictions', step, max_images
            )
    
    def train(self) -> Dict[str, Any]:
        """Run complete training pipeline.
        
        Returns:
            Dictionary with training results and statistics
        """
        self.logger.info("Starting training pipeline...")
        
        # Save configuration
        config_path = os.path.join(self.logs_dir, 'config.json')
        ConfigManager().save_config(self.config, config_path)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.n_epoch):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            if (epoch + 1) % self.config.training.validation_frequency == 0:
                val_metrics = self.validate_epoch(epoch)
            else:
                val_metrics = {}
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            
            # Learning rate scheduling
            if self.lr_scheduler and val_metrics:
                monitor_metric = val_metrics.get('val_mse', val_metrics.get('val_total_loss', 0.0))
                self.lr_scheduler.on_epoch_end(epoch, logs={'val_loss': monitor_metric})
            
            # Log epoch results
            current_lr = float(self.optimizer.learning_rate.numpy())
            self.training_logger.log_epoch(
                epoch + 1, 'Combined', all_metrics, current_lr
            )
            
            # Check for best model
            monitor_metric = val_metrics.get('val_mse', float('inf'))
            is_best = monitor_metric < self.best_metrics.get('val_mse', float('inf'))
            
            if is_best:
                self.best_metrics.update(val_metrics)
            
            # Save checkpoints
            if is_best or (epoch + 1) % self.config.training.save_frequency == 0:
                self.save_checkpoint(epoch + 1, all_metrics, is_best)
            
            # Early stopping
            if self.early_stopping and val_metrics:
                if self.early_stopping(monitor_metric, self.model):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
            
            # Update training history
            epoch_info = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': current_lr,
                'epoch_time': time.time() - epoch_start_time
            }
            self.training_history.append(epoch_info)
            
            self.current_epoch = epoch + 1
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Print performance statistics
        self.performance_profiler.print_stats()
        
        # Save final results
        results = {
            'config': self.config,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'total_training_time': total_time,
            'performance_stats': self.performance_profiler.get_stats()
        }
        
        results_path = os.path.join(self.logs_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Close loggers
        if self.tensorboard_logger:
            self.tensorboard_logger.close()
        
        return results
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            Serializable version of object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


# Convenience functions
def create_training_manager(config: TrainingPipelineConfig) -> TrainingManager:
    """Create a training manager instance.
    
    Args:
        config: Training pipeline configuration
        
    Returns:
        TrainingManager instance
    """
    return TrainingManager(config)


def train_world_coordinate_model(data_path: str, **kwargs) -> Dict[str, Any]:
    """Train a world coordinate model with default configuration.
    
    Args:
        data_path: Path to training data
        **kwargs: Override parameters
        
    Returns:
        Training results
    """
    config = ConfigManager.create_world_coordinate_config(
        data_path=data_path, **kwargs
    )
    manager = TrainingManager(config)
    return manager.train()


def train_backward_mapping_model(data_path: str, **kwargs) -> Dict[str, Any]:
    """Train a backward mapping model with default configuration.
    
    Args:
        data_path: Path to training data
        **kwargs: Override parameters
        
    Returns:
        Training results
    """
    config = ConfigManager.create_backward_mapping_config(
        data_path=data_path, **kwargs
    )
    manager = TrainingManager(config)
    return manager.train()