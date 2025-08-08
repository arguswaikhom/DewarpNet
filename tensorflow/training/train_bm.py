"""
TensorFlow implementation of backward mapping training script.
Replicates the PyTorch trainbm.py logic for training backward mapping regression from world coordinates.
"""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import random

# TensorBoard for visualization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import Progbar

# Import TensorFlow DewarpNet components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import ModelFactory
from loaders import Doc3DBMLoader, create_bm_dataset
from losses.loss_factory import LossFactory, LossLogger, CombinedLoss
from losses.recon_loss import UnwarpLoss
from utils.gpu_utils import setup_gpu, get_gpu_info
from utils.training_utils import (
    TrainingLogger, MetricsTracker, LearningRateScheduler,
    CheckpointManager, TensorBoardLogger
)


class BackwardMappingTrainer:
    """Trainer class for backward mapping regression model."""
    
    def __init__(self, args):
        """Initialize trainer with configuration."""
        self.args = args
        self.setup_logging()
        self.setup_gpu()
        self.setup_directories()
        
        # Training state
        self.global_step = 0
        self.best_val_mse = float('inf')
        self.best_val_uwarp_ssim = float('inf')
        self.epoch_start = 0
        
        # Initialize components
        self.setup_data_loaders()
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.setup_tensorboard()
        self.setup_checkpointing()
        
        # Load checkpoint if resuming
        if args.resume:
            self.load_checkpoint(args.resume)
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create experiment name matching PyTorch version
        self.experiment_name = 'dnetccnl_htan_swat3dmini1kbm_l1_noaug_scratch'
        self.logger.info(f"Starting experiment: {self.experiment_name}")
    
    def setup_gpu(self):
        """Setup GPU configuration."""
        self.gpu_info = setup_gpu()
        self.logger.info(f"GPU setup: {self.gpu_info}")
        
        # Disable mixed precision for compatibility
        # Mixed precision can cause issues with gradient loss functions
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        self.logger.info("Using float32 precision for compatibility")
    
    def setup_directories(self):
        """Setup output directories."""
        os.makedirs(self.args.logdir, exist_ok=True)
        
        # Log file setup
        self.log_file_path = os.path.join(
            self.args.logdir, 
            f"{self.experiment_name}.txt"
        )
        
        # Write experiment header
        with open(self.log_file_path, 'a') as f:
            f.write(f'\n---------------  {self.experiment_name}  ---------------\n')
    
    def setup_data_loaders(self):
        """Setup training and validation data loaders."""
        self.logger.info("Setting up data loaders...")
        
        # Training loader
        self.train_loader = Doc3DBMLoader(
            root=self.args.data_path,
            split='train',
            img_size=(self.args.img_rows, self.args.img_cols)
        )
        
        # Validation loader
        self.val_loader = Doc3DBMLoader(
            root=self.args.data_path,
            split='val',
            img_size=(self.args.img_rows, self.args.img_cols)
        )
        
        # Create TensorFlow datasets
        self.train_dataset = create_bm_dataset(
            root=self.args.data_path,
            split='train',
            batch_size=self.args.batch_size,
            img_size=(self.args.img_rows, self.args.img_cols),
            shuffle=True,
            num_parallel_calls=8,
            prefetch_buffer=tf.data.AUTOTUNE
        )
        
        self.val_dataset = create_bm_dataset(
            root=self.args.data_path,
            split='val',
            batch_size=self.args.batch_size,
            img_size=(self.args.img_rows, self.args.img_cols),
            shuffle=False,
            num_parallel_calls=8,
            prefetch_buffer=tf.data.AUTOTUNE
        )
        
        self.logger.info(f"Training samples: {len(self.train_loader)}")
        self.logger.info(f"Validation samples: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup the backward mapping regression model."""
        self.logger.info("Setting up model...")
        
        # Create model using factory (DenseNet for backward mapping)
        self.model = ModelFactory.create_backward_mapping_model(
            img_size=self.args.img_rows,
            in_channels=3,  # World coordinates input
            out_channels=2,  # 2D mapping coordinates
            filters=32
        )
        
        # Build model with dummy input
        dummy_input = tf.random.normal((1, self.args.img_rows, self.args.img_cols, 3))
        _ = self.model(dummy_input)
        
        self.logger.info(f"Model created with {self.model.count_params()} parameters")
        
        # Hardtanh activation (equivalent to PyTorch version)
        self.activation = lambda x: tf.clip_by_value(x, -1.0, 1.0)
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.logger.info("Setting up optimizer...")
        
        # Adam optimizer with same parameters as PyTorch
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.args.l_rate,
            weight_decay=5e-4,
            amsgrad=True
        )
        
        # Learning rate scheduler (ReduceLROnPlateau equivalent)
        self.lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mse',
            factor=0.5,
            patience=3,  # Different patience from WC training
            verbose=1,
            mode='min',
            min_lr=1e-8
        )
        
        # Set the model for the learning rate scheduler
        if hasattr(self, 'model') and self.model is not None:
            self.lr_scheduler.set_model(self.model)
    
    def setup_loss_functions(self):
        """Setup loss functions."""
        self.logger.info("Setting up loss functions...")
        
        # Create loss functions
        self.loss_factory = LossFactory()
        
        # Reconstruction/Unwarp loss (matching PyTorch implementation)
        self.unwarp_loss_fn = UnwarpLoss()
        
        # MSE and L1 loss functions
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        self.l1_loss_fn = tf.keras.losses.MeanAbsoluteError()
        
        # Loss logger
        self.loss_logger = LossLogger(log_frequency=50)
    
    def setup_tensorboard(self):
        """Setup TensorBoard logging."""
        if self.args.tboard:
            self.logger.info("Setting up TensorBoard...")
            
            # Create TensorBoard log directory
            tb_log_dir = os.path.join('runs', self.experiment_name)
            os.makedirs(tb_log_dir, exist_ok=True)
            
            # Create summary writer
            self.summary_writer = tf.summary.create_file_writer(tb_log_dir)
            self.logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
        else:
            self.summary_writer = None
    
    def setup_checkpointing(self):
        """Setup model checkpointing."""
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if os.path.isfile(checkpoint_path):
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
            
            try:
                # Load TensorFlow checkpoint
                status = self.checkpoint.restore(checkpoint_path)
                status.expect_partial()  # Ignore missing variables
                
                # Extract epoch from filename if possible
                filename = os.path.basename(checkpoint_path)
                if 'epoch' in filename:
                    try:
                        self.epoch_start = int(filename.split('_')[1])
                    except (IndexError, ValueError):
                        self.epoch_start = 0
                
                self.logger.info(f"Loaded checkpoint, resuming from epoch {self.epoch_start}")
                
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.epoch_start = 0
        else:
            self.logger.warning(f"No checkpoint found at: {checkpoint_path}")
    
    def write_log_file(self, losses: Dict[str, float], epoch: int, lr: float, phase: str):
        """Write training logs to file."""
        with open(self.log_file_path, 'a') as f:
            f.write(f"\n{phase} LRate: {lr:.8f} Epoch: {epoch} "
                   f"Loss: {losses['l1_loss']:.6f} MSE: {losses['mse_loss']:.6f} "
                   f"UnwarpL2: {losses['recon_loss']:.6f} "
                   f"UnwarpSSIMloss: {losses['ssim_loss']:.6f}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.learning_rate.numpy())
    
    @tf.function
    def train_step(self, images: tf.Tensor, labels: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single training step."""
        with tf.GradientTape() as tape:
            # Forward pass - use only world coordinates (channels 3:)
            world_coords = images[:, :, :, 3:]  # Skip RGB channels, use world coordinates
            outputs = self.model(world_coords, training=True)
            
            # Apply activation and transpose to match PyTorch format (NHWC)
            target_nhwc = self.activation(outputs)
            
            # Compute L1 loss
            l1_loss = self.l1_loss_fn(target_nhwc, labels)
            
            # Compute reconstruction/unwarp loss
            # Use RGB + world coords (skip last channel) for unwarping
            rgb_wc = images[:, :, :, :-1]  # RGB + world coordinates
            recon_loss, ssim_loss, uworg, uwpred = self.unwarp_loss_fn(rgb_wc, target_nhwc, labels)
            
            # Combined loss (matching PyTorch: 10.0*L1 + 0.5*recon)
            total_loss = (10.0 * l1_loss) + (0.5 * recon_loss)
            
            # MSE for monitoring
            mse_loss = self.mse_loss_fn(target_nhwc, labels)
            
            # Scale loss for mixed precision
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
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'recon_loss': recon_loss,
            'ssim_loss': ssim_loss,
            'mse_loss': mse_loss,
            'predictions': target_nhwc,
            'unwarp_gt': uworg,
            'unwarp_pred': uwpred
        }
    
    @tf.function
    def val_step(self, images: tf.Tensor, labels: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single validation step."""
        # Forward pass - use only world coordinates (channels 3:)
        world_coords = images[:, :, :, 3:]
        outputs = self.model(world_coords, training=False)
        
        # Apply activation and transpose to match PyTorch format
        target_nhwc = self.activation(outputs)
        
        # Compute losses
        l1_loss = self.l1_loss_fn(target_nhwc, labels)
        mse_loss = self.mse_loss_fn(target_nhwc, labels)
        
        # Compute reconstruction/unwarp loss
        rgb_wc = images[:, :, :, :-1]
        recon_loss, ssim_loss, uworg, uwpred = self.unwarp_loss_fn(rgb_wc, target_nhwc, labels)
        
        return {
            'l1_loss': l1_loss,
            'recon_loss': recon_loss,
            'ssim_loss': ssim_loss,
            'mse_loss': mse_loss,
            'predictions': target_nhwc,
            'unwarp_gt': uworg,
            'unwarp_pred': uwpred
        }
    
    def log_tensorboard_images(self, unwarp_pred: tf.Tensor, unwarp_gt: tf.Tensor, 
                              step: int, phase: str):
        """Log unwarp images to TensorBoard."""
        if self.summary_writer is None or unwarp_pred is None or unwarp_gt is None:
            return
        
        # Select random samples for visualization
        batch_size = tf.shape(unwarp_pred)[0]
        num_samples = min(8, batch_size)
        indices = tf.random.shuffle(tf.range(batch_size))[:num_samples]
        
        sample_pred = tf.gather(unwarp_pred, indices)
        sample_gt = tf.gather(unwarp_gt, indices)
        
        with self.summary_writer.as_default():
            # Log ground truth unwarp
            tf.summary.image(
                f'{phase} GT unwarp',
                sample_gt,
                step=step,
                max_outputs=num_samples
            )
            
            # Log predicted unwarp
            tf.summary.image(
                f'{phase} Pred Unwarp',
                sample_pred,
                step=step,
                max_outputs=num_samples
            )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.logger.info(f"Training epoch {epoch + 1}/{self.args.n_epoch}")
        
        # Initialize metrics
        epoch_losses = {
            'total_loss': 0.0,
            'l1_loss': 0.0,
            'recon_loss': 0.0,
            'ssim_loss': 0.0,
            'mse_loss': 0.0
        }
        
        num_batches = 0
        
        # Training loop
        for batch_idx, (images, labels) in enumerate(self.train_dataset):
            # Training step
            step_losses = self.train_step(images, labels)
            
            # Update metrics
            for key in epoch_losses:
                if key in step_losses:
                    epoch_losses[key] += float(step_losses[key])
            
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_losses['total_loss'] / (batch_idx + 1)
                self.logger.info(
                    f"Epoch[{epoch + 1}/{self.args.n_epoch}] "
                    f"Batch [{batch_idx + 1}] Loss: {avg_loss:.4f}"
                )
                
                # Reset running average for next 50 batches
                epoch_losses['total_loss'] = 0.0
            
            # TensorBoard logging
            if self.args.tboard and (batch_idx + 1) % 20 == 0:
                self.log_tensorboard_images(
                    step_losses.get('unwarp_pred'), 
                    step_losses.get('unwarp_gt'),
                    self.global_step, 'Train'
                )
                
                with self.summary_writer.as_default():
                    tf.summary.scalar('BM: L1 Loss/train', 
                                    epoch_losses['l1_loss'] / (batch_idx + 1), 
                                    step=self.global_step)
                    tf.summary.scalar('CB: Recon Loss/train', 
                                    epoch_losses['recon_loss'] / (batch_idx + 1), 
                                    step=self.global_step)
                    tf.summary.scalar('CB: SSIM Loss/train', 
                                    epoch_losses['ssim_loss'] / (batch_idx + 1), 
                                    step=self.global_step)
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.logger.info(f"Validating epoch {epoch + 1}")
        
        # Initialize metrics
        epoch_losses = {
            'l1_loss': 0.0,
            'recon_loss': 0.0,
            'ssim_loss': 0.0,
            'mse_loss': 0.0
        }
        
        num_batches = 0
        last_unwarp_pred = None
        last_unwarp_gt = None
        
        # Validation loop
        for batch_idx, (images, labels) in enumerate(tqdm(self.val_dataset, desc="Validation")):
            step_losses = self.val_step(images, labels)
            
            # Update metrics
            for key in epoch_losses:
                if key in step_losses:
                    epoch_losses[key] += float(step_losses[key])
            
            num_batches += 1
            
            # Keep last batch for visualization
            last_unwarp_pred = step_losses.get('unwarp_pred')
            last_unwarp_gt = step_losses.get('unwarp_gt')
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # TensorBoard logging
        if self.args.tboard:
            self.log_tensorboard_images(
                last_unwarp_pred, last_unwarp_gt,
                epoch + 1, 'Val'
            )
            
            with self.summary_writer.as_default():
                tf.summary.scalar('BM: L1 Loss/val', epoch_losses['l1_loss'], step=epoch + 1)
                tf.summary.scalar('CB: Recon Loss/val', epoch_losses['recon_loss'], step=epoch + 1)
                tf.summary.scalar('CB: SSIM Loss/val', epoch_losses['ssim_loss'], step=epoch + 1)
        
        return epoch_losses
    
    def save_checkpoint(self, epoch: int, train_losses: Dict[str, float], 
                       val_losses: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        # Create checkpoint filename
        suffix = "best_model" if is_best else "model"
        filename = (f"{self.args.arch}_{epoch + 1}_{val_losses['mse_loss']:.6f}_"
                   f"{train_losses['mse_loss']:.6f}_{self.experiment_name}_{suffix}.ckpt")
        
        checkpoint_path = os.path.join(self.args.logdir, filename)
        
        # Save checkpoint
        self.checkpoint.save(checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.epoch_start, self.args.n_epoch):
            # Training phase
            train_losses = self.train_epoch(epoch)
            
            # Log training results
            lr = self.get_learning_rate()
            self.logger.info(f"Training L1: {train_losses['l1_loss']:.6f}")
            self.logger.info(f"Training MSE: {train_losses['mse_loss']:.6f}")
            self.write_log_file(train_losses, epoch + 1, lr, 'Train')
            
            # Validation phase
            val_losses = self.validate_epoch(epoch)
            
            # Log validation results
            self.logger.info(f"Validation loss at epoch {epoch + 1}: {val_losses['l1_loss']:.6f}")
            self.logger.info(f"Validation MSE: {val_losses['mse_loss']:.6f}")
            self.write_log_file(val_losses, epoch + 1, lr, 'Val')
            
            # Learning rate scheduling
            # Create a dummy model-like object for the scheduler
            class DummyModel:
                def __init__(self, optimizer):
                    self.optimizer = optimizer
            
            dummy_model = DummyModel(self.optimizer)
            self.lr_scheduler.set_model(dummy_model)
            self.lr_scheduler.on_epoch_end(epoch, logs={'val_mse': val_losses['mse_loss']})
            
            # Save best model
            is_best = val_losses['mse_loss'] < self.best_val_mse
            if is_best:
                self.best_val_mse = val_losses['mse_loss']
                self.save_checkpoint(epoch, train_losses, val_losses, is_best=True)
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, train_losses, val_losses, is_best=False)
        
        self.logger.info("Training completed!")
        
        if self.summary_writer:
            self.summary_writer.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='TensorFlow Backward Mapping Training')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default='dnetccnl',
                       help='Architecture to use [dnetccnl, unetnc]')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Data path to load data')
    parser.add_argument('--img_rows', type=int, default=128,
                       help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=128,
                       help='Width of the input image')
    
    # Training arguments
    parser.add_argument('--n_epoch', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--l_rate', type=float, default=1e-5,
                       help='Learning rate')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--logdir', type=str, default='./checkpoints-bm/',
                       help='Path to store logs and checkpoints')
    
    # Visualization arguments
    parser.add_argument('--tboard', action='store_true',
                       help='Enable TensorBoard visualization')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = BackwardMappingTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()