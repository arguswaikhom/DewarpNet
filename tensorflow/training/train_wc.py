"""
TensorFlow implementation of world coordinate training script.
Replicates the PyTorch trainwc.py logic for training world coordinate regression from RGB images.
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
from loaders import Doc3DWCLoader, create_wc_dataset
from losses.loss_factory import LossFactory, LossLogger
from utils.gpu_utils import setup_gpu, get_gpu_info


class WorldCoordinateTrainer:
    """Trainer class for world coordinate regression model."""
    
    def __init__(self, args):
        """Initialize trainer with configuration."""
        self.args = args
        self.setup_logging()
        self.setup_gpu()
        self.setup_directories()
        
        # Training state
        self.global_step = 0
        self.best_val_mse = float('inf')
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
        self.experiment_name = 'htan_doc3d_l1grad_bghsaugk_scratch'
        self.logger.info(f"Starting experiment: {self.experiment_name}")
    
    def setup_gpu(self):
        """Setup GPU configuration."""
        setup_gpu()  # Setup GPU configuration
        self.gpu_info = get_gpu_info()  # Get GPU info
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
        self.train_loader = Doc3DWCLoader(
            root=self.args.data_path,
            split='train',
            img_size=(self.args.img_rows, self.args.img_cols),
            augmentations=True
        )
        
        # Validation loader
        self.val_loader = Doc3DWCLoader(
            root=self.args.data_path,
            split='val',
            img_size=(self.args.img_rows, self.args.img_cols),
            augmentations=False
        )
        
        # Create TensorFlow datasets
        self.train_dataset = create_wc_dataset(
            root=self.args.data_path,
            split='train',
            batch_size=self.args.batch_size,
            img_size=(self.args.img_rows, self.args.img_cols),
            augmentations=True,
            shuffle=True,
            num_parallel_calls=8,
            prefetch_buffer=tf.data.AUTOTUNE
        )
        
        self.val_dataset = create_wc_dataset(
            root=self.args.data_path,
            split='val',
            batch_size=self.args.batch_size,
            img_size=(self.args.img_rows, self.args.img_cols),
            augmentations=False,
            shuffle=False,
            num_parallel_calls=8,
            prefetch_buffer=tf.data.AUTOTUNE
        )
        
        self.logger.info(f"Training samples: {len(self.train_loader)}")
        self.logger.info(f"Validation samples: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup the world coordinate regression model."""
        self.logger.info("Setting up model...")
        
        # Create model using factory
        self.model = ModelFactory.create_world_coordinate_model(
            input_nc=3,
            output_nc=3,
            num_downs=7,
            ngf=64,
            use_dropout=False
        )
        
        # Build model with dummy input
        dummy_input = tf.random.normal((1, self.args.img_rows, self.args.img_cols, 3))
        _ = self.model(dummy_input)
        

        
        self.logger.info(f"Model created with {self.model.count_params()} parameters")
        
        # Hardtanh activation (equivalent to PyTorch version)
        self.activation = lambda x: tf.clip_by_value(x, 0.0, 1.0)
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.logger.info("Setting up optimizer...")
        
        # Adam optimizer with same parameters as PyTorch
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.args.l_rate,
            weight_decay=5e-4,
            amsgrad=True
        )
        
        # Custom learning rate scheduler to avoid model.optimizer dependency
        self.lr_scheduler_patience = 5
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_min_lr = 1e-8
        self.lr_scheduler_wait = 0
        self.lr_scheduler_best = float('inf')
    
    def setup_loss_functions(self):
        """Setup loss functions."""
        self.logger.info("Setting up loss functions...")
        
        # Create loss functions
        self.loss_factory = LossFactory()
        
        # Gradient loss (matching PyTorch implementation)
        self.grad_loss_fn = self.loss_factory.create_loss(
            'grad_loss',
            window_size=5,
            padding='SAME'
        )
        
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
    
    def update_learning_rate(self, val_loss):
        """Custom learning rate scheduling similar to ReduceLROnPlateau."""
        if val_loss < self.lr_scheduler_best:
            self.lr_scheduler_best = val_loss
            self.lr_scheduler_wait = 0
        else:
            self.lr_scheduler_wait += 1
            
        if self.lr_scheduler_wait >= self.lr_scheduler_patience:
            current_lr = float(self.optimizer.learning_rate.numpy())
            new_lr = max(current_lr * self.lr_scheduler_factor, self.lr_scheduler_min_lr)
            
            if new_lr < current_lr:
                self.optimizer.learning_rate.assign(new_lr)
                self.logger.info(f"Reducing learning rate from {current_lr:.2e} to {new_lr:.2e}")
                self.lr_scheduler_wait = 0
    
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
                   f"GradLoss: {losses['grad_loss']:.6f}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.learning_rate.numpy())
    
    @tf.function
    def train_step(self, images: tf.Tensor, labels: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single training step."""
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = self.model(images, training=True)
            pred = self.activation(outputs)
            
            # Compute losses
            grad_loss = self.grad_loss_fn(pred, labels)
            l1_loss = self.l1_loss_fn(pred, labels)
            mse_loss = self.mse_loss_fn(pred, labels)
            
            # Total loss (matching PyTorch: only L1 loss, grad loss commented out)
            total_loss = l1_loss  # + (0.2 * grad_loss)
            
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
            'grad_loss': grad_loss,
            'mse_loss': mse_loss,
            'predictions': pred
        }
    
    @tf.function
    def val_step(self, images: tf.Tensor, labels: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single validation step."""
        # Forward pass
        outputs = self.model(images, training=False)
        pred = self.activation(outputs)
        
        # Compute losses
        grad_loss = self.grad_loss_fn(pred, labels)
        l1_loss = self.l1_loss_fn(pred, labels)
        mse_loss = self.mse_loss_fn(pred, labels)
        
        return {
            'l1_loss': l1_loss,
            'grad_loss': grad_loss,
            'mse_loss': mse_loss,
            'predictions': pred
        }
    
    def log_tensorboard_images(self, images: tf.Tensor, labels: tf.Tensor, 
                              predictions: tf.Tensor, step: int, phase: str):
        """Log images to TensorBoard."""
        if self.summary_writer is None:
            return
        
        # Select random samples for visualization
        batch_size = tf.shape(images)[0]
        num_samples = min(8, batch_size)
        indices = tf.random.shuffle(tf.range(batch_size))[:num_samples]
        
        sample_images = tf.gather(images, indices)
        sample_labels = tf.gather(labels, indices)
        sample_preds = tf.gather(predictions, indices)
        
        with self.summary_writer.as_default():
            # Log input images
            tf.summary.image(
                f'{phase} Inputs',
                sample_images,
                step=step,
                max_outputs=num_samples
            )
            
            # Log ground truth world coordinates
            tf.summary.image(
                f'{phase} WCs',
                sample_labels,
                step=step,
                max_outputs=num_samples
            )
            
            # Log predicted world coordinates
            tf.summary.image(
                f'{phase} Pred. WCs',
                sample_preds,
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
            'grad_loss': 0.0,
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
                    images, labels, step_losses['predictions'],
                    self.global_step, 'Train'
                )
                
                with self.summary_writer.as_default():
                    tf.summary.scalar('WC: L1 Loss/train', 
                                    epoch_losses['l1_loss'] / (batch_idx + 1), 
                                    step=self.global_step)
                    tf.summary.scalar('WC: Grad Loss/train', 
                                    epoch_losses['grad_loss'] / (batch_idx + 1), 
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
            'grad_loss': 0.0,
            'mse_loss': 0.0
        }
        
        num_batches = 0
        last_predictions = None
        last_images = None
        last_labels = None
        
        # Validation loop
        for batch_idx, (images, labels) in enumerate(tqdm(self.val_dataset, desc="Validation")):
            step_losses = self.val_step(images, labels)
            
            # Update metrics
            for key in epoch_losses:
                if key in step_losses:
                    epoch_losses[key] += float(step_losses[key])
            
            num_batches += 1
            
            # Keep last batch for visualization
            last_predictions = step_losses['predictions']
            last_images = images
            last_labels = labels
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # TensorBoard logging
        if self.args.tboard and last_predictions is not None:
            self.log_tensorboard_images(
                last_images, last_labels, last_predictions,
                epoch + 1, 'Val'
            )
            
            with self.summary_writer.as_default():
                tf.summary.scalar('WC: L1 Loss/val', epoch_losses['l1_loss'], step=epoch + 1)
                tf.summary.scalar('WC: Grad Loss/val', epoch_losses['grad_loss'], step=epoch + 1)
        
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
            
            # Custom learning rate scheduling
            self.update_learning_rate(val_losses['mse_loss'])
            
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
    parser = argparse.ArgumentParser(description='TensorFlow World Coordinate Training')
    
    # Model arguments
    parser.add_argument('--arch', type=str, default='unetnc',
                       help='Architecture to use [unetnc, dnetccnl]')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Data path to load data')
    parser.add_argument('--img_rows', type=int, default=256,
                       help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=256,
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
    parser.add_argument('--logdir', type=str, default='./checkpoints-wc/',
                       help='Path to store logs and checkpoints')
    
    # Visualization arguments
    parser.add_argument('--tboard', action='store_true',
                       help='Enable TensorBoard visualization')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = WorldCoordinateTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()