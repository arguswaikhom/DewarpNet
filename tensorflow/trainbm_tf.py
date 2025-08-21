"""
TensorFlow training script for Backward Mapping (BM) regression.
Equivalent to PyTorch trainbm.py
"""

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("TensorFlow not installed. Please install it.")
    import sys
    sys.exit(1)

import os
import argparse
import numpy as np
from datetime import datetime
from typing import Optional

# Import TensorFlow implementations
from models_tf import get_model
from loaders_tf import get_loader_tf
from recon_lossc_tf import UnwarpLoss
from pytorch_ssim_tf import SSIM
from utils_tf import show_unwarp_tnsboard_tf, get_lr_tf


def write_log_file(log_file_name: str, losses: list, epoch: int, lrate: float, phase: str):
    """Write loss information to log file."""
    with open(log_file_name, 'a') as f:
        f.write(f"\n{phase} LRate: {lrate} Epoch: {epoch} Loss: {losses[0]} MSE: {losses[1]} "
                f"UnwarpL2: {losses[2]} UnwarpSSIMloss: {losses[3]}")


class BMTrainer:
    """Backward Mapping training class."""
    
    def __init__(self, args):
        self.args = args
        
        # Setup data loaders
        self.setup_data_loaders()
        
        # Setup model
        self.setup_model()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Setup losses
        self.setup_losses()
        
        # Setup logging
        self.setup_logging()
        
        # Setup checkpointing
        self.epoch_start = 0
        self.best_val_uwarpssim = 99999.0
        self.best_val_mse = 99999.0
        self.global_step = 0
        
        if args.resume:
            self.load_checkpoint()
    
    def setup_data_loaders(self):
        """Setup train and validation data loaders."""
        data_loader_factory = get_loader_tf('doc3dbmnic')
        
        self.train_dataset, self.val_dataset = data_loader_factory(
            root=self.args.data_path,
            batch_size=self.args.batch_size,
            img_size=(self.args.img_rows, self.args.img_cols),
            num_workers=8
        )
        
        # Calculate dataset sizes
        self.train_steps = tf.data.experimental.cardinality(self.train_dataset).numpy()
        self.val_steps = tf.data.experimental.cardinality(self.val_dataset).numpy()
        
    def setup_model(self):
        """Setup the DenseNet model."""
        self.model = get_model(
            arch=self.args.arch,
            n_classes=2,  # 2 output channels for backward mapping
            in_channels=3,  # 3 input channels (coordinate data from WC)
            img_size=self.args.img_rows
        )
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Adam optimizer with exact same parameters as PyTorch
        self.optimizer = keras.optimizers.Adam(
            learning_rate=self.args.l_rate,
            weight_decay=5e-4,
            amsgrad=True
        )
        
        # ReduceLROnPlateau equivalent
        self.lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,  # Different patience for BM training
            verbose=1,
            mode='min'
        )
    
    def setup_losses(self):
        """Setup loss functions."""
        self.mse_loss = keras.losses.MeanSquaredError()
        self.l1_loss = keras.losses.MeanAbsoluteError()
        self.unwarp_loss = UnwarpLoss()
    
    def setup_logging(self):
        """Setup logging and experiment tracking."""
        if not os.path.exists(self.args.logdir):
            os.makedirs(self.args.logdir)
        
        # Experiment name matching PyTorch version
        self.experiment_name = 'dnetccnl_htan_swat3dmini1kbm_l1_noaug_scratch'
        self.log_file_name = os.path.join(self.args.logdir, self.experiment_name + '.txt')
        
        if not os.path.isfile(self.log_file_name):
            with open(self.log_file_name, 'w') as f:
                f.write(f'\n---------------  {self.experiment_name}  ---------------\n')
        
        # Setup TensorBoard
        if self.args.tboard:
            log_dir = os.path.join("logs", self.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
    
    def load_checkpoint(self):
        """Load checkpoint for resuming training."""
        if os.path.isfile(self.args.resume):
            print(f"Loading model from checkpoint '{self.args.resume}'")
            try:
                self.model.load_weights(self.args.resume)
                # Extract epoch from filename if possible
                if 'epoch_' in self.args.resume:
                    self.epoch_start = int(self.args.resume.split('epoch_')[1].split('_')[0])
                print(f"Loaded checkpoint (epoch {self.epoch_start})")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"No checkpoint found at '{self.args.resume}'")
    
    @tf.function
    def train_step(self, images, labels):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Extract coordinate channels (last 3 channels of input)
            coords = images[:, :, :, 3:]  # Shape: (batch, 128, 128, 3)
            
            # Forward pass
            outputs = self.model(coords, training=True)
            
            # Transpose to match expected format (NCHW -> NHWC is already done)
            target_nhwc = outputs  # Already in NHWC format
            
            # Compute L1 loss
            l1_loss = self.l1_loss(labels, target_nhwc)
            
            # Compute reconstruction loss
            # Extract image channels (first 3 channels)
            img_channels = images[:, :, :, :3]
            
            # Combine image with coordinate channels for unwarp loss
            inp_combined = images[:, :, :, :-1]  # Remove last channel to get 6 channels total
            
            rloss, ssim_loss, uworg, uwpred = self.unwarp_loss(inp_combined, target_nhwc, labels)
            
            # Total loss (matching PyTorch version)
            total_loss = (10.0 * l1_loss) + (0.5 * rloss)
            
            # MSE for logging
            mse_loss = self.mse_loss(labels, target_nhwc)
        
        # Backward pass
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, l1_loss, rloss, ssim_loss, mse_loss, uworg, uwpred
    
    @tf.function
    def val_step(self, images, labels):
        """Single validation step."""
        # Extract coordinate channels
        coords = images[:, :, :, 3:]
        
        # Forward pass
        outputs = self.model(coords, training=False)
        target_nhwc = outputs
        
        # Compute losses
        l1_loss = self.l1_loss(labels, target_nhwc)
        
        # Compute reconstruction loss
        inp_combined = images[:, :, :, :-1]
        rloss, ssim_loss, uworg, uwpred = self.unwarp_loss(inp_combined, target_nhwc, labels)
        
        mse_loss = self.mse_loss(labels, target_nhwc)
        
        return l1_loss, rloss, ssim_loss, mse_loss, uworg, uwpred
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        # Reset metrics
        train_loss = keras.metrics.Mean()
        train_l1_loss = keras.metrics.Mean()
        train_rloss = keras.metrics.Mean()
        train_ssim_loss = keras.metrics.Mean()
        train_mse_loss = keras.metrics.Mean()
        
        # Training loop
        for batch_idx, (images, labels) in enumerate(self.train_dataset):
            total_loss, l1_loss, rloss, ssim_loss, mse_loss, uworg, uwpred = self.train_step(images, labels)
            
            # Update metrics
            train_loss.update_state(total_loss)
            train_l1_loss.update_state(l1_loss)
            train_rloss.update_state(rloss)
            train_ssim_loss.update_state(ssim_loss)
            train_mse_loss.update_state(mse_loss)
            
            self.global_step += 1
            
            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch[{epoch+1}/{self.args.n_epoch}] Batch [{batch_idx+1}/{self.train_steps}] "
                      f"Loss: {train_loss.result():.4f}")
            
            # TensorBoard logging
            if self.args.tboard and (batch_idx + 1) % 20 == 0:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('BM: L1 Loss/train', train_l1_loss.result(), step=self.global_step)
                    tf.summary.scalar('CB: Recon Loss/train', train_rloss.result(), step=self.global_step)
                    tf.summary.scalar('CB: SSIM Loss/train', train_ssim_loss.result(), step=self.global_step)
                    # Log unwarp images
                    show_unwarp_tnsboard_tf(self.global_step, uwpred, uworg, 8, 
                                           'Train GT unwarp', 'Train Pred Unwarp')
        
        return train_l1_loss.result(), train_mse_loss.result(), train_rloss.result(), train_ssim_loss.result()
    
    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        # Reset metrics
        val_l1_loss = keras.metrics.Mean()
        val_rloss = keras.metrics.Mean()
        val_ssim_loss = keras.metrics.Mean()
        val_mse_loss = keras.metrics.Mean()
        
        # Validation loop
        for batch_idx, (images, labels) in enumerate(self.val_dataset):
            l1_loss, rloss, ssim_loss, mse_loss, uworg, uwpred = self.val_step(images, labels)
            
            # Update metrics
            val_l1_loss.update_state(l1_loss)
            val_rloss.update_state(rloss)
            val_ssim_loss.update_state(ssim_loss)
            val_mse_loss.update_state(mse_loss)
        
        # TensorBoard logging
        if self.args.tboard:
            with self.val_summary_writer.as_default():
                tf.summary.scalar('BM: L1 Loss/val', val_l1_loss.result(), step=epoch+1)
                tf.summary.scalar('CB: Recon Loss/val', val_rloss.result(), step=epoch+1)
                tf.summary.scalar('CB: SSIM Loss/val', val_ssim_loss.result(), step=epoch+1)
                # Log unwarp images
                show_unwarp_tnsboard_tf(epoch+1, uwpred, uworg, 8,
                                       'Val GT unwarp', 'Val Pred Unwarp')
        
        return val_l1_loss.result(), val_mse_loss.result(), val_rloss.result(), val_ssim_loss.result()
    
    def save_checkpoint(self, epoch, val_mse, train_mse, is_best=False):
        """Save model checkpoint."""
        if is_best:
            checkpoint_path = os.path.join(
                self.args.logdir, 
                f"{self.args.arch}_{epoch+1}_{val_mse:.6f}_{train_mse:.6f}_{self.experiment_name}_best_model"
            )
        else:
            checkpoint_path = os.path.join(
                self.args.logdir,
                f"{self.args.arch}_{epoch+1}_{val_mse:.6f}_{train_mse:.6f}_{self.experiment_name}_model"
            )
        
        self.model.save_weights(checkpoint_path)
        
        # Also save optimizer state
        checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=tf.Variable(epoch + 1),
            val_mse=tf.Variable(val_mse),
            train_mse=tf.Variable(train_mse)
        )
        checkpoint.save(checkpoint_path + "_full")
    
    def train(self):
        """Main training loop."""
        # Setup TensorBoard writers
        if self.args.tboard:
            log_dir = f"logs/{self.experiment_name}"
            self.train_summary_writer = tf.summary.create_file_writer(f"{log_dir}/train")
            self.val_summary_writer = tf.summary.create_file_writer(f"{log_dir}/val")
        
        for epoch in range(self.epoch_start, self.args.n_epoch):
            print(f"\\nEpoch {epoch + 1}/{self.args.n_epoch}")
            
            # Training
            train_l1_loss, train_mse, train_rloss, train_ssim_loss = self.train_epoch(epoch)
            print(f"Training L1: {train_l1_loss:.4f}")
            print(f"Training MSE: {train_mse:.4f}")
            
            # Validation
            val_l1_loss, val_mse, val_rloss, val_ssim_loss = self.validate_epoch(epoch)
            print(f"Validation L1: {val_l1_loss:.4f}")
            print(f"Validation MSE: {val_mse:.4f}")
            
            # Log to file
            train_losses = [train_l1_loss, train_mse, train_rloss, train_ssim_loss]
            val_losses = [val_l1_loss, val_mse, val_rloss, val_ssim_loss]
            
            lrate = get_lr_tf(self.optimizer)
            write_log_file(self.log_file_name, train_losses, epoch + 1, lrate, 'Train')
            write_log_file(self.log_file_name, val_losses, epoch + 1, lrate, 'Val')
            
            # Update learning rate
            self.lr_scheduler.on_epoch_end(epoch, logs={'val_loss': val_mse})
            
            # Save best model
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.save_checkpoint(epoch, float(val_mse), float(train_mse), is_best=True)
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, float(val_mse), float(train_mse), is_best=False)


def main():
    parser = argparse.ArgumentParser(description='TensorFlow Backward Mapping Training')
    parser.add_argument('--arch', type=str, default='dnetccnl_tf', 
                        help='Architecture to use [dnetccnl_tf]')
    parser.add_argument('--data_path', type=str, default='./data/doc3d/', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', type=int, default=128, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=128, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', type=float, default=1e-4, 
                        help='Learning Rate')
    parser.add_argument('--resume', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', type=str, default='./checkpoints-bm-tf/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard')
    
    args = parser.parse_args()
    
    # Setup GPU
    print("Setting up GPU...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPUs found, using CPU")
    
    # Create trainer and start training
    trainer = BMTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()


# Example usage:
# python trainbm_tf.py --arch dnetccnl_tf --data_path ./data/doc3d/ --img_rows 128 --img_cols 128 --batch_size 50 --l_rate 0.0001 --tboard