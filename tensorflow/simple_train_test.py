#!/usr/bin/env python3
"""
Simple training test to validate the TensorFlow DewarpNet training pipeline.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_factory import ModelFactory
from loaders import Doc3DWCLoader, create_wc_dataset
from utils.gpu_utils import setup_gpu

def simple_training_test():
    """Run a simple training test."""
    print("=== Simple TensorFlow DewarpNet Training Test ===")
    
    # Setup GPU
    setup_gpu()
    
    # Create model
    print("Creating model...")
    model = ModelFactory.create_world_coordinate_model()
    
    # Build model with dummy input
    dummy_input = tf.random.normal((1, 256, 256, 3))
    _ = model(dummy_input)
    
    print(f"Model created with {model.count_params():,} parameters")
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Create loss function
    mse_loss = tf.keras.losses.MeanSquaredError()
    
    # Create dataset
    print("Creating dataset...")
    dataset = create_wc_dataset(
        root='tensorflow/data/doc3d_1000',
        split='train',
        batch_size=2,
        img_size=(256, 256),
        augmentations=False,
        shuffle=True
    )
    
    # Training loop
    print("Starting training...")
    num_epochs = 2
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataset.take(10), desc="Training")):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(images, training=True)
                
                # Compute loss
                loss = mse_loss(labels, predictions)
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += float(loss)
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: Loss = {float(loss):.6f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.6f}")
    
    print("\n✅ Simple training test completed successfully!")
    
    # Save model
    model_path = "tensorflow/checkpoints/simple_test_model"
    os.makedirs(model_path, exist_ok=True)
    model.save_weights(f"{model_path}/model_weights.h5")
    print(f"Model weights saved to: {model_path}/model_weights.h5")
    
    return model_path

if __name__ == '__main__':
    try:
        model_path = simple_training_test()
        print(f"\nTraining successful! Model saved at: {model_path}")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)