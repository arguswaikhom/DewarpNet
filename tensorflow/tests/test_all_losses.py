"""Simple test to verify all loss functions work together."""

import tensorflow as tf
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import (
    GradLoss, ReconLoss, UnwarpLoss, SSIMLoss,
    LossFactory, CombinedLoss, create_dewarpnet_loss
)

def test_all_losses():
    """Test that all loss functions can be created and used."""
    print("Testing TensorFlow DewarpNet Loss Functions...")
    
    # Create test data
    batch_size, height, width = 2, 32, 32
    
    # World coordinate data
    wc_true = tf.random.normal([batch_size, height, width, 3])
    wc_pred = tf.random.normal([batch_size, height, width, 3])
    
    # Backward mapping data
    inputs = tf.random.normal([batch_size, height, width, 6])
    bm_true = tf.random.uniform([batch_size, height, width, 2], -1, 1)
    bm_pred = tf.random.uniform([batch_size, height, width, 2], -1, 1)
    
    # Test 1: Gradient Loss
    print("1. Testing Gradient Loss...")
    grad_loss = GradLoss()
    grad_value = grad_loss(wc_true, wc_pred)
    print(f"   Gradient loss: {grad_value.numpy():.6f}")
    assert grad_value.numpy() > 0, "Gradient loss should be positive"
    
    # Test 2: SSIM Loss
    print("2. Testing SSIM Loss...")
    ssim_loss = SSIMLoss()
    ssim_value = ssim_loss(wc_true, wc_pred)
    print(f"   SSIM loss: {ssim_value.numpy():.6f}")
    assert 0 <= ssim_value.numpy() <= 2, "SSIM loss should be between 0 and 2"
    
    # Test 3: Reconstruction Loss
    print("3. Testing Reconstruction Loss...")
    recon_loss = ReconLoss()
    recon_result = recon_loss(inputs, bm_true, bm_pred)
    print(f"   Reconstruction loss: {recon_result['total_loss'].numpy():.6f}")
    print(f"   MSE component: {recon_result['mse_loss'].numpy():.6f}")
    print(f"   SSIM component: {recon_result['ssim_loss'].numpy():.6f}")
    assert recon_result['total_loss'].numpy() > 0, "Reconstruction loss should be positive"
    
    # Test 4: Unwarp Loss
    print("4. Testing Unwarp Loss...")
    unwarp_loss = UnwarpLoss()
    mse_loss, ssim_loss_val, uworg, uwpred = unwarp_loss(inputs, bm_pred, bm_true)
    print(f"   MSE loss: {mse_loss.numpy():.6f}")
    print(f"   SSIM loss: {ssim_loss_val.numpy():.6f}")
    assert mse_loss.numpy() > 0, "MSE loss should be positive"
    assert uworg.shape == (batch_size, height, width, 3), "Unwarped image shape incorrect"
    
    # Test 5: Loss Factory
    print("5. Testing Loss Factory...")
    factory = LossFactory()
    
    # Test creating different losses
    grad_loss_factory = factory.create_loss('grad_loss', window_size=5)
    recon_loss_factory = factory.create_loss('recon_loss', use_ssim=False)
    
    grad_value_factory = grad_loss_factory(wc_true, wc_pred)
    recon_result_factory = recon_loss_factory(inputs, bm_true, bm_pred)
    
    print(f"   Factory gradient loss: {grad_value_factory.numpy():.6f}")
    print(f"   Factory reconstruction loss: {recon_result_factory['total_loss'].numpy():.6f}")
    
    # Test 6: Combined Loss
    print("6. Testing Combined Loss...")
    combined_loss = create_dewarpnet_loss(grad_weight=1.0, recon_weight=0.5)
    print(f"   Created combined loss with weights: {combined_loss.weights}")
    
    # Test 7: List available losses
    print("7. Available Loss Functions:")
    available = factory.list_available_losses()
    for name, desc in available.items():
        print(f"   - {name}: {desc}")
    
    print("\n✅ All loss function tests passed!")
    print("🎉 TensorFlow DewarpNet loss functions are working correctly!")

if __name__ == '__main__':
    test_all_losses()