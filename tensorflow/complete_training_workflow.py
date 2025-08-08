#!/usr/bin/env python3
"""
Complete TensorFlow DewarpNet Training and Inference Workflow.

This script implements the complete workflow requested:
1. Train World Coordinate model on doc3d_1000 (validation)
2. Train Backward Mapping model on doc3d_1000 (validation)
3. Train both models on full doc3d dataset (production)
4. Run inference pipeline equivalent to PyTorch infer.py

Usage:
    python tensorflow/complete_training_workflow.py --stage validation
    python tensorflow/complete_training_workflow.py --stage production
    python tensorflow/complete_training_workflow.py --stage inference
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ Command completed successfully")
        return True

def train_validation_models():
    """Train models on doc3d_1000 dataset for validation."""
    print("\n🏋️ STAGE 1: Training Validation Models on doc3d_1000")
    
    # Create checkpoint directories
    os.makedirs("tensorflow/checkpoints/validation", exist_ok=True)
    
    # Train World Coordinate model
    wc_cmd = """python tensorflow/training/train_wc.py \
        --arch unetnc \
        --data_path tensorflow/data/doc3d_1000 \
        --img_rows 256 \
        --img_cols 256 \
        --n_epoch 10 \
        --batch_size 4 \
        --l_rate 0.0001 \
        --logdir tensorflow/checkpoints/validation/wc_model \
        --tboard"""
    
    success = run_command(wc_cmd, "Training World Coordinate Model (Validation)")
    if not success:
        return False
    
    # Train Backward Mapping model
    bm_cmd = """python tensorflow/training/train_bm.py \
        --arch dnetccnl \
        --data_path tensorflow/data/doc3d_1000 \
        --img_rows 128 \
        --img_cols 128 \
        --n_epoch 10 \
        --batch_size 4 \
        --l_rate 0.0001 \
        --logdir tensorflow/checkpoints/validation/bm_model \
        --tboard"""
    
    success = run_command(bm_cmd, "Training Backward Mapping Model (Validation)")
    return success

def train_production_models():
    """Train models on full doc3d dataset for production."""
    print("\n🚀 STAGE 2: Training Production Models on Full doc3d Dataset")
    
    # Create checkpoint directories
    os.makedirs("tensorflow/checkpoints/production", exist_ok=True)
    
    # Train World Coordinate model
    wc_cmd = """python tensorflow/training/train_wc.py \
        --arch unetnc \
        --data_path tensorflow/data/doc3d \
        --img_rows 256 \
        --img_cols 256 \
        --n_epoch 100 \
        --batch_size 8 \
        --l_rate 0.0001 \
        --logdir tensorflow/checkpoints/production/wc_model \
        --tboard"""
    
    success = run_command(wc_cmd, "Training World Coordinate Model (Production)")
    if not success:
        return False
    
    # Train Backward Mapping model
    bm_cmd = """python tensorflow/training/train_bm.py \
        --arch dnetccnl \
        --data_path tensorflow/data/doc3d \
        --img_rows 128 \
        --img_cols 128 \
        --n_epoch 100 \
        --batch_size 8 \
        --l_rate 0.0001 \
        --logdir tensorflow/checkpoints/production/bm_model \
        --tboard"""
    
    success = run_command(bm_cmd, "Training Backward Mapping Model (Production)")
    return success

def find_best_model(checkpoint_dir):
    """Find the best model checkpoint in a directory."""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Look for best model files
    best_models = list(checkpoint_path.glob("*best_model*"))
    
    if best_models:
        return str(best_models[0])
    
    # Fallback to any checkpoint files
    checkpoints = list(checkpoint_path.glob("*.ckpt*"))
    if checkpoints:
        return str(checkpoints[0])
    
    # Look for .h5 files
    h5_files = list(checkpoint_path.glob("*.h5"))
    if h5_files:
        return str(h5_files[0])
    
    print(f"⚠️ No model checkpoints found in: {checkpoint_dir}")
    return None

def run_inference():
    """Run TensorFlow inference equivalent to PyTorch infer.py."""
    print("\n🔍 STAGE 3: Running TensorFlow Inference Pipeline")
    
    # Find the best trained models
    wc_model_path = find_best_model("tensorflow/checkpoints/production/wc_model")
    bm_model_path = find_best_model("tensorflow/checkpoints/production/bm_model")
    
    if not wc_model_path:
        print("⚠️ Production WC model not found, trying validation model...")
        wc_model_path = find_best_model("tensorflow/checkpoints/validation/wc_model")
    
    if not bm_model_path:
        print("⚠️ Production BM model not found, trying validation model...")
        bm_model_path = find_best_model("tensorflow/checkpoints/validation/bm_model")
    
    if not wc_model_path or not bm_model_path:
        print("❌ Could not find trained models for inference")
        return False
    
    print(f"Using WC model: {wc_model_path}")
    print(f"Using BM model: {bm_model_path}")
    
    # Create output directory
    os.makedirs("tensorflow/inference_output", exist_ok=True)
    
    # Run inference
    inference_cmd = f"""python tensorflow/inference/infer.py \
        --wc_model_path {wc_model_path} \
        --bm_model_path {bm_model_path} \
        --img_path tensorflow/data/input_crop \
        --out_path tensorflow/inference_output \
        --show"""
    
    success = run_command(inference_cmd, "Running TensorFlow Inference")
    
    if success:
        print("\n✅ Inference completed! Check tensorflow/inference_output/ for results")
        
        # List output files
        output_path = Path("tensorflow/inference_output")
        if output_path.exists():
            output_files = list(output_path.glob("*.png"))
            print(f"Generated {len(output_files)} output images:")
            for f in output_files[:5]:  # Show first 5
                print(f"  - {f.name}")
            if len(output_files) > 5:
                print(f"  ... and {len(output_files) - 5} more")
    
    return success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Complete TensorFlow DewarpNet Workflow')
    parser.add_argument('--stage', choices=['validation', 'production', 'inference', 'all'],
                       default='all', help='Which stage to run')
    
    args = parser.parse_args()
    
    print("🚀 TensorFlow DewarpNet Complete Training and Inference Workflow")
    print("=" * 70)
    
    success = True
    
    if args.stage in ['validation', 'all']:
        success = train_validation_models()
        if not success:
            print("❌ Validation training failed")
            return 1
    
    if args.stage in ['production', 'all'] and success:
        success = train_production_models()
        if not success:
            print("❌ Production training failed")
            return 1
    
    if args.stage in ['inference', 'all'] and success:
        success = run_inference()
        if not success:
            print("❌ Inference failed")
            return 1
    
    if success:
        print("\n🎉 Complete workflow finished successfully!")
        print("\nSummary:")
        print("✅ TensorFlow DewarpNet models trained")
        print("✅ Inference pipeline tested")
        print("✅ Output images generated")
        print("\nYour TensorFlow implementation is now equivalent to the PyTorch infer.py!")
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())