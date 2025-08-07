#!/usr/bin/env python3
"""
Example usage script for TensorFlow DewarpNet inference.
Demonstrates how to use the inference pipeline and image utilities.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import DewarpNetInference, create_batch_processor, create_visualizer


def example_single_image_inference():
    """Example of processing a single image."""
    print("=== Single Image Inference Example ===")
    
    # Model paths (update these to your actual model paths)
    wc_model_path = "./eval/models/unetnc_doc3d_tf.ckpt"
    bm_model_path = "./eval/models/dnetccnl_doc3d_tf.ckpt"
    
    # Input and output paths
    input_image = "./eval/inp/sample.jpg"
    output_image = "./eval/uw/unwarped_sample.jpg"
    
    try:
        # Initialize inference pipeline
        inference = DewarpNetInference(
            wc_model_path=wc_model_path,
            bm_model_path=bm_model_path,
            device='auto'
        )
        
        # Process single image
        unwarped = inference.infer_single_image(
            image_path=input_image,
            output_path=output_image,
            show_result=True  # Display comparison
        )
        
        print(f"Successfully processed: {input_image}")
        print(f"Output saved to: {output_image}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def example_batch_inference():
    """Example of batch processing multiple images."""
    print("\n=== Batch Inference Example ===")
    
    # Model paths
    wc_model_path = "./eval/models/unetnc_doc3d_tf.ckpt"
    bm_model_path = "./eval/models/dnetccnl_doc3d_tf.ckpt"
    
    # Input and output directories
    input_dir = "./eval/inp/"
    output_dir = "./eval/uw/"
    
    try:
        # Initialize inference pipeline
        inference = DewarpNetInference(
            wc_model_path=wc_model_path,
            bm_model_path=bm_model_path,
            device='auto'
        )
        
        # Process batch of images
        stats = inference.infer_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            show_progress=True
        )
        
        print(f"\nBatch processing completed:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def example_image_utilities():
    """Example of using image processing utilities."""
    print("\n=== Image Utilities Example ===")
    
    # Create batch processor
    batch_processor = create_batch_processor(target_size=(256, 256))
    
    # Create visualizer
    visualizer = create_visualizer(figsize=(15, 10))
    
    input_dir = "./eval/inp/"
    
    try:
        # Load images from directory
        images = batch_processor.processor.batch_load_images(input_dir)
        
        if images:
            print(f"Loaded {len(images)} images")
            
            # Show first few images in a grid
            image_arrays = [img for _, img in images[:4]]
            image_names = [Path(path).name for path, _ in images[:4]]
            
            visualizer.show_grid(
                images=image_arrays,
                titles=image_names,
                save_path="./eval/image_grid.png"
            )
            
            print("Image grid saved to ./eval/image_grid.png")
        else:
            print("No images found in input directory")
            
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='DewarpNet TensorFlow Inference Examples',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--example', type=str, 
                       choices=['single', 'batch', 'utils', 'all'],
                       default='all',
                       help='Which example to run')
    
    args = parser.parse_args()
    
    print("DewarpNet TensorFlow Inference Examples")
    print("=" * 50)
    
    if args.example in ['single', 'all']:
        example_single_image_inference()
    
    if args.example in ['batch', 'all']:
        example_batch_inference()
    
    if args.example in ['utils', 'all']:
        example_image_utilities()
    
    print("\nExamples completed!")


if __name__ == '__main__':
    main()