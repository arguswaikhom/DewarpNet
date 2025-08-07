"""
TensorFlow inference script for DewarpNet document unwarping.
Replicates the functionality of the PyTorch infer.py script.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import ModelFactory
from utils.gpu_utils import setup_gpu


class DewarpNetInference:
    """TensorFlow inference pipeline for DewarpNet."""
    
    def __init__(self, wc_model_path: str, bm_model_path: str, device: str = 'auto'):
        """
        Initialize the inference pipeline.
        
        Args:
            wc_model_path: Path to world coordinate model checkpoint
            bm_model_path: Path to backward mapping model checkpoint
            device: Device to use ('auto', 'cpu', 'gpu')
        """
        self.wc_model_path = wc_model_path
        self.bm_model_path = bm_model_path
        self.device = device
        
        # Setup device
        self._setup_device()
        
        # Model configurations
        self.wc_img_size = (256, 256)
        self.bm_img_size = (128, 128)
        
        # Load models
        self.wc_model = None
        self.bm_model = None
        self._load_models()
        
        print(f"DewarpNet inference pipeline initialized")
        print(f"World Coordinate Model: {wc_model_path}")
        print(f"Backward Mapping Model: {bm_model_path}")
        print(f"Device: {self.device}")
    
    def _setup_device(self):
        """Setup computation device."""
        if self.device == 'auto':
            # Auto-detect best available device
            if tf.config.list_physical_devices('GPU'):
                self.device = 'gpu'
                setup_gpu()
            else:
                self.device = 'cpu'
        elif self.device == 'gpu':
            if not tf.config.list_physical_devices('GPU'):
                print("Warning: GPU requested but not available, falling back to CPU")
                self.device = 'cpu'
            else:
                setup_gpu()
        
        print(f"Using device: {self.device}")
    
    def _load_models(self):
        """Load both world coordinate and backward mapping models."""
        try:
            # Load world coordinate model (UNet)
            print("Loading world coordinate model...")
            self.wc_model = ModelFactory.create_world_coordinate_model()
            
            # Build model with dummy input
            dummy_wc_input = tf.random.normal((1, *self.wc_img_size, 3))
            _ = self.wc_model(dummy_wc_input)
            
            # Load weights
            self.wc_model.load_weights(self.wc_model_path)
            print(f"World coordinate model loaded successfully")
            
            # Load backward mapping model (DenseNet)
            print("Loading backward mapping model...")
            self.bm_model = ModelFactory.create_backward_mapping_model()
            
            # Build model with dummy input
            dummy_bm_input = tf.random.normal((1, *self.bm_img_size, 3))
            _ = self.bm_model(dummy_bm_input)
            
            # Load weights
            self.bm_model.load_weights(self.bm_model_path)
            print(f"Backward mapping model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
    def preprocess_image(self, image_path: str) -> Tuple[tf.Tensor, np.ndarray]:
        """
        Preprocess input image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_image)
        """
        # Read image
        img_original = cv2.imread(image_path)
        if img_original is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        
        # Resize for world coordinate prediction
        img_resized = cv2.resize(img_original, self.wc_img_size)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = tf.convert_to_tensor(img_normalized)
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        
        return img_tensor, img_original
    
    def unwarp_image(self, original_image: np.ndarray, 
                    backward_mapping: tf.Tensor) -> np.ndarray:
        """
        Unwarp image using backward mapping coordinates.
        
        Args:
            original_image: Original input image
            backward_mapping: Backward mapping coordinates from model
            
        Returns:
            Unwarped image
        """
        # Get image dimensions
        h, w = original_image.shape[:2]
        
        # Process backward mapping
        # Convert from [batch, height, width, channels] to [height, width, channels]
        bm = backward_mapping[0].numpy()
        
        # Apply blur to smooth the mapping (matching PyTorch implementation)
        bm0 = cv2.blur(bm[:, :, 0], (3, 3))
        bm1 = cv2.blur(bm[:, :, 1], (3, 3))
        
        # Resize to original image size
        bm0 = cv2.resize(bm0, (w, h))
        bm1 = cv2.resize(bm1, (w, h))
        
        # Stack channels
        bm_resized = np.stack([bm0, bm1], axis=-1)
        
        # Convert to TensorFlow format for grid_sample
        # TensorFlow expects grid in range [-1, 1]
        # Assuming the backward mapping is already in the correct range
        bm_tensor = tf.convert_to_tensor(bm_resized, dtype=tf.float32)
        bm_tensor = tf.expand_dims(bm_tensor, 0)  # Add batch dimension
        
        # Prepare image for grid sampling
        img_normalized = original_image.astype(np.float32) / 255.0
        img_tensor = tf.convert_to_tensor(img_normalized)
        img_tensor = tf.transpose(img_tensor, [2, 0, 1])  # HWC to CHW
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        img_tensor = tf.transpose(img_tensor, [0, 2, 3, 1])  # NCHW to NHWC
        
        # Apply grid sampling
        unwarped = tf.nn.grid_sample(
            img_tensor, 
            bm_tensor, 
            method='bilinear',
            padding_mode='border'
        )
        
        # Convert back to numpy and proper format
        unwarped = unwarped[0].numpy()  # Remove batch dimension
        
        # Ensure values are in [0, 1] range
        unwarped = np.clip(unwarped, 0.0, 1.0)
        
        return unwarped
    
    @tf.function
    def predict_world_coordinates(self, image: tf.Tensor) -> tf.Tensor:
        """
        Predict world coordinates from RGB image.
        
        Args:
            image: Input RGB image tensor
            
        Returns:
            World coordinate predictions
        """
        # Forward pass through world coordinate model
        wc_output = self.wc_model(image, training=False)
        
        # Apply hard tanh activation (0, 1) as in PyTorch version
        wc_pred = tf.nn.tanh(wc_output)  # First tanh to [-1, 1]
        wc_pred = tf.clip_by_value(wc_pred, 0.0, 1.0)  # Then clip to [0, 1]
        
        return wc_pred
    
    @tf.function
    def predict_backward_mapping(self, world_coords: tf.Tensor) -> tf.Tensor:
        """
        Predict backward mapping from world coordinates.
        
        Args:
            world_coords: World coordinate tensor
            
        Returns:
            Backward mapping predictions
        """
        # Resize world coordinates to backward mapping input size
        bm_input = tf.image.resize(world_coords, self.bm_img_size)
        
        # Forward pass through backward mapping model
        bm_output = self.bm_model(bm_input, training=False)
        
        return bm_output
    
    def infer_single_image(self, image_path: str, 
                          output_path: Optional[str] = None,
                          show_result: bool = False) -> np.ndarray:
        """
        Perform inference on a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            show_result: Whether to display the result
            
        Returns:
            Unwarped image as numpy array
        """
        print(f"Processing image: {image_path}")
        
        # Preprocess image
        img_tensor, img_original = self.preprocess_image(image_path)
        
        # Predict world coordinates
        wc_pred = self.predict_world_coordinates(img_tensor)
        
        # Predict backward mapping
        bm_pred = self.predict_backward_mapping(wc_pred)
        
        # Unwarp image
        unwarped = self.unwarp_image(img_original, bm_pred)
        
        # Save output if path provided
        if output_path:
            # Convert to uint8 and BGR for OpenCV
            output_img = (unwarped * 255).astype(np.uint8)
            output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output_img_bgr)
            print(f"Output saved to: {output_path}")
        
        # Show result if requested
        if show_result:
            self._show_comparison(img_original, unwarped)
        
        return unwarped
    
    def infer_batch(self, input_dir: str, output_dir: str, 
                   show_progress: bool = True) -> Dict[str, Any]:
        """
        Perform inference on a batch of images.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with processing statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in Path(input_dir).iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")
        
        print(f"Found {len(image_files)} images to process")
        
        # Process images
        processed_count = 0
        failed_count = 0
        failed_files = []
        
        for i, image_path in enumerate(image_files):
            try:
                # Generate output path
                output_filename = f"unwarped_{image_path.name}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Process image
                self.infer_single_image(str(image_path), output_path)
                processed_count += 1
                
                if show_progress:
                    print(f"Progress: {i+1}/{len(image_files)} ({processed_count} successful)")
                
            except Exception as e:
                print(f"Failed to process {image_path}: {str(e)}")
                failed_count += 1
                failed_files.append(str(image_path))
        
        # Return statistics
        stats = {
            'total_files': len(image_files),
            'processed': processed_count,
            'failed': failed_count,
            'failed_files': failed_files,
            'success_rate': processed_count / len(image_files) if image_files else 0
        }
        
        print(f"\nBatch processing complete:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        
        return stats
    
    def _show_comparison(self, original: np.ndarray, unwarped: np.ndarray):
        """
        Show side-by-side comparison of original and unwarped images.
        
        Args:
            original: Original image
            unwarped: Unwarped image
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(unwarped)
        axes[1].set_title('Unwarped Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'world_coordinate_model': {
                'path': self.wc_model_path,
                'input_shape': self.wc_model.input_shape if self.wc_model else None,
                'output_shape': self.wc_model.output_shape if self.wc_model else None,
                'parameters': self.wc_model.count_params() if self.wc_model else None
            },
            'backward_mapping_model': {
                'path': self.bm_model_path,
                'input_shape': self.bm_model.input_shape if self.bm_model else None,
                'output_shape': self.bm_model.output_shape if self.bm_model else None,
                'parameters': self.bm_model.count_params() if self.bm_model else None
            },
            'device': self.device,
            'image_sizes': {
                'world_coordinate': self.wc_img_size,
                'backward_mapping': self.bm_img_size
            }
        }
        
        return info


def main():
    """Main inference function with command line interface."""
    parser = argparse.ArgumentParser(
        description='TensorFlow DewarpNet Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument('--wc_model_path', type=str, required=True,
                       help='Path to the world coordinate model checkpoint')
    parser.add_argument('--bm_model_path', type=str, required=True,
                       help='Path to the backward mapping model checkpoint')
    
    # Input/Output paths
    parser.add_argument('--img_path', type=str, default='./eval/inp/',
                       help='Path to input image or directory')
    parser.add_argument('--out_path', type=str, default='./eval/uw/',
                       help='Path to output directory')
    
    # Options
    parser.add_argument('--show', action='store_true',
                       help='Show input and output images')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'gpu'],
                       help='Device to use for inference')
    parser.add_argument('--batch', action='store_true',
                       help='Process all images in input directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference pipeline
        inference = DewarpNetInference(
            wc_model_path=args.wc_model_path,
            bm_model_path=args.bm_model_path,
            device=args.device
        )
        
        # Print model information
        if args.show:
            model_info = inference.get_model_info()
            print("\nModel Information:")
            print(f"WC Model Parameters: {model_info['world_coordinate_model']['parameters']:,}")
            print(f"BM Model Parameters: {model_info['backward_mapping_model']['parameters']:,}")
        
        # Process images
        if args.batch or os.path.isdir(args.img_path):
            # Batch processing
            stats = inference.infer_batch(
                input_dir=args.img_path,
                output_dir=args.out_path,
                show_progress=True
            )
            
        else:
            # Single image processing
            if not os.path.isfile(args.img_path):
                raise FileNotFoundError(f"Image file not found: {args.img_path}")
            
            # Generate output filename
            input_filename = os.path.basename(args.img_path)
            output_filename = f"unwarped_{input_filename}"
            output_path = os.path.join(args.out_path, output_filename)
            
            # Create output directory
            os.makedirs(args.out_path, exist_ok=True)
            
            # Process single image
            unwarped = inference.infer_single_image(
                image_path=args.img_path,
                output_path=output_path,
                show_result=args.show
            )
            
            print(f"Successfully processed: {args.img_path}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())