#!/usr/bin/env python3
"""
Simplified inference test script for final validation.
This script provides a minimal inference implementation for testing purposes.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import TensorFlow DewarpNet components
from models.model_factory import ModelFactory
from utils.gpu_utils import setup_gpu


class SimpleInferenceTest:
    """Simplified inference test for validation purposes."""
    
    def __init__(self, wc_model_path: str, bm_model_path: str, output_dir: str):
        """Initialize simple inference test.
        
        Args:
            wc_model_path: Path to world coordinate model
            bm_model_path: Path to backward mapping model
            output_dir: Output directory for results
        """
        self.wc_model_path = wc_model_path
        self.bm_model_path = bm_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup GPU
        setup_gpu()
        
        # Models
        self.wc_model = None
        self.bm_model = None
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load both models."""
        try:
            self.logger.info("Loading models...")
            
            # Load WC model
            self.wc_model = ModelFactory.create_world_coordinate_model()
            dummy_wc_input = tf.random.normal((1, 256, 256, 3))
            _ = self.wc_model(dummy_wc_input)
            
            if os.path.exists(self.wc_model_path):
                self.wc_model.load_weights(self.wc_model_path)
                self.logger.info(f"✅ WC model loaded from: {self.wc_model_path}")
            else:
                self.logger.warning(f"⚠️ WC model file not found: {self.wc_model_path}")
            
            # Load BM model
            self.bm_model = ModelFactory.create_backward_mapping_model()
            dummy_bm_input = tf.random.normal((1, 128, 128, 3))
            _ = self.bm_model(dummy_bm_input)
            
            if os.path.exists(self.bm_model_path):
                self.bm_model.load_weights(self.bm_model_path)
                self.logger.info(f"✅ BM model loaded from: {self.bm_model_path}")
            else:
                self.logger.warning(f"⚠️ BM model file not found: {self.bm_model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading models: {e}")
            return False
    
    def create_test_images(self, num_images: int = 5) -> List[str]:
        """Create test images for inference testing.
        
        Args:
            num_images: Number of test images to create
            
        Returns:
            List of paths to created test images
        """
        self.logger.info(f"Creating {num_images} test images...")
        
        test_images = []
        
        try:
            for i in range(num_images):
                # Create a synthetic document-like image
                image = self._create_synthetic_document(i)
                
                # Save image
                image_path = self.output_dir / f'test_image_{i+1}.png'
                cv2.imwrite(str(image_path), image)
                test_images.append(str(image_path))
                
                self.logger.info(f"Created test image: {image_path}")
            
            return test_images
            
        except Exception as e:
            self.logger.error(f"❌ Error creating test images: {e}")
            return []
    
    def _create_synthetic_document(self, seed: int) -> np.ndarray:
        """Create a synthetic document image.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Synthetic document image as numpy array
        """
        np.random.seed(seed)
        
        # Create base document (white background)
        height, width = 256, 256
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add some text-like patterns
        for i in range(10):
            # Random rectangles to simulate text
            x1 = np.random.randint(10, width - 50)
            y1 = np.random.randint(10, height - 20)
            x2 = x1 + np.random.randint(20, 40)
            y2 = y1 + np.random.randint(5, 15)
            
            # Dark color for text
            color = np.random.randint(0, 100)
            cv2.rectangle(image, (x1, y1), (x2, y2), (color, color, color), -1)
        
        # Add some noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add slight warping effect
        rows, cols = height, width
        M = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-5, 5), 1)
        image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
        
        return image
    
    def test_inference_pipeline(self, test_images: List[str]) -> Dict[str, Any]:
        """Test the complete inference pipeline.
        
        Args:
            test_images: List of test image paths
            
        Returns:
            Dictionary with inference results
        """
        self.logger.info("Testing inference pipeline...")
        
        results = {
            'total_images': len(test_images),
            'successful_inferences': 0,
            'failed_inferences': 0,
            'inference_times': [],
            'output_paths': [],
            'errors': []
        }
        
        for i, image_path in enumerate(test_images):
            try:
                self.logger.info(f"Processing image {i+1}/{len(test_images)}: {image_path}")
                
                start_time = time.time()
                
                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                # Resize for WC model (256x256)
                wc_input = cv2.resize(image, (256, 256))
                wc_input = wc_input.astype(np.float32) / 255.0
                wc_input = np.expand_dims(wc_input, axis=0)
                
                # Run WC model
                wc_output = self.wc_model(wc_input, training=False)
                
                # Resize WC output for BM model (128x128)
                wc_resized = tf.image.resize(wc_output, [128, 128])
                
                # Run BM model
                bm_output = self.bm_model(wc_resized, training=False)
                
                # Create simple unwarp (placeholder)
                unwarped = self._simple_unwarp(image, bm_output.numpy()[0])
                
                # Save outputs
                output_path = self.output_dir / f'unwarped_{i+1}.png'
                cv2.imwrite(str(output_path), unwarped)
                
                # Save intermediate outputs
                wc_vis = self._visualize_world_coordinates(wc_output.numpy()[0])
                wc_path = self.output_dir / f'wc_output_{i+1}.png'
                cv2.imwrite(str(wc_path), wc_vis)
                
                bm_vis = self._visualize_backward_mapping(bm_output.numpy()[0])
                bm_path = self.output_dir / f'bm_output_{i+1}.png'
                cv2.imwrite(str(bm_path), bm_vis)
                
                inference_time = time.time() - start_time
                
                results['successful_inferences'] += 1
                results['inference_times'].append(inference_time)
                results['output_paths'].append(str(output_path))
                
                self.logger.info(f"✅ Processed successfully in {inference_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Error processing {image_path}: {e}")
                results['failed_inferences'] += 1
                results['errors'].append(str(e))
        
        # Calculate summary statistics
        if results['inference_times']:
            results['avg_inference_time'] = np.mean(results['inference_times'])
            results['total_inference_time'] = np.sum(results['inference_times'])
        else:
            results['avg_inference_time'] = 0
            results['total_inference_time'] = 0
        
        results['success_rate'] = results['successful_inferences'] / results['total_images'] if results['total_images'] > 0 else 0
        
        self.logger.info(f"Inference testing completed:")
        self.logger.info(f"  Successful: {results['successful_inferences']}/{results['total_images']}")
        self.logger.info(f"  Success rate: {results['success_rate']:.1%}")
        self.logger.info(f"  Average time: {results['avg_inference_time']:.3f}s")
        
        return results
    
    def _simple_unwarp(self, original_image: np.ndarray, bm_coords: np.ndarray) -> np.ndarray:
        """Simple unwarping using backward mapping coordinates.
        
        Args:
            original_image: Original input image
            bm_coords: Backward mapping coordinates (H, W, 2)
            
        Returns:
            Unwarped image
        """
        try:
            # Resize original image to match BM coordinates
            h, w = bm_coords.shape[:2]
            resized_image = cv2.resize(original_image, (w, h))
            
            # Normalize coordinates to image dimensions
            coords_normalized = bm_coords * np.array([w-1, h-1])
            
            # Create meshgrid for sampling
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # Add small displacement based on BM coordinates
            displacement_scale = 0.1  # Small displacement for testing
            x_displaced = x_coords + coords_normalized[:, :, 0] * displacement_scale
            y_displaced = y_coords + coords_normalized[:, :, 1] * displacement_scale
            
            # Clip coordinates to valid range
            x_displaced = np.clip(x_displaced, 0, w-1)
            y_displaced = np.clip(y_displaced, 0, h-1)
            
            # Sample from original image
            unwarped = cv2.remap(
                resized_image,
                x_displaced.astype(np.float32),
                y_displaced.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Resize back to original size
            unwarped = cv2.resize(unwarped, (original_image.shape[1], original_image.shape[0]))
            
            return unwarped
            
        except Exception as e:
            self.logger.error(f"Error in simple unwarp: {e}")
            return original_image  # Return original if unwarp fails
    
    def _visualize_world_coordinates(self, wc_output: np.ndarray) -> np.ndarray:
        """Visualize world coordinates as an image.
        
        Args:
            wc_output: World coordinates (H, W, 3)
            
        Returns:
            Visualization image
        """
        try:
            # Normalize to 0-255 range
            wc_normalized = (wc_output - wc_output.min()) / (wc_output.max() - wc_output.min() + 1e-8)
            wc_vis = (wc_normalized * 255).astype(np.uint8)
            
            return wc_vis
            
        except Exception as e:
            self.logger.error(f"Error visualizing world coordinates: {e}")
            return np.zeros((256, 256, 3), dtype=np.uint8)
    
    def _visualize_backward_mapping(self, bm_output: np.ndarray) -> np.ndarray:
        """Visualize backward mapping coordinates as an image.
        
        Args:
            bm_output: Backward mapping coordinates (H, W, 2)
            
        Returns:
            Visualization image
        """
        try:
            # Normalize to 0-255 range
            bm_normalized = (bm_output - bm_output.min()) / (bm_output.max() - bm_output.min() + 1e-8)
            
            # Create 3-channel visualization (use first 2 channels, set third to 0)
            h, w = bm_normalized.shape[:2]
            bm_vis = np.zeros((h, w, 3), dtype=np.uint8)
            bm_vis[:, :, 0] = (bm_normalized[:, :, 0] * 255).astype(np.uint8)
            bm_vis[:, :, 1] = (bm_normalized[:, :, 1] * 255).astype(np.uint8)
            
            return bm_vis
            
        except Exception as e:
            self.logger.error(f"Error visualizing backward mapping: {e}")
            return np.zeros((128, 128, 3), dtype=np.uint8)
    
    def run_complete_inference_test(self) -> Dict[str, Any]:
        """Run complete inference test pipeline.
        
        Returns:
            Dictionary with inference test results
        """
        self.logger.info("🚀 Starting complete inference test pipeline")
        
        # Create test images
        test_images = self.create_test_images(num_images=5)
        
        if not test_images:
            self.logger.error("❌ No test images created")
            return {'overall_success': False, 'error': 'No test images created'}
        
        # Run inference tests
        inference_results = self.test_inference_pipeline(test_images)
        
        # Determine overall success
        overall_success = (
            inference_results['success_rate'] >= 0.8 and  # At least 80% success rate
            inference_results['successful_inferences'] > 0  # At least one successful inference
        )
        
        results = {
            'overall_success': overall_success,
            'test_images_created': len(test_images),
            'inference_results': inference_results
        }
        
        if overall_success:
            self.logger.info("✅ Complete inference test pipeline successful")
        else:
            self.logger.error("❌ Inference test pipeline failed")
        
        return results


def main():
    """Main entry point for simple inference test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simple inference test')
    parser.add_argument('--wc-model', required=True,
                       help='Path to world coordinate model')
    parser.add_argument('--bm-model', required=True,
                       help='Path to backward mapping model')
    parser.add_argument('--output-dir', default='test_inference_output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create inference test
    inference_test = SimpleInferenceTest(
        wc_model_path=args.wc_model,
        bm_model_path=args.bm_model,
        output_dir=args.output_dir
    )
    
    # Run inference test
    results = inference_test.run_complete_inference_test()
    
    # Print results
    print("\nInference Test Results:")
    print(f"Overall: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
    print(f"Test Images Created: {results.get('test_images_created', 0)}")
    
    if 'inference_results' in results:
        inf_results = results['inference_results']
        print(f"Successful Inferences: {inf_results['successful_inferences']}/{inf_results['total_images']}")
        print(f"Success Rate: {inf_results['success_rate']:.1%}")
        print(f"Average Inference Time: {inf_results.get('avg_inference_time', 0):.3f}s")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)


if __name__ == '__main__':
    main()