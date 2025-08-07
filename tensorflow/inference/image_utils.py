"""
Image processing utilities for DewarpNet TensorFlow inference.
Provides functions for loading, preprocessing, postprocessing, and visualization.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import logging


class ImageProcessor:
    """Comprehensive image processing utilities for DewarpNet inference."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target size for image processing (width, height)
        """
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
    
    def load_image(self, image_path: str, 
                   color_mode: str = 'rgb') -> np.ndarray:
        """
        Load image from file with error handling.
        
        Args:
            image_path: Path to image file
            color_mode: Color mode ('rgb', 'bgr', 'grayscale')
            
        Returns:
            Loaded image as numpy array
            
        Raises:
            ValueError: If image cannot be loaded or invalid color mode
        """
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        try:
            # Load image using OpenCV
            if color_mode.lower() == 'grayscale':
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                
                if color_mode.lower() == 'rgb':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif color_mode.lower() != 'bgr':
                    raise ValueError(f"Unsupported color mode: {color_mode}")
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise ValueError(f"Could not load image {image_path}: {str(e)}")    
 
   def preprocess_image(self, image: np.ndarray, 
                        normalize: bool = True,
                        resize: bool = True) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array
            normalize: Whether to normalize pixel values to [0, 1]
            resize: Whether to resize to target size
            
        Returns:
            Tuple of (preprocessed_tensor, metadata)
        """
        metadata = {
            'original_shape': image.shape,
            'original_dtype': image.dtype,
            'normalized': normalize,
            'resized': resize
        }
        
        processed_image = image.copy()
        
        # Resize if requested
        if resize and processed_image.shape[:2] != self.target_size[::-1]:
            processed_image = cv2.resize(processed_image, self.target_size)
            metadata['resized_shape'] = processed_image.shape
        
        # Normalize if requested
        if normalize:
            if processed_image.dtype == np.uint8:
                processed_image = processed_image.astype(np.float32) / 255.0
            elif processed_image.dtype == np.uint16:
                processed_image = processed_image.astype(np.float32) / 65535.0
            else:
                # Assume already normalized or handle other dtypes
                processed_image = processed_image.astype(np.float32)
        
        # Convert to tensor and add batch dimension
        tensor = tf.convert_to_tensor(processed_image)
        if len(tensor.shape) == 2:  # Grayscale
            tensor = tf.expand_dims(tensor, -1)
        tensor = tf.expand_dims(tensor, 0)  # Add batch dimension
        
        return tensor, metadata
    
    def postprocess_image(self, tensor: tf.Tensor, 
                         metadata: Dict[str, Any],
                         restore_original_size: bool = True) -> np.ndarray:
        """
        Postprocess model output back to image format.
        
        Args:
            tensor: Model output tensor
            metadata: Metadata from preprocessing
            restore_original_size: Whether to restore original image size
            
        Returns:
            Postprocessed image as numpy array
        """
        # Remove batch dimension
        if len(tensor.shape) == 4:
            image = tensor[0].numpy()
        else:
            image = tensor.numpy()
        
        # Ensure values are in valid range
        image = np.clip(image, 0.0, 1.0)
        
        # Restore original size if requested
        if restore_original_size and 'original_shape' in metadata:
            original_h, original_w = metadata['original_shape'][:2]
            if image.shape[:2] != (original_h, original_w):
                image = cv2.resize(image, (original_w, original_h))
        
        # Convert back to uint8 if original was uint8
        if metadata.get('original_dtype') == np.uint8:
            image = (image * 255).astype(np.uint8)
        
        return image    

    def save_image(self, image: np.ndarray, output_path: str,
                   quality: int = 95, format_hint: Optional[str] = None) -> bool:
        """
        Save image to file with format conversion and error handling.
        
        Args:
            image: Image to save as numpy array
            output_path: Path to save the image
            quality: JPEG quality (0-100)
            format_hint: Format hint ('jpg', 'png', etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Determine format from extension or hint
            if format_hint:
                ext = f".{format_hint.lower()}"
            else:
                ext = Path(output_path).suffix.lower()
            
            # Ensure image is in correct format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image - convert to BGR for OpenCV
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            # Save with appropriate parameters
            if ext in ['.jpg', '.jpeg']:
                success = cv2.imwrite(output_path, save_image, 
                                    [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext == '.png':
                success = cv2.imwrite(output_path, save_image,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                success = cv2.imwrite(output_path, save_image)
            
            if success:
                self.logger.info(f"Image saved successfully: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving image {output_path}: {str(e)}")
            return False
    
    def batch_load_images(self, input_dir: str, 
                         recursive: bool = False) -> List[Tuple[str, np.ndarray]]:
        """
        Load all images from a directory.
        
        Args:
            input_dir: Directory containing images
            recursive: Whether to search subdirectories
            
        Returns:
            List of (filepath, image) tuples
        """
        images = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in Path(input_dir).glob(pattern):
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    image = self.load_image(str(file_path))
                    images.append((str(file_path), image))
                except Exception as e:
                    self.logger.warning(f"Skipping {file_path}: {str(e)}")
        
        return images    

    def validate_image_format(self, image_path: str) -> Dict[str, Any]:
        """
        Validate image format and get metadata.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            'valid': False,
            'error': None,
            'format': None,
            'size': None,
            'channels': None,
            'dtype': None
        }
        
        try:
            # Check file existence
            if not os.path.exists(image_path):
                result['error'] = "File not found"
                return result
            
            # Check file extension
            ext = Path(image_path).suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                result['error'] = f"Unsupported format: {ext}"
                return result
            
            # Try to load image
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                result['error'] = "Could not decode image"
                return result
            
            # Extract metadata
            result.update({
                'valid': True,
                'format': ext,
                'size': (image.shape[1], image.shape[0]),  # (width, height)
                'channels': image.shape[2] if len(image.shape) == 3 else 1,
                'dtype': str(image.dtype)
            })
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


class ImageVisualizer:
    """Utilities for visualizing images and results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def show_comparison(self, original: np.ndarray, processed: np.ndarray,
                       titles: Optional[List[str]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Show side-by-side comparison of two images.
        
        Args:
            original: Original image
            processed: Processed image
            titles: Titles for the images
            save_path: Path to save the comparison plot
        """
        if titles is None:
            titles = ['Original', 'Processed']
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        axes[0].imshow(original)
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        
        axes[1].imshow(processed)
        axes[1].set_title(titles[1])
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show() 
   
    def show_grid(self, images: List[np.ndarray], 
                  titles: Optional[List[str]] = None,
                  grid_size: Optional[Tuple[int, int]] = None,
                  save_path: Optional[str] = None) -> None:
        """
        Show multiple images in a grid layout.
        
        Args:
            images: List of images to display
            titles: List of titles for each image
            grid_size: Grid size (rows, cols). Auto-calculated if None
            save_path: Path to save the grid plot
        """
        n_images = len(images)
        
        if grid_size is None:
            # Auto-calculate grid size
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        
        # Handle single row/column cases
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(n_images):
            axes[i].imshow(images[i])
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def create_before_after_montage(self, image_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Create a montage showing before/after pairs.
        
        Args:
            image_pairs: List of (before, after) image tuples
            save_path: Path to save the montage
            
        Returns:
            Montage image as numpy array
        """
        if not image_pairs:
            raise ValueError("No image pairs provided")
        
        # Get dimensions from first pair
        h, w = image_pairs[0][0].shape[:2]
        n_pairs = len(image_pairs)
        
        # Create montage with 2 columns (before, after) and n_pairs rows
        montage_h = h * n_pairs
        montage_w = w * 2
        
        # Determine number of channels
        if len(image_pairs[0][0].shape) == 3:
            montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)
        else:
            montage = np.zeros((montage_h, montage_w), dtype=np.uint8)
        
        for i, (before, after) in enumerate(image_pairs):
            # Resize images to match first pair if necessary
            if before.shape[:2] != (h, w):
                before = cv2.resize(before, (w, h))
            if after.shape[:2] != (h, w):
                after = cv2.resize(after, (w, h))
            
            # Place images in montage
            y_start = i * h
            y_end = (i + 1) * h
            
            montage[y_start:y_end, :w] = before
            montage[y_start:y_end, w:] = after
        
        if save_path:
            cv2.imwrite(save_path, montage)
        
        return montage


class BatchProcessor:
    """Utilities for batch processing of images."""
    
    def __init__(self, processor: ImageProcessor, visualizer: ImageVisualizer):
        """
        Initialize batch processor.
        
        Args:
            processor: ImageProcessor instance
            visualizer: ImageVisualizer instance
        """
        self.processor = processor
        self.visualizer = visualizer
        self.logger = logging.getLogger(__name__)
    
    def process_directory(self, input_dir: str, output_dir: str,
                         processing_func: callable,
                         recursive: bool = False,
                         show_progress: bool = True,
                         create_summary: bool = True) -> Dict[str, Any]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            processing_func: Function to process each image
            recursive: Whether to process subdirectories
            show_progress: Whether to show progress
            create_summary: Whether to create processing summary
            
        Returns:
            Dictionary with processing statistics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all images
        image_files = self.processor.batch_load_images(input_dir, recursive)
        
        if not image_files:
            raise ValueError(f"No images found in {input_dir}")
        
        # Initialize statistics
        stats = {
            'total_files': len(image_files),
            'processed': 0,
            'failed': 0,
            'failed_files': [],
            'processing_times': [],
            'output_files': []
        }
        
        # Process each image
        for i, (file_path, image) in enumerate(image_files):
            try:
                import time
                start_time = time.time()
                
                # Generate output path
                input_filename = Path(file_path).name
                output_filename = f"processed_{input_filename}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Process image
                processed_image = processing_func(image)
                
                # Save result
                success = self.processor.save_image(processed_image, output_path)
                
                if success:
                    stats['processed'] += 1
                    stats['output_files'].append(output_path)
                    processing_time = time.time() - start_time
                    stats['processing_times'].append(processing_time)
                    
                    if show_progress:
                        print(f"Processed {i+1}/{len(image_files)}: {input_filename} "
                              f"({processing_time:.2f}s)")
                else:
                    raise Exception("Failed to save processed image")
                
            except Exception as e:
                stats['failed'] += 1
                stats['failed_files'].append(file_path)
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                
                if show_progress:
                    print(f"Failed {i+1}/{len(image_files)}: {Path(file_path).name}")
        
        # Calculate final statistics
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['total_processing_time'] = sum(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0
            stats['total_processing_time'] = 0
        
        stats['success_rate'] = stats['processed'] / stats['total_files']
        
        # Create summary if requested
        if create_summary:
            self._create_processing_summary(stats, output_dir)
        
        return stats  
  
    def _create_processing_summary(self, stats: Dict[str, Any], output_dir: str):
        """
        Create a summary report of batch processing.
        
        Args:
            stats: Processing statistics
            output_dir: Output directory for summary
        """
        summary_path = os.path.join(output_dir, 'processing_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("Batch Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total files: {stats['total_files']}\n")
            f.write(f"Successfully processed: {stats['processed']}\n")
            f.write(f"Failed: {stats['failed']}\n")
            f.write(f"Success rate: {stats['success_rate']:.2%}\n\n")
            
            if stats['processing_times']:
                f.write(f"Average processing time: {stats['avg_processing_time']:.2f}s\n")
                f.write(f"Total processing time: {stats['total_processing_time']:.2f}s\n\n")
            
            if stats['failed_files']:
                f.write("Failed files:\n")
                for failed_file in stats['failed_files']:
                    f.write(f"  - {failed_file}\n")
        
        self.logger.info(f"Processing summary saved to: {summary_path}")


def create_image_processor(target_size: Tuple[int, int] = (256, 256)) -> ImageProcessor:
    """
    Factory function to create an ImageProcessor instance.
    
    Args:
        target_size: Target size for image processing
        
    Returns:
        ImageProcessor instance
    """
    return ImageProcessor(target_size)


def create_visualizer(figsize: Tuple[int, int] = (12, 8)) -> ImageVisualizer:
    """
    Factory function to create an ImageVisualizer instance.
    
    Args:
        figsize: Default figure size for plots
        
    Returns:
        ImageVisualizer instance
    """
    return ImageVisualizer(figsize)


def create_batch_processor(target_size: Tuple[int, int] = (256, 256),
                          figsize: Tuple[int, int] = (12, 8)) -> BatchProcessor:
    """
    Factory function to create a BatchProcessor instance.
    
    Args:
        target_size: Target size for image processing
        figsize: Default figure size for plots
        
    Returns:
        BatchProcessor instance
    """
    processor = create_image_processor(target_size)
    visualizer = create_visualizer(figsize)
    return BatchProcessor(processor, visualizer)


# Convenience functions for common operations
def load_and_preprocess_image(image_path: str, 
                             target_size: Tuple[int, int] = (256, 256)) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """
    Convenience function to load and preprocess an image.
    
    Args:
        image_path: Path to image file
        target_size: Target size for processing
        
    Returns:
        Tuple of (preprocessed_tensor, metadata)
    """
    processor = create_image_processor(target_size)
    image = processor.load_image(image_path)
    return processor.preprocess_image(image)


def save_processed_image(tensor: tf.Tensor, output_path: str,
                        metadata: Dict[str, Any]) -> bool:
    """
    Convenience function to save a processed image tensor.
    
    Args:
        tensor: Processed image tensor
        output_path: Path to save the image
        metadata: Metadata from preprocessing
        
    Returns:
        True if successful, False otherwise
    """
    processor = create_image_processor()
    image = processor.postprocess_image(tensor, metadata)
    return processor.save_image(image, output_path)