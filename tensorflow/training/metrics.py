"""
Metrics computation and monitoring utilities for TensorFlow DewarpNet training.
Provides comprehensive metrics tracking and evaluation utilities.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from collections import defaultdict, deque
import time
import json
import os


class MetricsComputer:
    """Compute various metrics for DewarpNet training."""
    
    def __init__(self):
        """Initialize metrics computer."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def compute_mse(predictions: tf.Tensor, targets: tf.Tensor, 
                   mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute Mean Squared Error.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for selective computation
            
        Returns:
            MSE value
        """
        squared_diff = tf.square(predictions - targets)
        
        if mask is not None:
            squared_diff = squared_diff * mask
            return tf.reduce_sum(squared_diff) / tf.reduce_sum(mask)
        else:
            return tf.reduce_mean(squared_diff)
    
    @staticmethod
    def compute_mae(predictions: tf.Tensor, targets: tf.Tensor,
                   mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute Mean Absolute Error.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            mask: Optional mask for selective computation
            
        Returns:
            MAE value
        """
        abs_diff = tf.abs(predictions - targets)
        
        if mask is not None:
            abs_diff = abs_diff * mask
            return tf.reduce_sum(abs_diff) / tf.reduce_sum(mask)
        else:
            return tf.reduce_mean(abs_diff)
    
    @staticmethod
    def compute_psnr(predictions: tf.Tensor, targets: tf.Tensor,
                    max_val: float = 1.0) -> tf.Tensor:
        """Compute Peak Signal-to-Noise Ratio.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            max_val: Maximum possible value
            
        Returns:
            PSNR value
        """
        mse = tf.reduce_mean(tf.square(predictions - targets))
        return 20 * tf.math.log(max_val) / tf.math.log(10.0) - 10 * tf.math.log(mse) / tf.math.log(10.0)
    
    @staticmethod
    def compute_ssim(predictions: tf.Tensor, targets: tf.Tensor,
                    max_val: float = 1.0) -> tf.Tensor:
        """Compute Structural Similarity Index.
        
        Args:
            predictions: Predicted values [batch, height, width, channels]
            targets: Ground truth values [batch, height, width, channels]
            max_val: Maximum possible value
            
        Returns:
            SSIM value
        """
        return tf.image.ssim(predictions, targets, max_val=max_val)
    
    @staticmethod
    def compute_gradient_error(predictions: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """Compute gradient error using Sobel filters.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            
        Returns:
            Gradient error
        """
        # Sobel filters for gradient computation
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
        
        # Compute gradients for each channel
        pred_grad_x = tf.nn.conv2d(predictions, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        pred_grad_y = tf.nn.conv2d(predictions, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        target_grad_x = tf.nn.conv2d(targets, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        target_grad_y = tf.nn.conv2d(targets, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        # Compute gradient magnitude
        pred_grad_mag = tf.sqrt(tf.square(pred_grad_x) + tf.square(pred_grad_y))
        target_grad_mag = tf.sqrt(tf.square(target_grad_x) + tf.square(target_grad_y))
        
        # Compute L1 error between gradient magnitudes
        return tf.reduce_mean(tf.abs(pred_grad_mag - target_grad_mag))
    
    def compute_all_metrics(self, predictions: tf.Tensor, targets: tf.Tensor,
                           prefix: str = '') -> Dict[str, tf.Tensor]:
        """Compute all available metrics.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics[f'{prefix}mse'] = self.compute_mse(predictions, targets)
        metrics[f'{prefix}mae'] = self.compute_mae(predictions, targets)
        metrics[f'{prefix}psnr'] = self.compute_psnr(predictions, targets)
        
        # SSIM (if images have appropriate shape)
        if len(predictions.shape) == 4 and predictions.shape[-1] in [1, 3]:
            metrics[f'{prefix}ssim'] = tf.reduce_mean(self.compute_ssim(predictions, targets))
        
        # Gradient error
        metrics[f'{prefix}gradient_error'] = self.compute_gradient_error(predictions, targets)
        
        return metrics


class TrainingMonitor:
    """Monitor training progress and detect issues."""
    
    def __init__(self, window_size: int = 100):
        """Initialize training monitor.
        
        Args:
            window_size: Size of sliding window for monitoring
        """
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.logger = logging.getLogger(__name__)
        
        # Monitoring flags
        self.loss_exploded = False
        self.loss_plateaued = False
        self.gradients_vanished = False
    
    def update(self, losses: Dict[str, float], metrics: Optional[Dict[str, float]] = None,
               gradients: Optional[List[tf.Tensor]] = None):
        """Update monitor with new values.
        
        Args:
            losses: Dictionary of loss values
            metrics: Optional dictionary of metric values
            gradients: Optional list of gradient tensors
        """
        # Update loss history
        if 'total_loss' in losses:
            self.loss_history.append(losses['total_loss'])
        
        # Update metric history
        if metrics:
            for name, value in metrics.items():
                self.metric_history[name].append(value)
        
        # Check for issues
        self._check_loss_explosion(losses)
        self._check_loss_plateau()
        
        if gradients:
            self._check_gradient_vanishing(gradients)
    
    def _check_loss_explosion(self, losses: Dict[str, float], threshold: float = 1e6):
        """Check if loss has exploded.
        
        Args:
            losses: Current loss values
            threshold: Threshold for explosion detection
        """
        for name, value in losses.items():
            if value > threshold or np.isnan(value) or np.isinf(value):
                if not self.loss_exploded:
                    self.logger.warning(f"Loss explosion detected: {name} = {value}")
                    self.loss_exploded = True
                break
    
    def _check_loss_plateau(self, patience: int = 50, min_delta: float = 1e-6):
        """Check if loss has plateaued.
        
        Args:
            patience: Number of steps to wait before declaring plateau
            min_delta: Minimum change to consider as improvement
        """
        if len(self.loss_history) < patience:
            return
        
        recent_losses = list(self.loss_history)[-patience:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        
        if loss_change < min_delta:
            if not self.loss_plateaued:
                self.logger.warning(f"Loss plateau detected: change = {loss_change}")
                self.loss_plateaued = True
    
    def _check_gradient_vanishing(self, gradients: List[tf.Tensor], threshold: float = 1e-7):
        """Check if gradients are vanishing.
        
        Args:
            gradients: List of gradient tensors
            threshold: Threshold for vanishing detection
        """
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += tf.reduce_sum(tf.square(grad))
        
        total_norm = tf.sqrt(total_norm)
        
        if total_norm < threshold:
            if not self.gradients_vanished:
                self.logger.warning(f"Gradient vanishing detected: norm = {total_norm}")
                self.gradients_vanished = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status.
        
        Returns:
            Dictionary with monitoring status
        """
        status = {
            'loss_exploded': self.loss_exploded,
            'loss_plateaued': self.loss_plateaued,
            'gradients_vanished': self.gradients_vanished,
            'loss_history_length': len(self.loss_history),
            'recent_loss_trend': self._compute_loss_trend()
        }
        
        return status
    
    def _compute_loss_trend(self) -> Optional[str]:
        """Compute recent loss trend.
        
        Returns:
            Trend description ('improving', 'worsening', 'stable', or None)
        """
        if len(self.loss_history) < 10:
            return None
        
        recent_losses = list(self.loss_history)[-10:]
        first_half = np.mean(recent_losses[:5])
        second_half = np.mean(recent_losses[5:])
        
        change_ratio = (second_half - first_half) / first_half
        
        if change_ratio < -0.01:
            return 'improving'
        elif change_ratio > 0.01:
            return 'worsening'
        else:
            return 'stable'
    
    def reset(self):
        """Reset monitoring state."""
        self.loss_history.clear()
        self.metric_history.clear()
        self.loss_exploded = False
        self.loss_plateaued = False
        self.gradients_vanished = False


class PerformanceProfiler:
    """Profile training performance and identify bottlenecks."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timings = defaultdict(list)
        self.current_timings = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, name: str):
        """Start timing an operation.
        
        Args:
            name: Name of the operation
        """
        self.current_timings[name] = time.time()
    
    def end_timer(self, name: str):
        """End timing an operation.
        
        Args:
            name: Name of the operation
        """
        if name in self.current_timings:
            elapsed = time.time() - self.current_timings[name]
            self.timings[name].append(elapsed)
            del self.current_timings[name]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics.
        
        Returns:
            Dictionary with timing statistics for each operation
        """
        stats = {}
        
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        
        return stats
    
    def print_stats(self):
        """Print timing statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING RESULTS")
        print("="*60)
        
        for name, stat in stats.items():
            print(f"\n{name}:")
            print(f"  Mean: {stat['mean']:.4f}s")
            print(f"  Std:  {stat['std']:.4f}s")
            print(f"  Min:  {stat['min']:.4f}s")
            print(f"  Max:  {stat['max']:.4f}s")
            print(f"  Total: {stat['total']:.2f}s")
            print(f"  Count: {stat['count']}")
        
        print("="*60)
    
    def reset(self):
        """Reset profiling data."""
        self.timings.clear()
        self.current_timings.clear()


class MetricsAggregator:
    """Aggregate metrics across multiple batches or epochs."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.reset()
    
    def reset(self):
        """Reset aggregation state."""
        self.metrics = defaultdict(list)
        self.weights = defaultdict(list)
    
    def update(self, metrics: Dict[str, Union[float, tf.Tensor]], weight: float = 1.0):
        """Update aggregator with new metrics.
        
        Args:
            metrics: Dictionary of metric values
            weight: Weight for this update (e.g., batch size)
        """
        for name, value in metrics.items():
            if isinstance(value, tf.Tensor):
                value = float(value.numpy())
            
            self.metrics[name].append(value)
            self.weights[name].append(weight)
    
    def get_averages(self) -> Dict[str, float]:
        """Get weighted averages of all metrics.
        
        Returns:
            Dictionary of average metric values
        """
        averages = {}
        
        for name in self.metrics:
            values = np.array(self.metrics[name])
            weights = np.array(self.weights[name])
            
            if len(values) > 0:
                averages[name] = np.average(values, weights=weights)
            else:
                averages[name] = 0.0
        
        return averages
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics for each metric
        """
        summary = {}
        
        for name in self.metrics:
            values = np.array(self.metrics[name])
            weights = np.array(self.weights[name])
            
            if len(values) > 0:
                summary[name] = {
                    'mean': np.average(values, weights=weights),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                summary[name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return summary


class ValidationMetrics:
    """Specialized metrics for validation evaluation."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize validation metrics.
        
        Args:
            save_dir: Directory to save validation results
        """
        self.save_dir = save_dir
        self.results_history = []
        self.logger = logging.getLogger(__name__)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    def evaluate_epoch(self, model: tf.keras.Model, dataset: tf.data.Dataset,
                      metrics_computer: MetricsComputer) -> Dict[str, float]:
        """Evaluate model on validation dataset.
        
        Args:
            model: Model to evaluate
            dataset: Validation dataset
            metrics_computer: Metrics computer instance
            
        Returns:
            Dictionary of validation metrics
        """
        aggregator = MetricsAggregator()
        
        for batch_idx, (inputs, targets) in enumerate(dataset):
            # Forward pass
            predictions = model(inputs, training=False)
            
            # Compute metrics
            batch_metrics = metrics_computer.compute_all_metrics(
                predictions, targets, prefix='val_'
            )
            
            # Update aggregator
            batch_size = tf.shape(inputs)[0]
            aggregator.update(batch_metrics, weight=float(batch_size))
        
        # Get average metrics
        avg_metrics = aggregator.get_averages()
        
        # Store results
        self.results_history.append(avg_metrics)
        
        # Save results if directory specified
        if self.save_dir:
            self._save_results(avg_metrics, len(self.results_history) - 1)
        
        return avg_metrics
    
    def _save_results(self, metrics: Dict[str, float], epoch: int):
        """Save validation results to file.
        
        Args:
            metrics: Validation metrics
            epoch: Current epoch
        """
        results_file = os.path.join(self.save_dir, f'validation_epoch_{epoch:04d}.json')
        
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def get_best_epoch(self, metric_name: str = 'val_mse', 
                      mode: str = 'min') -> Tuple[int, float]:
        """Get epoch with best validation metric.
        
        Args:
            metric_name: Name of metric to optimize
            mode: 'min' or 'max' for optimization direction
            
        Returns:
            Tuple of (best_epoch, best_value)
        """
        if not self.results_history:
            return -1, float('inf') if mode == 'min' else float('-inf')
        
        values = [results.get(metric_name, float('inf')) for results in self.results_history]
        
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return best_idx, values[best_idx]
    
    def plot_metrics(self, metric_names: Optional[List[str]] = None, 
                    save_path: Optional[str] = None):
        """Plot validation metrics over time.
        
        Args:
            metric_names: List of metrics to plot (None for all)
            save_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return
        
        if not self.results_history:
            self.logger.warning("No validation results to plot")
            return
        
        # Determine metrics to plot
        if metric_names is None:
            metric_names = list(self.results_history[0].keys())
        
        # Create subplots
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        epochs = range(len(self.results_history))
        
        for i, metric_name in enumerate(metric_names):
            values = [results.get(metric_name, 0.0) for results in self.results_history]
            
            axes[i].plot(epochs, values, 'b-', linewidth=2)
            axes[i].set_title(f'Validation {metric_name}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Validation plots saved to: {save_path}")
        
        plt.show()


# Convenience functions
def compute_metrics(predictions: tf.Tensor, targets: tf.Tensor, 
                   prefix: str = '') -> Dict[str, tf.Tensor]:
    """Compute standard metrics for predictions."""
    computer = MetricsComputer()
    return computer.compute_all_metrics(predictions, targets, prefix)


def create_metrics_computer() -> MetricsComputer:
    """Create a metrics computer instance."""
    return MetricsComputer()


def create_training_monitor(window_size: int = 100) -> TrainingMonitor:
    """Create a training monitor instance."""
    return TrainingMonitor(window_size)


def create_performance_profiler() -> PerformanceProfiler:
    """Create a performance profiler instance."""
    return PerformanceProfiler()