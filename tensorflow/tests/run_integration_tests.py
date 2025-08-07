#!/usr/bin/env python3
"""
Comprehensive integration test runner for TensorFlow DewarpNet implementation.
Runs all integration tests with detailed reporting and performance analysis.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np


class IntegrationTestRunner:
    """Comprehensive integration test runner."""
    
    def __init__(self, output_dir: str = None):
        """Initialize test runner.
        
        Args:
            output_dir: Directory to save test results and outputs
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'integration_test_results')
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self._get_system_info(),
            'test_results': {},
            'performance_metrics': {},
            'summary': {}
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for test context."""
        info = {
            'tensorflow_version': tf.__version__,
            'python_version': sys.version,
            'platform': sys.platform,
        }
        
        # GPU information
        gpus = tf.config.experimental.list_physical_devices('GPU')
        info['gpu_count'] = len(gpus)
        info['gpu_names'] = []
        
        for gpu in gpus:
            try:
                gpu_name = tf.config.experimental.get_device_details(gpu).get('device_name', 'Unknown')
                info['gpu_names'].append(gpu_name)
            except:
                info['gpu_names'].append('Unknown GPU')
        
        return info
    
    def run_test_suite(self, test_categories: List[str] = None) -> bool:
        """Run integration test suite.
        
        Args:
            test_categories: List of test categories to run. If None, runs all.
            
        Returns:
            True if all tests passed, False otherwise
        """
        print("🚀 Starting TensorFlow DewarpNet Integration Test Suite")
        print("=" * 70)
        
        # Configure TensorFlow
        self._configure_tensorflow()
        
        # Available test categories
        all_categories = [
            'end_to_end_training',
            'model_compatibility', 
            'loss_compatibility',
            'data_pipeline',
            'inference_pipeline',
            'visual_output',
            'memory_performance',
            'configuration'
        ]
        
        categories_to_run = test_categories or all_categories
        
        print(f"Running test categories: {', '.join(categories_to_run)}")
        print("-" * 70)
        
        overall_success = True
        start_time = time.time()
        
        for category in categories_to_run:
            if category in all_categories:
                success = self._run_test_category(category)
                overall_success = overall_success and success
            else:
                print(f"⚠ Unknown test category: {category}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Generate summary
        self._generate_summary(overall_success, total_time)
        
        # Save results
        self._save_results()
        
        return overall_success
    
    def _configure_tensorflow(self):
        """Configure TensorFlow for testing."""
        # Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ Configured {len(gpus)} GPU(s) with memory growth")
            except RuntimeError as e:
                print(f"⚠ GPU configuration warning: {e}")
        else:
            print("ℹ No GPUs detected, using CPU")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def _run_test_category(self, category: str) -> bool:
        """Run a specific test category.
        
        Args:
            category: Test category name
            
        Returns:
            True if category tests passed, False otherwise
        """
        print(f"\n📋 Running {category.replace('_', ' ').title()} Tests")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Import and run specific test class
            if category == 'end_to_end_training':
                success = self._run_end_to_end_training_tests()
            elif category == 'model_compatibility':
                success = self._run_model_compatibility_tests()
            elif category == 'loss_compatibility':
                success = self._run_loss_compatibility_tests()
            elif category == 'data_pipeline':
                success = self._run_data_pipeline_tests()
            elif category == 'inference_pipeline':
                success = self._run_inference_pipeline_tests()
            elif category == 'visual_output':
                success = self._run_visual_output_tests()
            elif category == 'memory_performance':
                success = self._run_memory_performance_tests()
            elif category == 'configuration':
                success = self._run_configuration_tests()
            else:
                print(f"❌ Unknown test category: {category}")
                success = False
                
        except Exception as e:
            print(f"❌ Error running {category} tests: {e}")
            success = False
        
        end_time = time.time()
        test_time = end_time - start_time
        
        # Record results
        self.results['test_results'][category] = {
            'success': success,
            'duration': test_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"\n{status} - {category.replace('_', ' ').title()} Tests ({test_time:.2f}s)")
        
        return success
    
    def _run_end_to_end_training_tests(self) -> bool:
        """Run end-to-end training pipeline tests."""
        from test_integration import TestEndToEndTrainingPipeline
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndTrainingPipeline)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_model_compatibility_tests(self) -> bool:
        """Run model compatibility tests."""
        from test_integration import TestModelCompatibility
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestModelCompatibility)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_loss_compatibility_tests(self) -> bool:
        """Run loss compatibility tests."""
        from test_integration import TestLossCompatibility
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestLossCompatibility)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_data_pipeline_tests(self) -> bool:
        """Run data pipeline integration tests."""
        from test_integration import TestDataPipelineIntegration
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDataPipelineIntegration)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_inference_pipeline_tests(self) -> bool:
        """Run inference pipeline tests."""
        from test_integration import TestInferencePipelineIntegration
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestInferencePipelineIntegration)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_visual_output_tests(self) -> bool:
        """Run visual output comparison tests."""
        from test_integration import TestVisualOutputComparison
        import unittest
        
        # Set output directory for visual tests
        test_instance = TestVisualOutputComparison()
        test_instance.output_dir = os.path.join(self.output_dir, 'visual_outputs')
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestVisualOutputComparison)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_memory_performance_tests(self) -> bool:
        """Run memory and performance tests."""
        from test_integration import TestMemoryAndPerformance
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoryAndPerformance)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _run_configuration_tests(self) -> bool:
        """Run configuration integration tests."""
        from test_integration import TestConfigurationIntegration
        import unittest
        
        suite = unittest.TestLoader().loadTestsFromTestCase(TestConfigurationIntegration)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def _generate_summary(self, overall_success: bool, total_time: float):
        """Generate test summary."""
        passed_tests = sum(1 for result in self.results['test_results'].values() if result['success'])
        total_tests = len(self.results['test_results'])
        
        self.results['summary'] = {
            'overall_success': overall_success,
            'total_time': total_time,
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        print("\n" + "=" * 70)
        print("📊 INTEGRATION TEST SUMMARY")
        print("=" * 70)
        print(f"Overall Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({self.results['summary']['success_rate']:.1%})")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"System: TensorFlow {tf.__version__}, {len(tf.config.experimental.list_physical_devices('GPU'))} GPU(s)")
        
        if not overall_success:
            print("\n❌ Failed Test Categories:")
            for category, result in self.results['test_results'].items():
                if not result['success']:
                    print(f"  - {category.replace('_', ' ').title()}")
    
    def _save_results(self):
        """Save test results to files."""
        # Save JSON results
        results_file = os.path.join(self.output_dir, 'integration_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save human-readable report
        report_file = os.path.join(self.output_dir, 'integration_test_report.txt')
        with open(report_file, 'w') as f:
            f.write("TensorFlow DewarpNet Integration Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"TensorFlow Version: {self.results['system_info']['tensorflow_version']}\n")
            f.write(f"GPU Count: {self.results['system_info']['gpu_count']}\n\n")
            
            f.write("Test Results:\n")
            f.write("-" * 20 + "\n")
            for category, result in self.results['test_results'].items():
                status = "PASSED" if result['success'] else "FAILED"
                f.write(f"{category.replace('_', ' ').title()}: {status} ({result['duration']:.2f}s)\n")
            
            f.write(f"\nOverall: {'PASSED' if self.results['summary']['overall_success'] else 'FAILED'}\n")
            f.write(f"Success Rate: {self.results['summary']['success_rate']:.1%}\n")
            f.write(f"Total Time: {self.results['summary']['total_time']:.2f}s\n")
        
        print(f"\n📄 Results saved to: {self.output_dir}")
        print(f"  - JSON: {results_file}")
        print(f"  - Report: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run TensorFlow DewarpNet integration tests')
    parser.add_argument('--categories', nargs='+', 
                       help='Test categories to run (default: all)')
    parser.add_argument('--output-dir', 
                       help='Output directory for test results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = IntegrationTestRunner(output_dir=args.output_dir)
    
    # Run tests
    success = runner.run_test_suite(test_categories=args.categories)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()