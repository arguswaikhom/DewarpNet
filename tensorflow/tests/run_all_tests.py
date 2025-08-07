"""
Comprehensive test runner for TensorFlow DewarpNet implementation.
Runs all unit tests and provides detailed reporting.
"""

import unittest
import sys
import os
import time
from io import StringIO
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_unet import *
from test_densenet import *
from test_wc_loader import *
from test_bm_loader import *
from test_grad_loss import *
from test_recon_loss import *
from test_loss_factory import *
from test_model_factory import *
from test_inference import *
from test_data_pipeline import *
from test_training_manager import *
from test_training_utils import *


class TestResult:
    """Custom test result class for detailed reporting."""
    
    def __init__(self):
        self.tests_run = 0
        self.failures = []
        self.errors = []
        self.skipped = []
        self.successes = []
        self.start_time = None
        self.end_time = None
    
    def start_test(self, test):
        """Called when a test starts."""
        if self.start_time is None:
            self.start_time = time.time()
    
    def add_success(self, test):
        """Called when a test passes."""
        self.tests_run += 1
        self.successes.append(test)
    
    def add_error(self, test, err):
        """Called when a test has an error."""
        self.tests_run += 1
        self.errors.append((test, err))
    
    def add_failure(self, test, err):
        """Called when a test fails."""
        self.tests_run += 1
        self.failures.append((test, err))
    
    def add_skip(self, test, reason):
        """Called when a test is skipped."""
        self.tests_run += 1
        self.skipped.append((test, reason))
    
    def stop_test(self, test):
        """Called when a test ends."""
        self.end_time = time.time()
    
    def get_summary(self):
        """Get test summary."""
        total_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        return {
            'tests_run': self.tests_run,
            'successes': len(self.successes),
            'failures': len(self.failures),
            'errors': len(self.errors),
            'skipped': len(self.skipped),
            'total_time': total_time,
            'success_rate': len(self.successes) / self.tests_run if self.tests_run > 0 else 0
        }


def setup_tensorflow():
    """Setup TensorFlow for testing."""
    print("Setting up TensorFlow...")
    
    # Enable GPU memory growth to avoid OOM errors
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
    print("✓ Set random seed for reproducibility")
    
    # Print TensorFlow version
    print(f"✓ TensorFlow version: {tf.__version__}")


def create_test_suite():
    """Create comprehensive test suite."""
    print("Creating test suite...")
    
    # Define test modules and their descriptions
    test_modules = [
        ('test_unet', 'UNet Model Architecture'),
        ('test_densenet', 'DenseNet Model Architecture'),
        ('test_wc_loader', 'World Coordinate Data Loader'),
        ('test_bm_loader', 'Backward Mapping Data Loader'),
        ('test_grad_loss', 'Gradient Loss Function'),
        ('test_recon_loss', 'Reconstruction Loss Function'),
        ('test_loss_factory', 'Loss Function Factory'),
        ('test_model_factory', 'Model Factory'),
        ('test_inference', 'Inference Pipeline'),
        ('test_data_pipeline', 'Data Pipeline'),
        ('test_training_manager', 'Training Manager'),
        ('test_training_utils', 'Training Utilities'),
    ]
    
    # Create test suite
    suite = unittest.TestSuite()
    
    for module_name, description in test_modules:
        try:
            # Load test module
            module = sys.modules[module_name]
            
            # Add all test cases from module
            module_suite = unittest.TestLoader().loadTestsFromModule(module)
            suite.addTest(module_suite)
            
            print(f"✓ Added tests from {module_name} ({description})")
            
        except Exception as e:
            print(f"✗ Failed to load tests from {module_name}: {e}")
    
    return suite


def run_test_category(category_name, test_classes, verbose=True):
    """Run a specific category of tests."""
    print(f"\n{'='*60}")
    print(f"RUNNING {category_name.upper()} TESTS")
    print(f"{'='*60}")
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test_class))
    
    # Run tests
    stream = StringIO() if not verbose else sys.stdout
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2 if verbose else 1,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print(f"\n{category_name} Tests Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print(f"  Time: {end_time - start_time:.2f}s")
    
    if result.failures:
        print(f"\n{category_name} Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n{category_name} Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful(), result.testsRun, len(result.failures), len(result.errors)


def run_all_tests(verbose=True, categories=None):
    """Run all tests with detailed reporting."""
    print("🚀 Starting TensorFlow DewarpNet Test Suite")
    print("="*60)
    
    # Setup TensorFlow
    setup_tensorflow()
    
    # Define test categories
    test_categories = {
        'Model Architecture': [
            TestUnetSkipConnectionBlock,
            TestUnetGenerator,
            TestCoordConvChannels,
            TestDenseBlocks,
            TestWaspDenseEncoder128,
            TestWaspDenseDecoder128,
            TestDNetCCNL,
        ],
        'Data Loading': [
            TestDoc3DWCLoader,
            TestAugmentations,
            TestDatasetCreation,
            TestDoc3DBMLoader,
            TestBMDatasetCreation,
            TestTFDataPipeline,
            TestTrainingPipeline,
        ],
        'Loss Functions': [
            TestGradLoss,
            TestGradLossComparison,
            TestReconLoss,
            TestSSIMLoss,
            TestReconLossIntegration,
            TestLossFactory,
            TestCombinedLoss,
            TestLossLogger,
            TestConvenienceFunctions,
            TestLossFactoryIntegration,
        ],
        'Model Factory': [
            TestModelFactory,
            TestModelUtils,
            TestCheckpointConverter,
            TestModelValidator,
        ],
        'Inference Pipeline': [
            TestImageProcessor,
            TestImageVisualizer,
            TestBatchProcessor,
            TestDewarpNetInference,
        ],
        'Training System': [
            TestTrainingManager,
            TestMetricsComputer,
            TestTrainingMonitor,
            TestPerformanceProfiler,
            TestMetricsAggregator,
            TestValidationMetrics,
            TestTrainingUtilsIntegration,
        ],
        'Training Utilities': [
            TestTrainingLogger,
            TestCheckpointManager,
            TestTensorBoardLogger,
            TestEarlyStopping,
            TestLearningRateScheduler,
            TestModelCheckpointer,
            TestTrainingStateManager,
            TestTrainingUtilsIntegration,
        ],
    }
    
    # Filter categories if specified
    if categories:
        test_categories = {k: v for k, v in test_categories.items() if k in categories}
    
    # Run tests by category
    total_tests = 0
    total_failures = 0
    total_errors = 0
    category_results = {}
    
    overall_start_time = time.time()
    
    for category_name, test_classes in test_categories.items():
        success, tests_run, failures, errors = run_test_category(
            category_name, test_classes, verbose
        )
        
        category_results[category_name] = {
            'success': success,
            'tests_run': tests_run,
            'failures': failures,
            'errors': errors
        }
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
    
    overall_end_time = time.time()
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"Total test time: {overall_end_time - overall_start_time:.2f}s")
    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Success rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    print(f"\nCategory Breakdown:")
    for category, results in category_results.items():
        status = "✓ PASS" if results['success'] else "✗ FAIL"
        print(f"  {status} {category}: {results['tests_run']} tests, "
              f"{results['failures']} failures, {results['errors']} errors")
    
    # Overall result
    overall_success = total_failures == 0 and total_errors == 0
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ TensorFlow DewarpNet implementation is working correctly!")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        print(f"   {total_failures} failures, {total_errors} errors")
        print("   Please review the test output above for details.")
    
    return overall_success


def run_quick_tests():
    """Run a quick subset of tests for rapid feedback."""
    print("🏃 Running Quick Test Suite (essential tests only)")
    
    quick_categories = ['Model Architecture', 'Loss Functions']
    return run_all_tests(verbose=False, categories=quick_categories)


def run_specific_test(test_name):
    """Run a specific test by name."""
    print(f"🎯 Running specific test: {test_name}")
    
    # Try to find and run the specific test
    suite = unittest.TestSuite()
    
    try:
        # Load test by name
        suite.addTest(unittest.TestLoader().loadTestsFromName(test_name))
        
        # Run test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Failed to run test {test_name}: {e}")
        return False


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TensorFlow DewarpNet Test Runner')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test suite (essential tests only)')
    parser.add_argument('--test', type=str, 
                       help='Run specific test by name')
    parser.add_argument('--category', type=str, nargs='+',
                       help='Run specific test categories')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    if args.test:
        success = run_specific_test(args.test)
    elif args.quick:
        success = run_quick_tests()
    else:
        success = run_all_tests(
            verbose=not args.quiet,
            categories=args.category
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()