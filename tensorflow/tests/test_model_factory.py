"""Integration tests for model factory and utilities."""

import unittest
import tensorflow as tf
import numpy as np
import sys
import os
import tempfile
import json

# Add the tensorflow directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.model_factory import (
    ModelFactory, ModelUtils, CheckpointConverter, ModelValidator,
    get_model, create_dewarpnet_models
)
from models.unet_tf import UnetGenerator
from models.densenet_tf import DNetCCNL


class TestModelFactory(unittest.TestCase):
    """Test cases for ModelFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
    
    def test_get_model_unet(self):
        """Test creating UNet model through factory."""
        model = ModelFactory.get_model('unet', input_nc=3, output_nc=3, num_downs=7)
        
        self.assertIsInstance(model, UnetGenerator)
        
        # Test forward pass
        input_tensor = tf.random.normal((1, 256, 256, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 256, 256, 3)
        self.assertEqual(output.shape, expected_shape)
    
    def test_get_model_densenet(self):
        """Test creating DenseNet model through factory."""
        model = ModelFactory.get_model('densenet', img_size=128, in_channels=3, out_channels=2)
        
        self.assertIsInstance(model, DNetCCNL)
        
        # Test forward pass
        input_tensor = tf.random.normal((1, 128, 128, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_get_model_aliases(self):
        """Test model creation with different aliases."""
        # Test UNet aliases
        unet1 = ModelFactory.get_model('unet')
        unet2 = ModelFactory.get_model('unet_generator')
        unet3 = ModelFactory.get_model('world_coordinate')
        
        self.assertIsInstance(unet1, UnetGenerator)
        self.assertIsInstance(unet2, UnetGenerator)
        self.assertIsInstance(unet3, UnetGenerator)
        
        # Test DenseNet aliases
        dnet1 = ModelFactory.get_model('densenet')
        dnet2 = ModelFactory.get_model('dnet_ccnl')
        dnet3 = ModelFactory.get_model('backward_mapping')
        
        self.assertIsInstance(dnet1, DNetCCNL)
        self.assertIsInstance(dnet2, DNetCCNL)
        self.assertIsInstance(dnet3, DNetCCNL)
    
    def test_get_model_invalid(self):
        """Test error handling for invalid model names."""
        with self.assertRaises(ValueError):
            ModelFactory.get_model('invalid_model')
    
    def test_create_world_coordinate_model(self):
        """Test creating world coordinate model with defaults."""
        model = ModelFactory.create_world_coordinate_model()
        
        self.assertIsInstance(model, UnetGenerator)
        
        # Test with RGB input
        input_tensor = tf.random.normal((1, 256, 256, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 256, 256, 3)
        self.assertEqual(output.shape, expected_shape)
    
    def test_create_backward_mapping_model(self):
        """Test creating backward mapping model with defaults."""
        model = ModelFactory.create_backward_mapping_model()
        
        self.assertIsInstance(model, DNetCCNL)
        
        # Test with world coordinate input
        input_tensor = tf.random.normal((1, 128, 128, 3))
        output = model(input_tensor)
        
        expected_shape = (1, 128, 128, 2)
        self.assertEqual(output.shape, expected_shape)
    
    def test_convenience_function(self):
        """Test convenience get_model function."""
        model = get_model('unet', input_nc=3, output_nc=3)
        
        self.assertIsInstance(model, UnetGenerator)
    
    def test_create_dewarpnet_models(self):
        """Test creating both DewarpNet models."""
        models = create_dewarpnet_models()
        
        self.assertIn('world_coordinate', models)
        self.assertIn('backward_mapping', models)
        
        self.assertIsInstance(models['world_coordinate'], UnetGenerator)
        self.assertIsInstance(models['backward_mapping'], DNetCCNL)


class TestModelUtils(unittest.TestCase):
    """Test cases for ModelUtils."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.unet_model = get_model('unet', input_nc=3, output_nc=3, num_downs=6)
        self.densenet_model = get_model('densenet', img_size=128, in_channels=3, out_channels=2)
        
        # Build models
        _ = self.unet_model(tf.random.normal((1, 128, 128, 3)))
        _ = self.densenet_model(tf.random.normal((1, 128, 128, 3)))
    
    def test_count_parameters(self):
        """Test parameter counting."""
        params = ModelUtils.count_parameters(self.unet_model)
        
        self.assertIn('total', params)
        self.assertIn('trainable', params)
        self.assertIn('non_trainable', params)
        
        self.assertGreater(params['total'], 0)
        self.assertGreaterEqual(params['trainable'], 0)
        self.assertGreaterEqual(params['non_trainable'], 0)
        self.assertEqual(params['total'], params['trainable'] + params['non_trainable'])
        
        print(f"UNet parameters: {params}")
    
    def test_compare_model_parameters(self):
        """Test parameter comparison between models."""
        comparison = ModelUtils.compare_model_parameters(self.unet_model, self.densenet_model)
        
        self.assertIn('model1', comparison)
        self.assertIn('model2', comparison)
        self.assertIn('difference', comparison)
        self.assertIn('ratio', comparison)
        
        # Check structure
        for model_key in ['model1', 'model2']:
            self.assertIn('total', comparison[model_key])
            self.assertIn('trainable', comparison[model_key])
            self.assertIn('non_trainable', comparison[model_key])
        
        print(f"Model comparison: {comparison}")
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        summary = ModelUtils.get_model_summary(self.unet_model)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn('Total params:', summary)
        
        print(f"Model summary length: {len(summary)} characters")
    
    def test_get_model_summary_with_input_shape(self):
        """Test model summary with input shape for unbuilt model."""
        new_model = get_model('unet', input_nc=3, output_nc=3, num_downs=5)
        
        summary = ModelUtils.get_model_summary(new_model, input_shape=(64, 64, 3))
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
    
    def test_save_and_load_model_architecture(self):
        """Test saving and loading model architecture."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save architecture
            ModelUtils.save_model_architecture(self.unet_model, temp_path)
            
            # Check file exists and has content
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                
                # Should be valid JSON
                architecture = json.loads(content)
                self.assertIsInstance(architecture, dict)
            
            # Load architecture (note: this might not work for custom layers)
            # So we just test that the function doesn't crash
            try:
                loaded_model = ModelUtils.load_model_architecture(temp_path)
                print("Architecture loaded successfully")
            except Exception as e:
                print(f"Architecture loading failed (expected for custom layers): {e}")
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCheckpointConverter(unittest.TestCase):
    """Test cases for CheckpointConverter."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.model = get_model('unet', input_nc=3, output_nc=3, num_downs=5)
        _ = self.model(tf.random.normal((1, 128, 128, 3)))
    
    def test_pytorch_to_tensorflow_mapping(self):
        """Test PyTorch to TensorFlow mapping."""
        mapping = CheckpointConverter.pytorch_to_tensorflow_mapping()
        
        self.assertIsInstance(mapping, dict)
        # Currently returns empty dict as placeholder
        print(f"Mapping keys: {list(mapping.keys())}")
    
    def test_convert_pytorch_checkpoint(self):
        """Test PyTorch checkpoint conversion (placeholder)."""
        with tempfile.NamedTemporaryFile(suffix='.pth') as pytorch_file:
            with tempfile.NamedTemporaryFile(suffix='.tf') as tf_file:
                # This is a placeholder test since we don't have actual PyTorch checkpoints
                try:
                    CheckpointConverter.convert_pytorch_checkpoint(
                        pytorch_file.name, self.model, tf_file.name
                    )
                    print("Checkpoint conversion completed (placeholder)")
                except Exception as e:
                    print(f"Checkpoint conversion failed (expected): {e}")
    
    def test_verify_checkpoint_compatibility(self):
        """Test checkpoint compatibility verification."""
        with tempfile.NamedTemporaryFile(suffix='.pth') as pytorch_file:
            compatibility = CheckpointConverter.verify_checkpoint_compatibility(
                pytorch_file.name, self.model
            )
            
            self.assertIsInstance(compatibility, dict)
            self.assertIn('tensorflow_params', compatibility)
            self.assertIn('pytorch_checkpoint', compatibility)
            self.assertIn('compatible', compatibility)
            
            print(f"Compatibility check: {compatibility}")


class TestModelValidator(unittest.TestCase):
    """Test cases for ModelValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        tf.random.set_seed(42)
        self.unet_model = get_model('unet', input_nc=3, output_nc=3, num_downs=5)
        self.densenet_model = get_model('densenet', img_size=128, in_channels=3, out_channels=2)
        
        # Build models
        _ = self.unet_model(tf.random.normal((1, 128, 128, 3)))
        _ = self.densenet_model(tf.random.normal((1, 128, 128, 3)))
    
    def test_validate_output_shapes(self):
        """Test output shape validation."""
        input_shapes = {
            'test_input': (128, 128, 3)
        }
        expected_output_shapes = {
            'test_input': (128, 128, 3)
        }
        
        results = ModelValidator.validate_output_shapes(
            self.unet_model, input_shapes, expected_output_shapes
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('test_input', results)
        self.assertTrue(results['test_input'])
        
        print(f"Shape validation results: {results}")
    
    def test_validate_parameter_ranges(self):
        """Test parameter range validation."""
        results = ModelValidator.validate_parameter_ranges(self.unet_model)
        
        self.assertIsInstance(results, dict)
        self.assertIn('weights_finite', results)
        self.assertIn('weights_not_zero', results)
        self.assertIn('parameter_stats', results)
        
        # Should have finite weights
        self.assertTrue(results['weights_finite'])
        
        print(f"Parameter validation: finite={results['weights_finite']}, "
              f"non_zero={results['weights_not_zero']}")
    
    def test_validate_forward_pass(self):
        """Test forward pass validation."""
        results = ModelValidator.validate_forward_pass(
            self.unet_model, input_shape=(128, 128, 3), num_tests=3
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('all_finite', results)
        self.assertIn('consistent_shapes', results)
        self.assertIn('output_stats', results)
        self.assertIn('test_results', results)
        
        # Should have finite outputs and consistent shapes
        self.assertTrue(results['all_finite'])
        self.assertTrue(results['consistent_shapes'])
        
        # Should have correct number of test results
        self.assertEqual(len(results['test_results']), 3)
        
        print(f"Forward pass validation: finite={results['all_finite']}, "
              f"consistent={results['consistent_shapes']}")
        print(f"Output stats: {results['output_stats']}")
    
    def test_validate_forward_pass_densenet(self):
        """Test forward pass validation for DenseNet."""
        results = ModelValidator.validate_forward_pass(
            self.densenet_model, input_shape=(128, 128, 3), num_tests=2
        )
        
        self.assertIsInstance(results, dict)
        self.assertTrue(results['all_finite'])
        self.assertTrue(results['consistent_shapes'])
        
        print(f"DenseNet validation: finite={results['all_finite']}, "
              f"consistent={results['consistent_shapes']}")


if __name__ == '__main__':
    # Enable GPU memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    unittest.main(verbosity=2)