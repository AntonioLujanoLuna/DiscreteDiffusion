"""
test_config.py

Unit tests for configuration functionality in the Discrete Diffusion project.
Tests configuration validation, loading, and saving.
"""

import unittest
import tempfile
import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from config import (
    ExperimentConfig, ModelType, TrainingMode, ModelConfig, DataConfig,
    TrainingConfig, LoggingConfig, get_default_config, get_ddm_config, get_dot_config
)


class TestConfiguration(unittest.TestCase):
    """Test suite for configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        
        # Check that config is an ExperimentConfig instance
        self.assertIsInstance(config, ExperimentConfig)
        
        # Check that config has the expected attributes
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
        
        # Check default values
        self.assertEqual(config.experiment_name, "default_experiment")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.training.mode, TrainingMode.STANDARD)
    
    def test_ddm_config(self):
        """Test DDM configuration."""
        config = get_ddm_config()
        
        # Check that config is an ExperimentConfig instance
        self.assertIsInstance(config, ExperimentConfig)
        
        # Check that training mode is DDM
        self.assertEqual(config.training.mode, TrainingMode.DDM)
    
    def test_dot_config(self):
        """Test DoT configuration."""
        config = get_dot_config()
        
        # Check that config is an ExperimentConfig instance
        self.assertIsInstance(config, ExperimentConfig)
        
        # Check that training mode is DoT
        self.assertEqual(config.training.mode, TrainingMode.DOT)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = get_default_config()
        
        # Valid configuration should not raise an exception
        config.validate()
        
        # Test invalid device
        config.device = "invalid_device"
        with self.assertRaises(ValueError):
            config.validate()
        config.device = "cuda"  # Reset to valid value
        
        # Test invalid clue ratio
        config.data.clue_ratio = 1.5
        with self.assertRaises(ValueError):
            config.validate()
        config.data.clue_ratio = 0.3  # Reset to valid value
        
        # Test invalid num_timesteps
        config.training.num_timesteps = 0
        with self.assertRaises(ValueError):
            config.validate()
        config.training.num_timesteps = 100  # Reset to valid value
        
        # Test invalid lambda_evidence for DDM
        config.training.mode = TrainingMode.DDM
        config.training.lambda_evidence = 0
        with self.assertRaises(ValueError):
            config.validate()
        config.training.lambda_evidence = 0.5  # Reset to valid value
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = get_default_config()
        
        # Save configuration to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            config.save(temp_file.name)
            
            # Check that the file exists and has content
            self.assertTrue(os.path.exists(temp_file.name))
            
            # Load configuration from the file
            loaded_config = ExperimentConfig.load(temp_file.name)
            
            # Check that loaded configuration is an ExperimentConfig instance
            self.assertIsInstance(loaded_config, ExperimentConfig)
            
            # Check that loaded configuration has the same values as the original
            self.assertEqual(loaded_config.experiment_name, config.experiment_name)
            self.assertEqual(loaded_config.seed, config.seed)
            self.assertEqual(loaded_config.device, config.device)
            self.assertEqual(loaded_config.training.mode, config.training.mode)
        
        # Clean up
        os.unlink(temp_file.name)


if __name__ == '__main__':
    unittest.main()