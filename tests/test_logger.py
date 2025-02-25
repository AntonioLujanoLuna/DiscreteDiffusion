"""
test_logger.py

Unit tests for logging functionality in the Discrete Diffusion project.
Tests the Logger class and related functionality.
"""

import unittest
import os
import tempfile
import shutil
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from logger import Logger, get_logger


class TestLogger(unittest.TestCase):
    """Test suite for logger functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create logger
        self.experiment_name = "test_experiment"
        self.logger = Logger(
            experiment_name=self.experiment_name,
            log_dir=self.temp_dir,
            use_tensorboard=False  # Disable TensorBoard for testing
        )
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
        
        # Close logger to release file handles
        if hasattr(self, 'logger'):
            self.logger.close()
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        # Check that log directory was created
        self.assertTrue(os.path.exists(self.logger.run_log_dir))
        
        # Check that log file was created
        log_file = os.path.join(
            self.logger.run_log_dir, 
            f"{self.logger.run_name}.log"
        )
        self.assertTrue(os.path.exists(log_file))
    
    def test_logging_levels(self):
        """Test logging at different levels."""
        # Log messages at different levels
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")
        
        # Check that messages were logged to file
        log_file = os.path.join(
            self.logger.run_log_dir, 
            f"{self.logger.run_name}.log"
        )
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("DEBUG - Debug message", log_content)
        self.assertIn("INFO - Info message", log_content)
        self.assertIn("WARNING - Warning message", log_content)
        self.assertIn("ERROR - Error message", log_content)
        self.assertIn("CRITICAL - Critical message", log_content)
    
    def test_log_metrics(self):
        """Test logging metrics."""
        # Log metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.9
        }
        self.logger.log_metrics(metrics, step=10, prefix="train")
        
        # Check that metrics were logged to file
        log_file = os.path.join(
            self.logger.run_log_dir, 
            f"{self.logger.run_name}.log"
        )
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Step 10 - train metrics: loss: 0.5000, accuracy: 0.9000", log_content)
    
    def test_log_training_batch(self):
        """Test logging training batch metrics."""
        # Log batch metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.9
        }
        self.logger.log_training_batch(metrics, epoch=0, batch_idx=10, num_batches=100)
        
        # Check that batch metrics were logged to file
        log_file = os.path.join(
            self.logger.run_log_dir, 
            f"{self.logger.run_name}.log"
        )
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Epoch: 1, Batch: 11/100 - loss: 0.5000, accuracy: 0.9000", log_content)
    
    def test_log_epoch(self):
        """Test logging epoch metrics."""
        # Log epoch metrics
        metrics = {
            "loss": 0.5,
            "accuracy": 0.9
        }
        self.logger.log_epoch(metrics, epoch=5, prefix="train")
        
        # Check that epoch metrics were logged to file
        log_file = os.path.join(
            self.logger.run_log_dir, 
            f"{self.logger.run_name}.log"
        )
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        self.assertIn("Epoch 6 - train metrics: loss: 0.5000, accuracy: 0.9000", log_content)
    
    def test_get_logger(self):
        """Test get_logger function."""
        # Clean up logger from setUp to avoid interference
        self.logger.close()
        
        # Create a new logger with get_logger
        logger1 = get_logger(
            experiment_name="test_get_logger",
            log_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Get the same logger again
        logger2 = get_logger()
        
        # Check that both references point to the same logger
        self.assertEqual(logger1, logger2)
        
        # Clean up
        logger1.close()


if __name__ == '__main__':
    unittest.main()