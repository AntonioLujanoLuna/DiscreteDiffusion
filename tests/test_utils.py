"""
test_utils.py

Unit tests for utility functions in the Discrete Diffusion project.
Tests various utility modules including common, diffusion, and model utilities.
"""

import unittest
import torch
import numpy as np
import sys
import os
import tempfile

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.common import set_seeds, validate_trajectory
from utils.diffusion_utils import enforce_clue_logits, LearnedNoiseSchedule
from utils.model_utils import create_model_from_config


class TestCommonUtils(unittest.TestCase):
    """Test suite for common utility functions."""
    
    def test_set_seeds(self):
        """Test set_seeds function."""
        # Set a seed
        set_seeds(42)
        
        # Generate random values with numpy
        np_rand1 = np.random.rand()
        
        # Generate random values with PyTorch
        torch_rand1 = torch.rand(1).item()
        
        # Reset the seed
        set_seeds(42)
        
        # Generate random values again
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()
        
        # Check that the random values are the same
        self.assertEqual(np_rand1, np_rand2)
        self.assertEqual(torch_rand1, torch_rand2)
    
    def test_validate_trajectory(self):
        """Test validate_trajectory function."""
        # Create a trajectory with mixed types
        trajectory = [
            np.zeros((9, 9)),  # 2D array
            np.zeros((1, 9, 9)),  # 3D array with singleton dimension
            [[0] * 9 for _ in range(9)]  # List of lists
        ]
        
        # Validate the trajectory
        validated = validate_trajectory(trajectory)
        
        # Check that all boards are 2D numpy arrays with shape (9, 9)
        for board in validated:
            self.assertIsInstance(board, np.ndarray)
            self.assertEqual(board.shape, (9, 9))
        
        # Test with invalid board shape
        trajectory_invalid = [
            np.zeros((9, 9)),
            np.zeros((8, 8))  # Invalid shape
        ]
        
        # Check that validation raises an error
        with self.assertRaises(ValueError):
            validate_trajectory(trajectory_invalid)


class TestDiffusionUtils(unittest.TestCase):
    """Test suite for diffusion utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_tokens = 10
        
        # Create sample inputs
        self.logits = torch.randn((self.batch_size, 9, 9, self.num_tokens), device=self.device)
        self.board = torch.randint(0, self.num_tokens, (self.batch_size, 9, 9), device=self.device)
        self.clue_mask = torch.zeros((self.batch_size, 9, 9), device=self.device)
        self.clue_mask[:, :2, :2] = 1  # Set some cells as clues
    
    def test_enforce_clue_logits(self):
        """Test enforce_clue_logits function."""
        # Apply enforce_clue_logits
        modified_logits = enforce_clue_logits(self.logits, self.board, self.clue_mask)
        
        # Check that logits for clue cells are modified
        clue_cells = torch.where(self.clue_mask == 1)
        for b, i, j in zip(*clue_cells):
            token = self.board[b, i, j]
            # The logit for the correct token should be very high
            self.assertTrue(modified_logits[b, i, j, token] > 1e5)
            # Logits for other tokens should be very low
            other_logits = torch.cat([modified_logits[b, i, j, :token], modified_logits[b, i, j, token+1:]])
            self.assertTrue((other_logits < -1e5).all())
        
        # Check that logits for non-clue cells are unchanged
        non_clue_cells = torch.where(self.clue_mask == 0)
        for b, i, j in zip(*non_clue_cells):
            self.assertTrue(torch.allclose(modified_logits[b, i, j], self.logits[b, i, j]))
    
    def test_learned_noise_schedule(self):
        """Test LearnedNoiseSchedule."""
        # Create learned noise schedule
        noise_schedule = LearnedNoiseSchedule(hidden_dim=16).to(self.device)
        
        # Test with different timesteps
        for t_val in [1, 50, 100]:
            t = torch.full((self.batch_size, 1), t_val, device=self.device, dtype=torch.float)
            
            # Get noise probability
            noise_prob = noise_schedule(t, 100)
            
            # Check shape
            self.assertEqual(noise_prob.shape, (self.batch_size, 1))
            
            # Check that noise probability is between 0 and 1
            self.assertTrue((noise_prob >= 0).all() and (noise_prob <= 1).all())


class TestModelUtils(unittest.TestCase):
    """Test suite for model utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
    
    def test_create_model_from_config(self):
        """Test create_model_from_config function."""
        # Test creating ImprovedSudokuDenoiser
        config_base = {
            "model_type": "Base",
            "num_tokens": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "nhead": 4,
            "dropout": 0.1
        }
        
        model_base = create_model_from_config(config_base, self.device)
        
        # Check that the model has the right type
        from models import ImprovedSudokuDenoiser
        self.assertIsInstance(model_base, ImprovedSudokuDenoiser)
        
        # Check that model parameters match the config
        self.assertEqual(model_base.num_tokens, config_base["num_tokens"])
        self.assertEqual(model_base.hidden_dim, config_base["hidden_dim"])
        
        # Test creating HybridSudokuDenoiser
        config_hybrid = {
            "model_type": "Hybrid",
            "num_tokens": 10,
            "hidden_dim": 64,
            "num_layers": 2,
            "nhead": 4,
            "dropout": 0.1,
            "num_conv_layers": 2,
            "conv_kernel_size": 3
        }
        
        model_hybrid = create_model_from_config(config_hybrid, self.device)
        
        # Check that the model has the right type
        from models import HybridSudokuDenoiser
        self.assertIsInstance(model_hybrid, HybridSudokuDenoiser)
        
        # Check that model parameters match the config
        self.assertEqual(model_hybrid.num_tokens, config_hybrid["num_tokens"])
        self.assertEqual(model_hybrid.hidden_dim, config_hybrid["hidden_dim"])
        
        # Test with invalid model type
        config_invalid = {
            "model_type": "Invalid",
            "num_tokens": 10
        }
        
        # Check that creation raises an error
        with self.assertRaises(ValueError):
            create_model_from_config(config_invalid, self.device)


if __name__ == '__main__':
    unittest.main()