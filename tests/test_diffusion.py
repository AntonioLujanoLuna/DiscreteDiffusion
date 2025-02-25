"""
test_diffusion.py

Unit tests for diffusion processes in the Discrete Diffusion project.
Tests both forward and reverse diffusion functionality.
"""

import torch
import numpy as np
import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from diffusion import forward_process, reverse_diffusion_inference
from utils import LearnedNoiseSchedule


class TestDiffusionProcesses(unittest.TestCase):
    """Test suite for diffusion processes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_tokens = 10
        self.num_timesteps = 10
        
        # Create a simple board
        self.x0 = torch.randint(1, 10, (self.batch_size, 9, 9), device=self.device)
        self.clue_mask = torch.zeros((self.batch_size, 9, 9), device=self.device)
        self.clue_mask[:, :2, :2] = 1  # Set some cells as clues
        
    def test_forward_process(self):
        """Test forward diffusion process."""
        # Test with different timesteps
        for t_val in [1, 5, 10]:
            t = torch.full((self.batch_size, 1), t_val, device=self.device, dtype=torch.float)
            
            # Apply forward process
            x_t = forward_process(self.x0, t, self.num_timesteps, self.clue_mask)
            
            # Check shape
            self.assertEqual(x_t.shape, self.x0.shape)
            
            # Check that clue cells are preserved
            clue_cells = torch.where(self.clue_mask == 1)
            for b, i, j in zip(*clue_cells):
                self.assertEqual(x_t[b, i, j].item(), self.x0[b, i, j].item())
            
            # Check that non-clue cells are noised
            if t_val == self.num_timesteps:  # At max timestep, most cells should be noised
                non_clue_cells = torch.where(self.clue_mask == 0)
                noised_count = 0
                total_non_clues = len(non_clue_cells[0])
                
                for b, i, j in zip(*non_clue_cells):
                    if x_t[b, i, j].item() != self.x0[b, i, j].item():
                        noised_count += 1
                
                # Check that a substantial portion of non-clue cells were noised
                self.assertGreater(noised_count / total_non_clues, 0.7)
    
    def test_learned_noise_schedule(self):
        """Test learned noise schedule."""
        # Create learned noise schedule
        noise_schedule = LearnedNoiseSchedule(hidden_dim=16).to(self.device)
        
        # Test with different timesteps
        for t_val in [1, 5, 10]:
            t = torch.full((self.batch_size, 1), t_val, device=self.device, dtype=torch.float)
            
            # Get noise probability
            noise_prob = noise_schedule(t, self.num_timesteps)
            
            # Check shape
            self.assertEqual(noise_prob.shape, (self.batch_size, 1))
            
            # Check that noise probability is between 0 and 1
            self.assertTrue((noise_prob >= 0).all() and (noise_prob <= 1).all())


class TestReverseProcess(unittest.TestCase):
    """Test suite for reverse diffusion."""
    
    def setUp(self):
        """Set up test fixtures including a simple model."""
        self.device = torch.device("cpu")
        self.num_tokens = 10
        self.num_timesteps = 5
        
        # Import here to avoid circular imports
        from models import ImprovedSudokuDenoiser
        
        # Create a simple model
        self.model = ImprovedSudokuDenoiser(
            num_tokens=self.num_tokens,
            hidden_dim=32,
            num_layers=2,
            nhead=4
        ).to(self.device)
        
        # Create a simple board
        self.solved_board = torch.randint(1, 10, (1, 9, 9), device=self.device)
        self.clue_mask = torch.zeros((1, 9, 9), device=self.device)
        self.clue_mask[0, :2, :2] = 1  # Set some cells as clues
        
        # Create initial noisy board
        self.initial_board = self.solved_board.clone()
        self.initial_board[self.clue_mask == 0] = 0  # Set non-clue cells to noise token
        
    def test_reverse_diffusion_inference(self):
        """Test reverse diffusion inference."""
        # Run reverse diffusion
        trajectory = reverse_diffusion_inference(
            self.model, self.initial_board, self.clue_mask, self.num_timesteps, self.device
        )
        
        # Check trajectory length
        self.assertEqual(len(trajectory), self.num_timesteps + 1)
        
        # Check that each board in trajectory has the right shape
        for board in trajectory:
            self.assertEqual(board.shape, (1, 9, 9))
        
        # Check that clue cells are preserved throughout
        for board_np in trajectory:
            board = torch.tensor(board_np, device=self.device)
            clue_cells = torch.where(self.clue_mask == 1)
            for b, i, j in zip(*clue_cells):
                self.assertEqual(board[b, i, j].item(), self.solved_board[b, i, j].item())


if __name__ == '__main__':
    unittest.main()