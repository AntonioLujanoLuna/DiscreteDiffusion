"""
test_losses.py

Unit tests for loss functions in the Discrete Diffusion project.
Tests various loss strategies and constraint losses.
"""

import torch
import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from training.losses import (
    StandardLossStrategy, DDMLossStrategy, DoTLossStrategy,
    compute_constraint_loss, compute_uniqueness_loss
)


class TestLossFunctions(unittest.TestCase):
    """Test suite for loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.batch_size = 2
        self.num_tokens = 10
        self.num_timesteps = 100
        
        # Create sample inputs
        self.logits = torch.randn((self.batch_size, 9, 9, self.num_tokens), device=self.device)
        self.solved_board = torch.randint(1, 10, (self.batch_size, 9, 9), device=self.device)
        self.puzzle_board = self.solved_board.clone()
        self.clue_mask = torch.zeros((self.batch_size, 9, 9), device=self.device)
        self.clue_mask[:, :2, :2] = 1  # Set some cells as clues
        
        # For non-clue cells, set puzzle board to 0
        self.puzzle_board[self.clue_mask == 0] = 0
        
        self.batch = {
            "solved_board": self.solved_board,
            "puzzle_board": self.puzzle_board,
            "clue_mask": self.clue_mask
        }
        
        self.t = torch.full((self.batch_size, 1), 50, device=self.device, dtype=torch.float)
        
        # Mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_tokens = 10
            
            def forward(self, x, t, clue_mask):
                return torch.randn((x.size(0), 9, 9, self.num_tokens), device=x.device)
        
        self.model = MockModel()
    
    def test_constraint_loss(self):
        """Test constraint loss calculation."""
        loss = compute_constraint_loss(self.logits)
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)
        
        # Create a perfect solution and check that loss is near zero
        perfect_logits = torch.zeros_like(self.logits)
        # For each cell, set one token to have very high probability
        for i in range(9):
            for j in range(9):
                token = (i * 3 + j) % 9 + 1  # Assign a token (1-9) to each cell
                perfect_logits[:, i, j, token] = 10.0
        
        perfect_loss = compute_constraint_loss(perfect_logits)
        self.assertLess(perfect_loss.item(), 0.01)
    
    def test_uniqueness_loss(self):
        """Test uniqueness loss calculation."""
        loss = compute_uniqueness_loss(self.logits, self.clue_mask)
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check that loss is non-negative
        self.assertGreaterEqual(loss.item(), 0)
        
        # Create deterministic predictions and check that loss is near zero
        deterministic_logits = torch.zeros_like(self.logits)
        for i in range(9):
            for j in range(9):
                token = (i * 3 + j) % 9 + 1  # Assign a token (1-9) to each cell
                deterministic_logits[:, i, j, token] = 10.0
        
        deterministic_loss = compute_uniqueness_loss(deterministic_logits, self.clue_mask)
        self.assertLess(deterministic_loss.item(), 0.01)
    
    def test_standard_loss_strategy(self):
        """Test standard loss strategy."""
        strategy = StandardLossStrategy()
        
        # Compute losses
        loss_dict = strategy.compute_losses(
            self.model, self.batch, self.t, self.num_timesteps, self.device
        )
        
        # Check that all expected keys are present
        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("constraint_loss", loss_dict)
        
        # Check that metric keys are correct
        self.assertEqual(
            set(strategy.get_metric_keys()),
            {"total_loss", "ce_loss", "constraint_loss"}
        )
    
    def test_ddm_loss_strategy(self):
        """Test DDM loss strategy."""
        strategy = DDMLossStrategy()
        
        # Compute losses
        loss_dict = strategy.compute_losses(
            self.model, self.batch, self.t, self.num_timesteps, self.device
        )
        
        # Check that all expected keys are present
        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("constraint_loss", loss_dict)
        self.assertIn("evidence_loss", loss_dict)
        
        # Check that metric keys are correct
        self.assertEqual(
            set(strategy.get_metric_keys()),
            {"total_loss", "ce_loss", "constraint_loss", "evidence_loss"}
        )
    
    def test_dot_loss_strategy(self):
        """Test DoT loss strategy."""
        strategy = DoTLossStrategy()
        
        # Compute losses
        loss_dict = strategy.compute_losses(
            self.model, self.batch, self.t, self.num_timesteps, self.device
        )
        
        # Check that all expected keys are present
        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("constraint_loss", loss_dict)
        self.assertIn("trajectory_loss", loss_dict)
        
        # Check that metric keys are correct
        self.assertEqual(
            set(strategy.get_metric_keys()),
            {"total_loss", "ce_loss", "constraint_loss", "trajectory_loss"}
        )


if __name__ == '__main__':
    unittest.main()