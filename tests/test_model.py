"""
test_model.py

Unit tests for model components of the Discrete Diffusion project.
Ensures that model architectures are implemented correctly and produce
the expected outputs.
"""

import torch
import unittest
import sys
import os

# Add the src directory to the path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import ImprovedSudokuDenoiser, HybridSudokuDenoiser
from utils import enforce_clue_logits


class TestModelArchitectures(unittest.TestCase):
    """Test suite for model architectures."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.num_tokens = 10
        self.hidden_dim = 64
        self.num_layers = 2
        self.nhead = 4
        
        # Create sample inputs
        self.board = torch.randint(0, self.num_tokens, (self.batch_size, 9, 9), device=self.device)
        self.t = torch.full((self.batch_size, 1), 50, device=self.device, dtype=torch.float)
        self.clue_mask = torch.zeros((self.batch_size, 9, 9), device=self.device)
        self.clue_mask[:, :2, :2] = 1  # Set some cells as clues
    
    def test_improved_sudoku_denoiser(self):
        """Test ImprovedSudokuDenoiser forward pass."""
        model = ImprovedSudokuDenoiser(
            num_tokens=self.num_tokens,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            nhead=self.nhead
        ).to(self.device)
        
        # Test forward pass
        logits = model(self.board, self.t, self.clue_mask)
        
        # Check shape
        self.assertEqual(logits.shape, (self.batch_size, 9, 9, self.num_tokens))
        
        # Check that clue cells have forced logits
        clue_cells = torch.where(self.clue_mask == 1)
        for b, i, j in zip(*clue_cells):
            token = self.board[b, i, j]
            pred_token = logits[b, i, j].argmax().item()
            self.assertEqual(pred_token, token, f"Clue cell at {b},{i},{j} should predict token {token}, got {pred_token}")
    
    def test_hybrid_sudoku_denoiser(self):
        """Test HybridSudokuDenoiser forward pass."""
        model = HybridSudokuDenoiser(
            num_tokens=self.num_tokens,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            nhead=self.nhead,
            num_conv_layers=2,
            conv_kernel_size=3
        ).to(self.device)
        
        # Test forward pass
        logits = model(self.board, self.t, self.clue_mask)
        
        # Check shape
        self.assertEqual(logits.shape, (self.batch_size, 9, 9, self.num_tokens))
        
        # Check that clue cells have forced logits
        clue_cells = torch.where(self.clue_mask == 1)
        for b, i, j in zip(*clue_cells):
            token = self.board[b, i, j]
            pred_token = logits[b, i, j].argmax().item()
            self.assertEqual(pred_token, token, f"Clue cell at {b},{i},{j} should predict token {token}, got {pred_token}")
    
    def test_enforce_clue_logits(self):
        """Test enforce_clue_logits function."""
        # Create random logits
        logits = torch.randn((self.batch_size, 9, 9, self.num_tokens), device=self.device)
        
        # Apply enforce_clue_logits
        modified_logits = enforce_clue_logits(logits, self.board, self.clue_mask)
        
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
            self.assertTrue(torch.allclose(modified_logits[b, i, j], logits[b, i, j]))


if __name__ == '__main__':
    unittest.main()