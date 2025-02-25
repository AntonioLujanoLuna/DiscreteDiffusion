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

from models import ImprovedSudokuDenoiser, HybridSudokuDenoiser
from models.components import SinusoidalTimeEmbedding
from utils.diffusion_utils import enforce_clue_logits


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
    
    def test_sinusoidal_time_embedding(self):
        """Test SinusoidalTimeEmbedding."""
        embedding = SinusoidalTimeEmbedding(hidden_dim=self.hidden_dim).to(self.device)
        
        # Test with different timesteps
        for t_val in [1, 50, 100]:
            t = torch.full((self.batch_size, 1), t_val, device=self.device, dtype=torch.float)
            
            # Get embedding
            t_emb = embedding(t)
            
            # Check shape
            self.assertEqual(t_emb.shape, (self.batch_size, self.hidden_dim))
            
            # Check that values are not all the same (basic check)
            self.assertTrue(t_emb.unique().shape[0] > 1)


if __name__ == '__main__':
    unittest.main()