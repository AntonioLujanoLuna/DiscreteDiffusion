"""
test_inference.py

Unit tests for inference functionality in the Discrete Diffusion project.
Tests inference strategies and engine.
"""

import torch
import numpy as np
import unittest
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference import (
    run_inference, InferenceEngine, 
    StandardInferenceStrategy, DDMInferenceStrategy, DoTInferenceStrategy
)


class TestInferenceStrategies(unittest.TestCase):
    """Test suite for inference strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
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
    
    def test_standard_strategy(self):
        """Test standard inference strategy."""
        strategy = StandardInferenceStrategy()
        
        # Initialize state
        state = strategy._initialize_state(
            self.model, self.initial_board, self.clue_mask, self.device
        )
        
        # Check that state has the expected keys
        self.assertIn("current_board", state)
        
        # Create probs for testing update step
        probs = torch.zeros((1, 9, 9, self.num_tokens), device=self.device)
        for i in range(9):
            for j in range(9):
                token = (i + j) % 9 + 1
                probs[0, i, j, token] = 1.0
        
        # Test update step
        state = strategy._update_step(state, probs, 5, self.clue_mask)
        
        # Check that board was updated correctly
        self.assertIn("current_board", state)
        self.assertEqual(state["current_board"].shape, (1, 9, 9))
        
        # Check finalize results
        trajectory = [self.initial_board.cpu().numpy(), state["current_board"].cpu().numpy()]
        result = strategy._finalize_results(state, trajectory)
        
        # Check that result is a list of boards
        self.assertEqual(len(result), 2)
    
    def test_ddm_strategy(self):
        """Test DDM inference strategy."""
        strategy = DDMInferenceStrategy()
        
        # Initialize state
        state = strategy._initialize_state(
            self.model, self.initial_board, self.clue_mask, self.device
        )
        
        # Check that state has the expected keys
        self.assertIn("current_board", state)
        self.assertIn("accumulator", state)
        self.assertIn("decision_made", state)
        self.assertIn("final_decision", state)
        
        # Create probs for testing update step
        probs = torch.zeros((1, 9, 9, self.num_tokens), device=self.device)
        for i in range(9):
            for j in range(9):
                token = (i + j) % 9 + 1
                probs[0, i, j, token] = 1.0
        
        # Test update step
        state = strategy._update_step(
            state, probs, 5, self.clue_mask, threshold=0.9, beta=0.9, tau=0.05, num_timesteps=10
        )
        
        # Check that state was updated correctly
        for key in ["current_board", "accumulator", "decision_made", "final_decision"]:
            self.assertIn(key, state)
        
        # Check finalize results
        trajectory = [self.initial_board.cpu().numpy(), state["current_board"].cpu().numpy()]
        result = strategy._finalize_results(state, trajectory)
        
        # Check that result is a list of boards
        self.assertEqual(len(result), 2)
    
    def test_dot_strategy(self):
        """Test DoT inference strategy."""
        strategy = DoTInferenceStrategy()
        
        # Test run_inference method directly as it overrides the template method
        result = strategy.run_inference(
            self.model, self.initial_board, self.clue_mask, self.num_timesteps, self.device,
            num_trajectories=2
        )
        
        # Check that it returns a tuple (final_board, trajectories)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(len(result), 2)
        
        # Check that final_board is a numpy array with shape (1, 9, 9)
        final_board, trajectories = result
        self.assertTrue(isinstance(final_board, np.ndarray))
        self.assertEqual(final_board.shape, (1, 9, 9))
        
        # Check that trajectories is a list of trajectories
        self.assertTrue(isinstance(trajectories, list))
        self.assertEqual(len(trajectories), 2)  # num_trajectories=2
        
        # Check that each trajectory is a list of boards
        for traj in trajectories:
            self.assertTrue(isinstance(traj, list))
            self.assertGreater(len(traj), 0)
    
    def test_inference_engine(self):
        """Test inference engine."""
        # Create engine with standard strategy
        strategy = StandardInferenceStrategy()
        engine = InferenceEngine(
            model=self.model,
            device=self.device,
            num_timesteps=self.num_timesteps,
            strategy=strategy
        )
        
        # Test prepare initial board
        initial_board = engine.prepare_initial_board(self.solved_board, self.clue_mask)
        
        # Check shape
        self.assertEqual(initial_board.shape, self.solved_board.shape)
        
        # Check that clue cells are preserved
        clue_cells = torch.where(self.clue_mask == 1)
        for b, i, j in zip(*clue_cells):
            self.assertEqual(initial_board[b, i, j].item(), self.solved_board[b, i, j].item())
        
        # Test run_inference
        trajectory = engine.run_inference(self.solved_board, self.clue_mask)
        
        # Check that trajectory is a list of boards
        self.assertGreater(len(trajectory), 0)
    
    def test_run_inference(self):
        """Test run_inference function."""
        # Test with standard strategy
        trajectory_standard = run_inference(
            model=self.model,
            solved_board=self.solved_board,
            clue_mask=self.clue_mask,
            num_timesteps=self.num_timesteps,
            device=self.device,
            mode="standard"
        )
        
        # Check that it returns a list of boards
        self.assertTrue(isinstance(trajectory_standard, list))
        self.assertGreater(len(trajectory_standard), 0)
        
        # Test with ddm strategy
        trajectory_ddm = run_inference(
            model=self.model,
            solved_board=self.solved_board,
            clue_mask=self.clue_mask,
            num_timesteps=self.num_timesteps,
            device=self.device,
            mode="ddm",
            threshold=0.9
        )
        
        # Check that it returns a list of boards
        self.assertTrue(isinstance(trajectory_ddm, list))
        self.assertGreater(len(trajectory_ddm), 0)
        
        # Test with dot strategy
        result_dot = run_inference(
            model=self.model,
            solved_board=self.solved_board,
            clue_mask=self.clue_mask,
            num_timesteps=self.num_timesteps,
            device=self.device,
            mode="dot",
            num_trajectories=2
        )
        
        # Check that it returns a tuple (final_board, trajectories)
        self.assertTrue(isinstance(result_dot, tuple))
        self.assertEqual(len(result_dot), 2)
        
        # Test with invalid mode
        with self.assertRaises(ValueError):
            run_inference(
                model=self.model,
                solved_board=self.solved_board,
                clue_mask=self.clue_mask,
                num_timesteps=self.num_timesteps,
                device=self.device,
                mode="invalid"
            )


if __name__ == '__main__':
    unittest.main()