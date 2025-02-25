"""
test_training.py

Unit tests for training functionality in the Discrete Diffusion project.
Tests the training engine, loss strategies, and training/validation loops.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from training import (
    run_training_loop, run_validation_loop, LossStrategy,
    StandardLossStrategy, DDMLossStrategy, DoTLossStrategy
)
from training.engine import TrainingEngine


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    def __init__(self):
        super(MockModel, self).__init__()
        self.num_tokens = 10
        self.linear = torch.nn.Linear(10, 10)
    
    def forward(self, x, t, clue_mask):
        batch_size = x.size(0)
        return torch.randn(batch_size, 9, 9, self.num_tokens, device=x.device)


class MockLossStrategy(LossStrategy):
    """Mock loss strategy for testing."""
    def compute_losses(self, model, batch, t, num_timesteps, device, **kwargs):
        return {
            "total_loss": torch.tensor(0.5, device=device),
            "ce_loss": torch.tensor(0.3, device=device),
            "constraint_loss": torch.tensor(0.2, device=device)
        }
    
    def get_metric_keys(self):
        return ["total_loss", "ce_loss", "constraint_loss"]


class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for testing."""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            "solved_board": torch.randint(1, 10, (9, 9)),
            "puzzle_board": torch.randint(0, 10, (9, 9)),
            "clue_mask": torch.randint(0, 2, (9, 9), dtype=torch.float)
        }


class TestTrainingEngine(unittest.TestCase):
    """Test suite for training engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = MockModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_strategy = MockLossStrategy()
        self.batch_size = 2
        self.num_timesteps = 10
        
        # Create a dataset and dataloader
        self.dataset = MockDataset(num_samples=10)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def test_training_engine_init(self):
        """Test TrainingEngine initialization."""
        engine = TrainingEngine(
            model=self.model,
            optimizer=self.optimizer,
            loss_strategy=self.loss_strategy,
            device=self.device
        )
        
        # Check that engine has the right attributes
        self.assertEqual(engine.model, self.model)
        self.assertEqual(engine.optimizer, self.optimizer)
        self.assertEqual(engine.loss_strategy, self.loss_strategy)
        self.assertEqual(engine.device, self.device)
        
        # Check initial state
        self.assertEqual(engine.global_step, 0)
        self.assertEqual(engine.best_val_loss, float("inf"))
    
    def test_training_engine_train_epoch(self):
        """Test TrainingEngine train_epoch method."""
        engine = TrainingEngine(
            model=self.model,
            optimizer=self.optimizer,
            loss_strategy=self.loss_strategy,
            device=self.device
        )
        
        # Train for one epoch
        metrics = engine.train_epoch(
            dataloader=self.dataloader,
            epoch=0,
            num_epochs=1,
            num_timesteps=self.num_timesteps,
            lambda_constraint=1.0
        )
        
        # Check that metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)
        self.assertIn("ce_loss", metrics)
        self.assertIn("constraint_loss", metrics)
        
        # Check that global step was updated
        self.assertEqual(engine.global_step, len(self.dataloader))
    
    def test_training_engine_validate(self):
        """Test TrainingEngine validate method."""
        engine = TrainingEngine(
            model=self.model,
            optimizer=self.optimizer,
            loss_strategy=self.loss_strategy,
            device=self.device
        )
        
        # Run validation
        metrics = engine.validate(
            dataloader=self.dataloader,
            num_timesteps=self.num_timesteps,
            lambda_constraint=1.0
        )
        
        # Check that metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)
        self.assertIn("ce_loss", metrics)
        self.assertIn("constraint_loss", metrics)
        
        # Check that global step was not updated
        self.assertEqual(engine.global_step, 0)
    
    def test_run_training_loop(self):
        """Test run_training_loop function."""
        # Run training loop for 1 epoch
        metrics_history = run_training_loop(
            model=self.model,
            dataloader=self.dataloader,
            num_timesteps=self.num_timesteps,
            optimizer=self.optimizer,
            device=self.device,
            num_epochs=1,
            loss_strategy=self.loss_strategy
        )
        
        # Check that metrics history was returned
        self.assertIsInstance(metrics_history, dict)
        self.assertIn("train_total_loss", metrics_history)
        self.assertIn("train_ce_loss", metrics_history)
        self.assertIn("train_constraint_loss", metrics_history)
        
        # Check that metrics were tracked
        self.assertEqual(len(metrics_history["train_total_loss"]), 1)
    
    def test_run_validation_loop(self):
        """Test run_validation_loop function."""
        # Run validation loop
        metrics = run_validation_loop(
            model=self.model,
            dataloader=self.dataloader,
            num_timesteps=self.num_timesteps,
            device=self.device,
            loss_strategy=self.loss_strategy
        )
        
        # Check that metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertIn("total_loss", metrics)
        self.assertIn("ce_loss", metrics)
        self.assertIn("constraint_loss", metrics)


class TestLossStrategies(unittest.TestCase):
    """Test suite for loss strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = MockModel().to(self.device)
        self.batch_size = 2
        self.num_tokens = 10
        self.num_timesteps = 10
        
        # Create batch data
        self.solved_board = torch.randint(1, 10, (self.batch_size, 9, 9), device=self.device)
        self.puzzle_board = self.solved_board.clone()
        self.clue_mask = torch.zeros((self.batch_size, 9, 9), device=self.device)
        self.clue_mask[:, :2, :2] = 1
        self.puzzle_board[self.clue_mask == 0] = 0
        
        self.batch = {
            "solved_board": self.solved_board,
            "puzzle_board": self.puzzle_board,
            "clue_mask": self.clue_mask
        }
        
        self.t = torch.full((self.batch_size, 1), 5, device=self.device, dtype=torch.float)
    
    def test_standard_loss_strategy(self):
        """Test StandardLossStrategy."""
        strategy = StandardLossStrategy()
        
        # Check metric keys
        self.assertEqual(
            set(strategy.get_metric_keys()),
            {"total_loss", "ce_loss", "constraint_loss"}
        )
        
        # Compute losses
        loss_dict = strategy.compute_losses(
            self.model, self.batch, self.t, self.num_timesteps, self.device
        )
        
        # Check that all expected keys are present
        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("constraint_loss", loss_dict)
        
        # Check that losses are tensors
        for key, val in loss_dict.items():
            self.assertIsInstance(val, torch.Tensor)
            self.assertEqual(val.dim(), 0)  # Scalar tensor
    
    def test_ddm_loss_strategy(self):
        """Test DDMLossStrategy."""
        strategy = DDMLossStrategy()
        
        # Check metric keys
        self.assertEqual(
            set(strategy.get_metric_keys()),
            {"total_loss", "ce_loss", "constraint_loss", "evidence_loss"}
        )
        
        # Compute losses
        loss_dict = strategy.compute_losses(
            self.model, self.batch, self.t, self.num_timesteps, self.device
        )
        
        # Check that all expected keys are present
        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("constraint_loss", loss_dict)
        self.assertIn("evidence_loss", loss_dict)
    
    def test_dot_loss_strategy(self):
        """Test DoTLossStrategy."""
        strategy = DoTLossStrategy()
        
        # Check metric keys
        self.assertEqual(
            set(strategy.get_metric_keys()),
            {"total_loss", "ce_loss", "constraint_loss", "trajectory_loss"}
        )
        
        # Compute losses
        loss_dict = strategy.compute_losses(
            self.model, self.batch, self.t, self.num_timesteps, self.device
        )
        
        # Check that all expected keys are present
        self.assertIn("total_loss", loss_dict)
        self.assertIn("ce_loss", loss_dict)
        self.assertIn("constraint_loss", loss_dict)
        self.assertIn("trajectory_loss", loss_dict)


if __name__ == '__main__':
    unittest.main()