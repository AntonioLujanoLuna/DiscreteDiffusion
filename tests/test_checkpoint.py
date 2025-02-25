"""
test_checkpoint.py

Unit tests for checkpoint functionality in the Discrete Diffusion project.
Tests saving, loading, and managing model checkpoints.
"""

import unittest
import torch
import os
import tempfile
import shutil
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from checkpoint import CheckpointManager, save_checkpoint, load_checkpoint


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


class TestCheckpoint(unittest.TestCase):
    """Test suite for checkpoint functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create model, optimizer, etc.
        self.device = torch.device("cpu")
        self.model = SimpleModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.experiment_name = "test_experiment"
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            checkpoint_dir=self.temp_dir
        )
        
        # Check that checkpoint file was created
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Load checkpoint
        new_model = SimpleModel().to(self.device)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        loaded_checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            device=self.device
        )
        
        # Check that checkpoint contains the right keys
        self.assertIn("model_state_dict", loaded_checkpoint)
        self.assertIn("optimizer_state_dict", loaded_checkpoint)
        self.assertIn("epoch", loaded_checkpoint)
        
        # Check that epoch was correctly loaded
        self.assertEqual(loaded_checkpoint["epoch"], 5)
        
        # Check that model parameters were loaded correctly
        for p1, p2 in zip(self.model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
        
        # Check optimizer state
        self.assertEqual(
            self.optimizer.state_dict()["param_groups"][0]["lr"],
            new_optimizer.state_dict()["param_groups"][0]["lr"]
        )
    
    def test_checkpoint_manager_save(self):
        """Test CheckpointManager save functionality."""
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            max_to_keep=3
        )
        
        # Save checkpoints
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        
        for epoch in range(5):
            path = manager.save(
                state_dict=state_dict,
                epoch=epoch,
                metric_value=10.0 - epoch
            )
            self.assertTrue(os.path.exists(path))
        
        # Check that only the last 3 checkpoints are kept (max_to_keep=3)
        self.assertEqual(len(manager.checkpoint_history), 3)
        
        # Check that the checkpoints with the highest epochs are kept
        epochs = [item[1] for item in manager.checkpoint_history]
        self.assertEqual(sorted(epochs), [2, 3, 4])
        
        # Check that the best checkpoint was saved
        self.assertIsNotNone(manager.best_checkpoint_path)
        self.assertTrue(os.path.exists(manager.best_checkpoint_path))
    
    def test_checkpoint_manager_load(self):
        """Test CheckpointManager load functionality."""
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            experiment_name=self.experiment_name,
            max_to_keep=3
        )
        
        # Save some checkpoints
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        
        for epoch in range(3):
            manager.save(
                state_dict=state_dict,
                epoch=epoch,
                metric_value=10.0 - epoch
            )
        
        # Load the best checkpoint
        loaded_checkpoint = manager.load(load_best=True)
        
        # Check that checkpoint contains the right keys
        self.assertIn("model_state_dict", loaded_checkpoint)
        self.assertIn("optimizer_state_dict", loaded_checkpoint)
        
        # Check that the right epoch was loaded (epoch 2 should have the best metric)
        self.assertEqual(loaded_checkpoint["epoch"], 2)
    
    def test_save_experiment_metadata(self):
        """Test saving experiment metadata."""
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=self.temp_dir,
            experiment_name=self.experiment_name
        )
        
        # Create a simple config
        config = {
            "model": {
                "type": "Base",
                "hidden_dim": 128
            },
            "training": {
                "num_epochs": 10,
                "learning_rate": 0.001
            }
        }
        
        # Save metadata
        manager.save_experiment_metadata(config)
        
        # Check that metadata file was created
        metadata_path = os.path.join(manager.checkpoint_dir, "experiment_config.json")
        self.assertTrue(os.path.exists(metadata_path))
        
        # Check that metadata contains the right data
        import json
        with open(metadata_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["model"]["type"], "Base")
        self.assertEqual(loaded_config["model"]["hidden_dim"], 128)
        self.assertEqual(loaded_config["training"]["num_epochs"], 10)
        self.assertEqual(loaded_config["training"]["learning_rate"], 0.001)


if __name__ == '__main__':
    unittest.main()