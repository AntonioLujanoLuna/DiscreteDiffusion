"""
Checkpoint management functionality for the Discrete Diffusion project.

This module provides classes and functions for saving, loading, and managing checkpoints.
"""

import os
import torch
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, List, Union

class CheckpointManager:
    """
    Manages model checkpoints for saving, loading, and resuming training.
    
    Features:
    - Save checkpoints with model, optimizer, scheduler, and training state
    - Save best model based on validation metrics
    - Load checkpoints for training resumption
    - Track and manage checkpoint history
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 experiment_name: str,
                 max_to_keep: int = 5,
                 save_best_only: bool = False) -> None:
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory to store checkpoints
            experiment_name (str): Name of the experiment for organization
            max_to_keep (int): Maximum number of checkpoints to keep
            save_best_only (bool): Whether to save only the best checkpoint based on validation
        """
        self.base_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.max_to_keep = max_to_keep
        self.save_best_only = save_best_only
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{timestamp}"
        self.checkpoint_dir = os.path.join(self.base_dir, self.run_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize checkpoint tracking
        self.checkpoint_history = []
        self.best_metric_value = float('inf')  # Lower is better
        self.best_checkpoint_path = None
        
        # Get logger
        try:
            from logger import get_logger
            self.logger = get_logger(experiment_name=self.experiment_name)
            if self.logger is None:
                # Create a simple logger
                self.logger = type('SimpleLogger', (), {
                    'info': lambda _, msg: print(f"[INFO] {msg}"),
                    'warning': lambda _, msg: print(f"[WARNING] {msg}"),
                    'error': lambda _, msg: print(f"[ERROR] {msg}"),
                    'debug': lambda _, msg: print(f"[DEBUG] {msg}")
                })()
        except (ImportError, AttributeError):
            # If logger module is not available, use a simple print function
            self.logger = type('SimpleLogger', (), {
                'info': lambda _, msg: print(f"[INFO] {msg}"),
                'warning': lambda _, msg: print(f"[WARNING] {msg}"),
                'error': lambda _, msg: print(f"[ERROR] {msg}"),
                'debug': lambda _, msg: print(f"[DEBUG] {msg}")
            })()
                
        self.logger.info(f"Checkpoint manager initialized at {self.checkpoint_dir}")
    
    def _get_checkpoint_filename(self, epoch: int, step: Optional[int] = None) -> str:
        """
        Generate a checkpoint filename.
        
        Args:
            epoch (int): Current epoch number
            step (int, optional): Current step/iteration
            
        Returns:
            str: Checkpoint filename
        """
        if step is not None:
            return f"checkpoint_epoch{epoch:03d}_step{step:07d}.pt"
        else:
            return f"checkpoint_epoch{epoch:03d}.pt"
    
    def save(self, 
             state_dict: Dict[str, Any],
             epoch: int,
             step: Optional[int] = None,
             metric_value: Optional[float] = None,
             is_best: bool = False) -> str:
        """
        Save a checkpoint.
        
        Args:
            state_dict (Dict[str, Any]): Dictionary containing model state, optimizer state, etc.
            epoch (int): Current epoch number
            step (int, optional): Current step/iteration
            metric_value (float, optional): Validation metric value for determining best model
            is_best (bool): Flag to force this checkpoint as the best one
            
        Returns:
            str: Path to the saved checkpoint
        """
        # Create checkpoint filename and path
        filename = self._get_checkpoint_filename(epoch, step)
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        # Add metadata to the state dict
        save_dict = {
            **state_dict,
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metric_value': metric_value
        }
        
        # Save the checkpoint
        torch.save(save_dict, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Update checkpoint history
        self.checkpoint_history.append((checkpoint_path, epoch, step, metric_value))
        
        # Check if this is the best model
        if metric_value is not None and (is_best or metric_value < self.best_metric_value):
            self.best_metric_value = metric_value
            self.best_checkpoint_path = checkpoint_path
            
            # Save a copy as the best model
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            shutil.copy(checkpoint_path, best_path)
            self.logger.info(f"New best model saved with metric value: {metric_value:.4f}")
        
        # Clean up old checkpoints if needed
        if not self.save_best_only and len(self.checkpoint_history) > self.max_to_keep:
            # Sort by epoch and step
            self.checkpoint_history.sort(key=lambda x: (x[1], x[2] or 0))
            while len(self.checkpoint_history) > self.max_to_keep:
                old_path, _, _, _ = self.checkpoint_history.pop(0)
                if old_path != self.best_checkpoint_path and os.path.exists(old_path):
                    os.remove(old_path)
                    self.logger.debug(f"Removed old checkpoint: {old_path}")
        
        return checkpoint_path
    
    def load(self, 
             checkpoint_path: str = None, 
             device: torch.device = None, 
             load_best: bool = False) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path (str, optional): Path to the checkpoint to load
            device (torch.device, optional): Device to load the model to
            load_best (bool): Whether to load the best checkpoint
            
        Returns:
            Dict[str, Any]: Loaded checkpoint state_dict
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
        """
        if load_best and self.best_checkpoint_path is not None:
            checkpoint_path = self.best_checkpoint_path
        elif load_best:
            # Try to find best_model.pt
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if os.path.exists(best_path):
                checkpoint_path = best_path
            else:
                self.logger.warning("No best model found, trying to load the latest checkpoint")
                if not self.checkpoint_history:
                    raise FileNotFoundError("No checkpoints found")
                # Sort by epoch and step in reverse order
                sorted_history = sorted(self.checkpoint_history, key=lambda x: (x[1], x[2] or 0), reverse=True)
                checkpoint_path = sorted_history[0][0]
        
        if checkpoint_path is None:
            # Try to find the latest checkpoint
            if not self.checkpoint_history:
                raise FileNotFoundError("No checkpoints found")
            # Sort by epoch and step in reverse order
            sorted_history = sorted(self.checkpoint_history, key=lambda x: (x[1], x[2] or 0), reverse=True)
            checkpoint_path = sorted_history[0][0]
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load the checkpoint
        if device is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Extract epoch and step if available
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', None)
        metric_value = checkpoint.get('metric_value', None)
        self.logger.info(f"Loaded state from epoch {epoch}" + 
                         (f", step {step}" if step is not None else "") +
                         (f", metric {metric_value:.4f}" if metric_value is not None else ""))
        
        return checkpoint
    
    def save_experiment_metadata(self, config: Any) -> None:
        """
        Save experiment metadata along with checkpoints.
        
        Args:
            config (Any): Configuration object or dictionary
        """
        # Convert config to dictionary if it's not already
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif hasattr(config, '__dict__'):
            config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
        else:
            config_dict = dict(config)
        
        # Save to JSON file
        metadata_path = os.path.join(self.checkpoint_dir, "experiment_config.json")
        with open(metadata_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info(f"Saved experiment metadata to {metadata_path}")


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    checkpoint_dir: str,
                    scheduler: Optional[Any] = None,
                    metric_value: Optional[float] = None,
                    **kwargs) -> str:
    """
    Convenience function to save a checkpoint with model and optimizer state.
    
    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Current epoch number
        checkpoint_dir (str): Directory to save checkpoint to
        scheduler (Any, optional): Learning rate scheduler
        metric_value (float, optional): Validation metric value
        **kwargs: Additional items to save in the checkpoint
        
    Returns:
        str: Path to the saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    
    if scheduler is not None:
        state_dict['scheduler_state_dict'] = scheduler.state_dict()
        
    # Add any additional kwargs
    state_dict.update(kwargs)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch:03d}.pt")
    torch.save(state_dict, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: str,
                    model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Convenience function to load a checkpoint with model and optimizer state.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model (torch.nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        scheduler (Any, optional): Learning rate scheduler to load state into
        device (torch.device, optional): Device to load tensors to
        
    Returns:
        Dict[str, Any]: The loaded checkpoint state dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint