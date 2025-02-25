"""
Validation utilities for the Discrete Diffusion project.

This module provides functions for validating diffusion models on Sudoku puzzles.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

def validate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_timesteps: int,
    device: torch.device,
    loss_strategy: Optional[Any] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Validate a diffusion model on a dataset.
    
    This is a convenience wrapper around run_validation_loop that simplifies
    the validation process for common use cases.
    
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (DataLoader): Validation dataloader.
        num_timesteps (int): Number of diffusion timesteps.
        device (torch.device): Device for computation.
        loss_strategy (Any, optional): Strategy for computing losses.
        **kwargs: Additional arguments to pass to run_validation_loop.
        
    Returns:
        Dict[str, float]: Dictionary of validation metrics.
    """
    # Import here to avoid circular imports
    from .engine import run_validation_loop
    
    return run_validation_loop(
        model=model,
        dataloader=dataloader,
        num_timesteps=num_timesteps,
        device=device,
        loss_strategy=loss_strategy,
        **kwargs
    )