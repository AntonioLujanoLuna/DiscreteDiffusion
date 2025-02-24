"""
Common utility functions for the Discrete Diffusion project.

This module provides general-purpose utility functions that are used across the project.
"""

import random
import torch
import numpy as np
from typing import List, Union

def set_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def validate_trajectory(trajectory: List[Union[np.ndarray, List[List[int]]]]) -> List[np.ndarray]:
    """
    Validates that each board in the trajectory is a 2D numpy array with shape (9, 9).
    If a board has a singleton dimension (e.g., shape (1, 9, 9)), it is squeezed.
    
    Args:
        trajectory (list): List of boards (numpy arrays or lists).
    
    Returns:
        list: A validated (and possibly squeezed) trajectory.
    
    Raises:
        ValueError: If a board does not have the expected shape (9, 9) after squeezing.
    """
    validated = []
    for idx, board in enumerate(trajectory):
        board = np.array(board)
        if board.ndim == 3 and board.shape[0] == 1:
            board = board.squeeze(0)
        if board.shape != (9, 9):
            raise ValueError(f"Board at index {idx} has invalid shape {board.shape}. Expected (9, 9).")
        validated.append(board)
    return validated

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Move all tensors in a batch dictionary to the specified device.
    
    Args:
        batch (dict): Dictionary containing tensors.
        device (torch.device): Target device.
        
    Returns:
        dict: Dictionary with all tensors moved to the device.
    """
    return {k: v.to(device) for k, v in batch.items()}

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                    ignore_mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute accuracy between predictions and targets, optionally ignoring certain positions.
    
    Args:
        predictions (torch.Tensor): Predicted values of shape (B, ...).
        targets (torch.Tensor): Target values of the same shape as predictions.
        ignore_mask (torch.Tensor, optional): Binary mask where 1 indicates positions to ignore.
        
    Returns:
        float: Accuracy value between 0 and 1.
    """
    if ignore_mask is not None:
        # Only consider positions where ignore_mask is 0
        active_positions = (ignore_mask == 0)
        if not active_positions.any():
            return 1.0  # If all positions are ignored, return perfect accuracy
        
        correct = (predictions == targets) & active_positions
        return correct.sum().item() / active_positions.sum().item()
    else:
        correct = (predictions == targets)
        return correct.float().mean().item()