"""
Base model definitions for the Discrete Diffusion project.

This module provides abstract base classes and interfaces for model implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from utils.diffusion_utils import enforce_clue_logits

class BaseDenoiser(nn.Module):
    """
    Abstract base class for all denoiser models in the Discrete Diffusion project.
    
    Defines the common interface and shared functionality across all denoiser variants.
    """
    
    def __init__(self, num_tokens: int = 10, hidden_dim: int = 128):
        """
        Initialize the base denoiser.
        
        Args:
            num_tokens (int): Total number of tokens (0 for noise; 1-9 for Sudoku digits).
            hidden_dim (int): Dimensionality of token and time embeddings.
        """
        super(BaseDenoiser, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                clue_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the denoiser.
        
        Args:
            x (torch.Tensor): Noisy board of shape (batch_size, 9, 9) with token values.
            t (torch.Tensor): Timestep tensor of shape (batch_size, 1).
            clue_mask (torch.Tensor): Binary mask of shape (batch_size, 9, 9) where 1 indicates a clue.
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, 9, 9, num_tokens).
        """
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def configure_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Configure the optimizer based on the provided configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary with optimizer settings.
            
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=config.get('learning_rate', 1e-4))
