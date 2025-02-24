"""
Diffusion-specific utility functions for the Discrete Diffusion project.

This module provides utility functions related to diffusion processes.
"""

import torch
import torch.nn as nn
import math
from typing import Callable, Optional, Tuple

def enforce_clue_logits(logits: torch.Tensor, board: torch.Tensor, clue_mask: torch.Tensor) -> torch.Tensor:
    """
    Overrides logits for clue cells so that they are forced to output the known token.

    Args:
        logits (torch.Tensor): Tensor of shape (B, 9, 9, num_tokens) containing raw logits.
        board (torch.Tensor): The original board tensor of shape (B, 9, 9) with tokens in {0, ..., 9}.
        clue_mask (torch.Tensor): Binary mask of shape (B, 9, 9) where 1 indicates a clue cell.

    Returns:
        torch.Tensor: The modified logits with clue cells fixed.
    """
    fixed_logits = torch.full_like(logits, -1e10)
    fixed_logits.scatter_(3, board.unsqueeze(-1), 1e10)
    clue_mask_expanded = clue_mask.unsqueeze(-1)
    return clue_mask_expanded * fixed_logits + (1 - clue_mask_expanded) * logits

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Create a noise schedule based on a cosine curve.
    
    Args:
        timesteps (int): Number of timesteps in the diffusion process.
        s (float): Small offset to prevent the noise level from going to zero.
        
    Returns:
        torch.Tensor: Tensor of shape (timesteps,) containing the noise schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Create a linear schedule for the noise variance (beta).
    
    Args:
        timesteps (int): Number of timesteps in the diffusion process.
        beta_start (float): Starting value for beta.
        beta_end (float): Ending value for beta.
        
    Returns:
        torch.Tensor: Tensor of shape (timesteps,) containing the beta schedule.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract specific timestep values from a schedule tensor.
    
    Args:
        vals (torch.Tensor): Tensor containing the schedule values.
        t (torch.Tensor): Tensor containing timestep indices of shape (B, 1) or (B,).
        x_shape (Tuple[int, ...]): Shape of the target tensor to be noised.
        
    Returns:
        torch.Tensor: Tensor of schedule values matching the batch dimensions of x_shape.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()).reshape(batch_size, *([1] * (len(x_shape) - 1))).to(t.device)
    return out


class LearnedNoiseSchedule(nn.Module):
    """
    A simple MLP-based noise schedule that takes in a normalized time step
    and outputs a noise probability between 0 and 1.
    """
    def __init__(self, hidden_dim: int = 16):
        """
        Initialize the learned noise schedule.
        
        Args:
            hidden_dim (int): Dimensionality of hidden layers.
        """
        super(LearnedNoiseSchedule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # output in (0,1)
        )
    
    def forward(self, t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
        """
        Compute the noise probability for the given timestep.
        
        Args:
            t (torch.Tensor): Timestep tensor of shape (B, 1).
            num_timesteps (int): Total number of timesteps.
            
        Returns:
            torch.Tensor: Noise probability of shape (B, 1).
        """
        # Normalize t to be in [0, 1]
        t_norm = t / num_timesteps
        noise_prob = self.net(t_norm)
        return noise_prob