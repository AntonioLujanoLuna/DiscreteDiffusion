"""
Model components for the Discrete Diffusion project.

This module provides reusable components that can be used across different model architectures.
"""

import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Computes sinusoidal time embeddings similar to the positional encodings used in Transformers.
    
    Given a time tensor t of shape (B, 1), produces an embedding of shape (B, hidden_dim).
    """
    def __init__(self, hidden_dim: int):
        """
        Initialize the sinusoidal time embedding module.
        
        Args:
            hidden_dim (int): Dimensionality of the time embedding.
        """
        super(SinusoidalTimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the sinusoidal time embedding.
        
        Args:
            t (torch.Tensor): Tensor of shape (B, 1) representing time steps.
            
        Returns:
            torch.Tensor: Sinusoidal embeddings of shape (B, hidden_dim).
        """
        half_dim = self.hidden_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb_scale)
        emb = t * freq  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=-1)
        return emb

def create_2d_sinusoidal_embeddings(height: int, width: int, dim: int) -> torch.Tensor:
    """
    Creates fixed 2D sinusoidal positional embeddings for a grid.
    
    Args:
        height (int): Height of the grid (number of rows).
        width (int): Width of the grid (number of columns).
        dim (int): Embedding dimensionality.
    
    Returns:
        torch.Tensor: Positional embeddings of shape (height, width, dim).
    """
    pe = torch.zeros(height, width, dim)
    for i in range(height):
        for j in range(width):
            for k in range(0, dim, 2):
                div_term = math.exp((k * -math.log(10000.0)) / dim)
                pe[i, j, k] = math.sin(i * div_term) + math.sin(j * div_term)
                if k + 1 < dim:
                    pe[i, j, k + 1] = math.cos(i * div_term) + math.cos(j * div_term)
    return pe