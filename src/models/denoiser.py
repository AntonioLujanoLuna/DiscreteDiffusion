"""
Denoiser model implementations for the Discrete Diffusion project.

This module provides concrete implementations of different denoiser architectures 
for solving Sudoku puzzles using discrete diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseDenoiser
from .components import SinusoidalTimeEmbedding, create_2d_sinusoidal_embeddings
from utils.diffusion_utils import enforce_clue_logits

class ImprovedSudokuDenoiser(BaseDenoiser):
    """
    Improved Denoiser Network for Discrete Diffusion on Sudoku.
    
    This network incorporates:
      - A token embedding layer.
      - 2D sinusoidal positional embeddings computed for a 9x9 grid.
      - A sinusoidal time embedding module.
      - A transformer encoder for sequence processing.
      - An output projection that is post-processed to enforce the clues.
    """
    def __init__(self, num_tokens=10, hidden_dim=128, num_layers=8, nhead=8, dropout=0.1):
        """
        Args:
            num_tokens (int): Total number of tokens (0 for noise; 1-9 for Sudoku digits).
            hidden_dim (int): Dimensionality of token and time embeddings.
            num_layers (int): Number of transformer layers.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(ImprovedSudokuDenoiser, self).__init__(num_tokens, hidden_dim)
        
        # Token embedding.
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)
        # Register fixed 2D sinusoidal positional embeddings for a 9x9 grid.
        self.register_buffer("pos_embedding", create_2d_sinusoidal_embeddings(9, 9, hidden_dim))
        
        # Time embedding using the sinusoidal method.
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        
        # Transformer encoder (with batch_first=True for easier integration).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layer mapping transformer features to token logits.
        self.output_layer = nn.Linear(hidden_dim, num_tokens)

    def forward(self, x, t, clue_mask):
        """
        Forward pass for the improved denoiser.

        Args:
            x (torch.LongTensor): Noisy board of shape (B, 9, 9) with token values in {0,...,9}.
            t (torch.FloatTensor): Timestep tensor of shape (B, 1).
            clue_mask (torch.FloatTensor): Binary mask of shape (B, 9, 9) where 1 indicates a clue.

        Returns:
            torch.FloatTensor: Logits of shape (B, 9, 9, num_tokens).
        """
        batch_size = x.size(0)
        # Embed the tokens and add fixed 2D positional embeddings.
        x_emb = self.token_embedding(x)           # (B, 9, 9, hidden_dim)
        x_emb = x_emb + self.pos_embedding          # Broadcasting over batch.
        
        # Flatten the grid to a sequence.
        x_emb = x_emb.view(batch_size, -1, self.hidden_dim)  # (B, 81, hidden_dim)
        
        # Ensure t has the right shape
        if t.dim() == 1:
            t = t.unsqueeze(1)  # Ensure t has shape (B, 1)
        # Get time embedding
        t_emb = self.time_embedding(t)              # (B, hidden_dim)
        # Unsqueeze for broadcasting
        t_emb = t_emb.unsqueeze(1)                  # (B, 1, hidden_dim)
        # Explicitly expand to match x_emb
        t_emb = t_emb.expand(-1, x_emb.size(1), -1)  # (B, 81, hidden_dim)
        x_emb = x_emb + t_emb
        
        # Process with the transformer encoder.
        x_emb = self.transformer(x_emb)             # (B, 81, hidden_dim)
        
        # Project to token logits.
        logits = self.output_layer(x_emb)           # (B, 81, num_tokens)
        logits = logits.view(batch_size, 9, 9, self.num_tokens)
        
        # Use the helper function to enforce clue consistency.
        logits = enforce_clue_logits(logits, x, clue_mask)
        return logits

class HybridSudokuDenoiser(BaseDenoiser):
    """
    A hybrid model that uses convolutional layers to capture local spatial structure
    before flattening the grid and applying a transformer encoder. This combination
    helps to encode the inherent 2D relationships of the Sudoku board more effectively.
    """
    def __init__(self, num_tokens=10, hidden_dim=128, num_conv_layers=2, conv_kernel_size=3,
                 num_layers=8, nhead=8, dropout=0.1):
        """
        Args:
            num_tokens (int): Total number of tokens (0 for noise; 1-9 for Sudoku digits).
            hidden_dim (int): Embedding dimension.
            num_conv_layers (int): Number of convolutional layers to use.
            conv_kernel_size (int): Kernel size for the convolutional layers.
            num_layers (int): Number of transformer encoder layers.
            nhead (int): Number of attention heads in the transformer.
            dropout (float): Dropout rate for the transformer.
        """
        super(HybridSudokuDenoiser, self).__init__(num_tokens, hidden_dim)
        
        # Token embedding: convert discrete tokens into continuous vectors.
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)
        
        # Convolutional layers to capture local spatial information.
        conv_layers = []
        in_channels = hidden_dim
        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=conv_kernel_size, 
                                         padding=conv_kernel_size // 2))
            conv_layers.append(nn.BatchNorm2d(hidden_dim))
            conv_layers.append(nn.ReLU())
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Time embedding: inject timestep information.
        self.time_embed = nn.Linear(1, hidden_dim)
        
        # Transformer encoder: processes the flattened sequence.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection: maps transformer outputs to logits over tokens.
        self.output_layer = nn.Linear(hidden_dim, num_tokens)
    
    def forward(self, x, t, clue_mask):
        """
        Forward pass for the Hybrid Sudoku Denoiser.

        Args:
            x (torch.LongTensor): Input board of shape (B, 9, 9) with tokens in {0,...,9}.
            t (torch.FloatTensor): Timestep tensor of shape (B, 1).
            clue_mask (torch.FloatTensor): Binary mask of shape (B, 9, 9) indicating clue cells.

        Returns:
            torch.FloatTensor: Logits of shape (B, 9, 9, num_tokens).
        """
        batch_size = x.size(0)
        
        # Token embedding and rearrangement for convolution.
        x_emb = self.token_embedding(x)           # (B, 9, 9, hidden_dim)
        x_emb = x_emb.permute(0, 3, 1, 2)           # (B, hidden_dim, 9, 9)
        
        # Process with convolutional layers.
        x_conv = self.conv_net(x_emb)               # (B, hidden_dim, 9, 9)
        x_conv = x_conv.permute(0, 2, 3, 1)           # (B, 9, 9, hidden_dim)
        
        # Flatten to a sequence.
        x_seq = x_conv.view(batch_size, -1, self.hidden_dim)  # (B, 81, hidden_dim)
        
        # Add time embedding.
        t_emb = self.time_embed(t)                  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1)                  # (B, 1, hidden_dim)
        x_seq = x_seq + t_emb
        
        # Process with transformer encoder.
        x_trans = self.transformer(x_seq)           # (B, 81, hidden_dim)
        
        # Project to logits.
        logits = self.output_layer(x_trans)         # (B, 81, num_tokens)
        logits = logits.view(batch_size, 9, 9, self.num_tokens)
        
        # Enforce clue consistency using the helper function.
        logits = enforce_clue_logits(logits, x, clue_mask)
        return logits