"""
Core diffusion processes for the Discrete Diffusion project.

This module implements the forward and reverse diffusion processes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Union, Tuple

def forward_process(
    x0: torch.Tensor, 
    t: torch.Tensor, 
    num_timesteps: int, 
    clue_mask: torch.Tensor, 
    noise_schedule_fn: Optional[Callable] = None
) -> torch.Tensor:
    """
    Applies the discrete forward noising process with a noise schedule.
    
    Args:
        x0 (torch.Tensor): Solved board of shape (B, 9, 9) with tokens in {1,...,9}.
        t (torch.Tensor): Timestep tensor of shape (B, 1) with values in [0, num_timesteps].
        num_timesteps (int): Total diffusion timesteps.
        clue_mask (torch.Tensor): Clue mask of shape (B, 9, 9).
        noise_schedule_fn (callable, optional): A function that takes (t, num_timesteps) and returns noise probability.
                                            If None, the default cosine-based schedule is used.
    
    Returns:
        torch.Tensor: Noisy board of shape (B, 9, 9) with tokens in {0,...,9}.
    """
    batch_size, height, width = x0.shape
    
    # Compute noise probability
    if noise_schedule_fn is None:
        time_fraction = t / num_timesteps
        noise_prob = 1 - torch.cos(time_fraction * (math.pi / 2))
    else:
        noise_prob = noise_schedule_fn(t, num_timesteps)
    
    # View for broadcasting
    noise_prob = noise_prob.view(-1, 1, 1)
    
    # Generate noise mask with one tensor operation
    noise_mask = (torch.rand_like(x0, dtype=torch.float) < noise_prob) & (clue_mask == 0)
    
    # Create output tensor
    x_t = x0.clone()
    
    # Count noise positions
    noise_positions = noise_mask.sum().item()
    
    # Only generate random tokens if there are positions to apply noise to
    if noise_positions > 0:
        # Generate random tokens for all noise positions at once
        random_tokens = torch.randint(0, 10, (noise_positions,), device=x0.device)
        
        # Use masked_scatter for efficient update
        x_t = x_t.masked_scatter_(noise_mask, random_tokens)
    
    return x_t

def reverse_diffusion_inference(
    model: nn.Module, 
    initial_board: torch.Tensor, 
    clue_mask: torch.Tensor, 
    num_timesteps: int, 
    device: torch.device
) -> List[torch.Tensor]:
    """
    Performs reverse diffusion inference to solve a Sudoku puzzle.
    Starting from an initial board (with noise in non-clue positions),
    iteratively denoises the board by sampling from the model at decreasing timesteps.
    
    Args:
        model (nn.Module): Trained denoiser model.
        initial_board (torch.Tensor): Board of shape (1, 9, 9) with clues filled and noise in others.
        clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
        num_timesteps (int): Total diffusion timesteps.
        device (torch.device): Device for inference.
    
    Returns:
        List[torch.Tensor]: List of board states showing the evolution.
    """
    model.eval()
    with torch.no_grad():
        # Initialize 
        current_board = initial_board.clone()  # (1,9,9)
        trajectory = [current_board.cpu().numpy()]
        
        # Pre-compute timestep tensors for all steps
        timesteps = torch.tensor(
            [[t] for t in reversed(range(1, num_timesteps + 1))], 
            device=device, dtype=torch.float
        )
        
        # For efficiency, create a mask for non-clue cells
        non_clue_mask = (clue_mask == 0)
        
        # Process all timesteps
        for t_tensor in timesteps:
            # Get model predictions
            logits = model(current_board, t_tensor, clue_mask)  # (1,9,9,num_tokens)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # For non-clue cells only, sample from the predicted distribution
            # This avoids unnecessary sampling for clue cells
            probs_flat = probs.view(-1, probs.size(-1))[non_clue_mask.view(-1)]
            sampled_flat = torch.multinomial(probs_flat, 1).view(-1)
            
            # Create new board with updated non-clue cells
            new_board = current_board.clone()
            new_board[non_clue_mask] = sampled_flat
            
            # Update current board
            current_board = new_board
            
            # Add to trajectory
            trajectory.append(current_board.cpu().numpy())
        
        return trajectory