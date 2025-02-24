"""
Diffusion Decision Model (DDM) utilities for the Discrete Diffusion project.

This module provides utilities for DDM-based diffusion training and inference.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def simulate_reverse_diffusion_ddm(
    model: nn.Module,
    initial_board: torch.Tensor,
    clue_mask: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    threshold: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates the reverse diffusion process using evidence accumulation for non-clue cells.
    For each reverse timestep, the model computes a probability distribution over tokens.
    These probabilities (for non-clue cells) are added to an accumulator. In parallel,
    a hard decision is made for each cell by taking the argmax of the current probabilities.
    
    Note: The hard decision update (via argmax) is non-differentiable; however, the accumulator
    update uses soft probabilities, so gradients can flow through the evidence loss.

    Args:
        model (nn.Module): The trained denoiser model.
        initial_board (torch.Tensor): Noisy board of shape (B, 9, 9) with clue cells fixed.
        clue_mask (torch.Tensor): Binary mask of shape (B, 9, 9) where 1 indicates a clue.
        num_timesteps (int): Total number of reverse diffusion timesteps.
        device (torch.device): Device for computation.
        threshold (float, optional): (Not used in the training simulation for hard gating)
            Provided for compatibility with inference; defaults to 0.9.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - final_board: Tensor of shape (B, 9, 9) with final predicted token indices.
            - accumulator: Tensor of shape (B, 9, 9, num_tokens) containing the accumulated evidence.
    """
    # Get batch size and number of tokens.
    B = initial_board.size(0)
    num_tokens: int = model.num_tokens

    # Initialize the board and accumulator.
    current_board: torch.Tensor = initial_board.clone()  # (B, 9, 9)
    accumulator: torch.Tensor = torch.zeros(B, 9, 9, num_tokens, device=device)

    # Loop over reverse diffusion timesteps.
    for t in reversed(range(1, num_timesteps + 1)):
        # Create a timestep tensor of shape (B, 1)
        t_tensor: torch.Tensor = torch.full((B, 1), t, device=device, dtype=torch.float)
        # Compute model logits for current board state.
        logits: torch.Tensor = model(current_board, t_tensor, clue_mask)  # (B, 9, 9, num_tokens)
        # Convert logits to probabilities.
        probs: torch.Tensor = torch.softmax(logits, dim=-1)  # (B, 9, 9, num_tokens)

        # Update the accumulator for non-clue cells.
        update_mask: torch.Tensor = (clue_mask == 0)  # (B, 9, 9)
        accumulator[update_mask] = accumulator[update_mask] + probs[update_mask]

        # For non-clue cells, update current_board with a hard decision (argmax).
        # This step is non-differentiable but serves as a surrogate for the next input.
        predicted_tokens: torch.Tensor = probs.argmax(dim=-1)  # (B, 9, 9)
        # Keep clue cells unchanged.
        current_board = torch.where(clue_mask.bool(), current_board, predicted_tokens)

    return current_board, accumulator


def compute_evidence_loss(
    accumulator: torch.Tensor,
    solved_board: torch.Tensor,
    clue_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the evidence consistency loss. For non-clue cells, the accumulated evidence
    is normalized to form a probability distribution. The loss is computed as the negative log-likelihood
    (cross entropy) of the ground truth token under this distribution.

    Args:
        accumulator (torch.Tensor): Accumulated evidence of shape (B, 9, 9, num_tokens).
        solved_board (torch.Tensor): Ground truth solved board of shape (B, 9, 9) with token indices.
        clue_mask (torch.Tensor): Binary mask of shape (B, 9, 9) where 1 indicates a clue.

    Returns:
        torch.Tensor: A scalar loss value representing the evidence consistency loss.
    """
    # Normalize the accumulator along the token dimension.
    accumulator_sum: torch.Tensor = accumulator.sum(dim=-1, keepdim=True) + 1e-8
    accumulator_norm: torch.Tensor = accumulator / accumulator_sum  # (B, 9, 9, num_tokens)
    # Compute log probabilities.
    log_probs: torch.Tensor = torch.log(accumulator_norm + 1e-8)  # (B, 9, 9, num_tokens)
    
    # Flatten tensors and select only non-clue cells.
    B, H, W, num_tokens = accumulator.shape
    mask: torch.Tensor = (clue_mask == 0)  # (B, 9, 9)
    
    # Reshape for loss computation.
    log_probs_flat: torch.Tensor = log_probs.view(B * H * W, num_tokens)
    solved_board_flat: torch.Tensor = solved_board.view(B * H * W)
    mask_flat: torch.Tensor = mask.view(B * H * W)
    
    # Select only non-clue entries.
    log_probs_selected: torch.Tensor = log_probs_flat[mask_flat]
    targets: torch.Tensor = solved_board_flat[mask_flat]
    
    # Compute negative log likelihood loss.
    evidence_loss: torch.Tensor = F.nll_loss(log_probs_selected, targets)
    return evidence_loss