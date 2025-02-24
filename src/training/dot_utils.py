"""
Diffusion of Thought (DoT) utilities for the Discrete Diffusion project.

This module provides utilities for DoT-based diffusion training and inference.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

def simulate_reverse_diffusion_dot(
    model: nn.Module,
    initial_board: torch.Tensor,
    clue_mask: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    num_trajectories: int = 5,
) -> torch.Tensor:
    """
    Simulates the reverse diffusion process in a differentiable manner for multiple trajectories.
    Instead of committing to hard decisions via sampling, the process updates the board using the
    model's softmax outputs. Clue cells remain fixed as one-hot vectors, while non-clue cells are
    updated with soft predictions.

    Args:
        model (nn.Module): The trained denoiser model.
        initial_board (torch.Tensor): Noisy board of shape (B, 9, 9) with clue cells fixed.
        clue_mask (torch.Tensor): Binary mask of shape (B, 9, 9) where 1 indicates a clue.
        num_timesteps (int): Total number of reverse diffusion timesteps.
        device (torch.device): Device for computation.
        num_trajectories (int, optional): Number of trajectories to simulate. Defaults to 5.

    Returns:
        torch.Tensor: A tensor of shape (B, num_trajectories, 9, 9, num_tokens) containing the final
                      soft predictions (probability distributions) for each trajectory.
    """
    B = initial_board.size(0)
    num_tokens = model.num_tokens
    
    # Initialize aggregators for online computation
    trajectory_sum = torch.zeros((B, 9, 9, num_tokens), device=device)
    kl_divergence_sum = torch.tensor(0.0, device=device)
    
    for traj_idx in range(num_trajectories):
        # Process one trajectory
        board_soft = F.one_hot(initial_board.long(), num_classes=num_tokens).float()
        
        # Run reverse diffusion for this trajectory
        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor = torch.full((B, 1), t, device=device, dtype=torch.float)
            board_indices = board_soft.argmax(dim=-1)
            logits = model(board_indices, t_tensor, clue_mask)
            probs = torch.softmax(logits, dim=-1)
            
            board_soft = torch.where(
                clue_mask.unsqueeze(-1).bool(),
                board_soft,
                probs
            )
        
        # Add to running sum for online mean calculation
        trajectory_sum += board_soft
        
        # Store this trajectory's result if needed for inference
        if traj_idx == 0:
            first_trajectory = board_soft.detach().clone()
    
    # Compute final aggregated prediction
    final_prediction = trajectory_sum / num_trajectories
    
    # For inference, return just the final prediction
    # For training, we would compute consistency loss here
    return final_prediction


def compute_trajectory_consistency_loss(
    trajectories: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the trajectory consistency loss for the DoT approach. For each cell in each sample,
    the loss is defined as the average KL divergence between each trajectory's predicted soft
    distribution and the mean prediction across all trajectories. This encourages the multiple
    trajectories to converge to similar predictions.

    Args:
        trajectories (torch.Tensor): Tensor of shape (B, T, 9, 9, num_tokens), where T is the number of
                                    trajectories.

    Returns:
        torch.Tensor: A scalar loss value representing the trajectory consistency loss.
    """
    # Compute the mean soft prediction over trajectories for each cell: shape (B, 9, 9, num_tokens)
    """
    Compute trajectory consistency loss without storing all trajectories.
    
    Uses a two-pass algorithm:
    1. First pass: compute the mean prediction across all trajectories
    2. Second pass: compute KL divergence between each trajectory and the mean
    """
    B = initial_board.size(0)
    num_tokens = model.num_tokens
    
    # First pass: compute mean prediction
    mean_prediction = torch.zeros((B, 9, 9, num_tokens), device=device)
    
    for traj_idx in range(num_trajectories):
        # Simulate one trajectory
        board_soft = F.one_hot(initial_board.long(), num_classes=num_tokens).float()
        
        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor = torch.full((B, 1), t, device=device, dtype=torch.float)
            board_indices = board_soft.argmax(dim=-1)
            logits = model(board_indices, t_tensor, clue_mask)
            probs = torch.softmax(logits, dim=-1)
            
            board_soft = torch.where(
                clue_mask.unsqueeze(-1).bool(),
                board_soft,
                probs
            )
        
        # Add to mean prediction
        mean_prediction += board_soft
    
    # Finalize mean prediction
    mean_prediction = mean_prediction / num_trajectories
    mean_prediction = mean_prediction + 1e-8  # Numerical stability
    mean_prediction = mean_prediction / mean_prediction.sum(dim=-1, keepdim=True)
    
    # Second pass: compute KL divergence
    total_kl_div = torch.tensor(0.0, device=device)
    
    for traj_idx in range(num_trajectories):
        # Simulate one trajectory again
        board_soft = F.one_hot(initial_board.long(), num_classes=num_tokens).float()
        
        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor = torch.full((B, 1), t, device=device, dtype=torch.float)
            board_indices = board_soft.argmax(dim=-1)
            logits = model(board_indices, t_tensor, clue_mask)
            probs = torch.softmax(logits, dim=-1)
            
            board_soft = torch.where(
                clue_mask.unsqueeze(-1).bool(),
                board_soft,
                probs
            )
        
        # Compute KL divergence with the mean
        traj_pred = board_soft + 1e-8
        traj_pred = traj_pred / traj_pred.sum(dim=-1, keepdim=True)
        kl_div = torch.sum(traj_pred * (torch.log(traj_pred) - torch.log(mean_prediction)), dim=-1)
        total_kl_div += kl_div.mean()
    
    consistency_loss = total_kl_div / num_trajectories
    return consistency_loss, mean_prediction