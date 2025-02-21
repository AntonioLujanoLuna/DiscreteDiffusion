"""
dot_utils.py

Functions:
    - simulate_reverse_diffusion_dot: Simulates multiple reverse diffusion trajectories and
      returns soft predictions for each trajectory.
    - compute_trajectory_consistency_loss: Computes an auxiliary loss that penalizes divergence
      between the trajectories’ predictions for each cell.
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
    model’s softmax outputs. Clue cells remain fixed as one-hot vectors, while non-clue cells are
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
    num_tokens: int = 10

    # To store the final soft predictions from each trajectory.
    trajectories: List[torch.Tensor] = []

    for _ in range(num_trajectories):
        # Initialize the board as soft distributions.
        # For clue cells, create a one-hot representation; for non-clue, use the forward-process board.
        board_soft: torch.Tensor = F.one_hot(initial_board.long(), num_classes=num_tokens).float()  # (B, 9, 9, num_tokens)

        # Run reverse diffusion over timesteps in a soft (differentiable) manner.
        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor: torch.Tensor = torch.full((B, 1), t, device=device, dtype=torch.float)
            # The model expects a discrete board; we use the hard surrogate via argmax,
            # but we update using the soft predictions to retain differentiability.
            board_indices: torch.Tensor = board_soft.argmax(dim=-1)  # (B, 9, 9)
            logits: torch.Tensor = model(board_indices, t_tensor, clue_mask)  # (B, 9, 9, num_tokens)
            probs: torch.Tensor = torch.softmax(logits, dim=-1)  # (B, 9, 9, num_tokens)

            # Update non-clue cells with the new soft predictions while keeping clue cells fixed.
            board_soft = torch.where(
                clue_mask.unsqueeze(-1).bool(),
                board_soft,
                probs
            )
        # Append the final soft board for this trajectory.
        trajectories.append(board_soft)

    # Stack trajectories to form a tensor of shape (B, num_trajectories, 9, 9, num_tokens)
    trajectories_tensor: torch.Tensor = torch.stack(trajectories, dim=1)
    return trajectories_tensor


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
    mean_prediction: torch.Tensor = trajectories.mean(dim=1)
    eps: float = 1e-8  # For numerical stability
    mean_prediction = mean_prediction + eps
    mean_prediction = mean_prediction / mean_prediction.sum(dim=-1, keepdim=True)

    B, T, H, W, num_tokens = trajectories.shape
    loss: torch.Tensor = 0.0
    # Compute the KL divergence for each trajectory relative to the mean.
    for i in range(T):
        traj_pred: torch.Tensor = trajectories[:, i, :, :, :] + eps
        traj_pred = traj_pred / traj_pred.sum(dim=-1, keepdim=True)
        # KL divergence: sum(p * log(p / q)) per cell.
        kl_div: torch.Tensor = torch.sum(traj_pred * (torch.log(traj_pred) - torch.log(mean_prediction)), dim=-1)  # (B, 9, 9)
        loss += kl_div.mean()
    consistency_loss: torch.Tensor = loss / T
    return consistency_loss
