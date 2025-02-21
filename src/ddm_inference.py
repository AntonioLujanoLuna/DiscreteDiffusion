"""
ddm_inference.py

This module implements the Diffusion Decision Model (DDM)-inspired reverse inference
for your discrete diffusion Sudoku solver. The approach uses evidence accumulation
per cell during reverse diffusion to decide when to fix a cell's value based on
a threshold.

Functions:
    - reverse_diffusion_inference_ddm: Performs reverse diffusion with evidence accumulation.
    - run_inference_ddm: Prepares the initial board and runs the DDM-based reverse diffusion.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the forward process from your diffusion module.
from diffusion import forward_process


def reverse_diffusion_inference_ddm(
    model: nn.Module,
    initial_board: torch.Tensor,
    clue_mask: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    threshold: float = 0.9,
) -> List[np.ndarray]:
    """
    Performs reverse diffusion inference with evidence accumulation and a soft gating
    mechanism to decide when to commit a cell's value. Instead of a hard threshold,
    a sigmoid function (with temperature tau) is used to gradually commit decisions.
    """
    model.eval()
    with torch.no_grad():
        current_board: torch.Tensor = initial_board.clone()
        num_tokens: int = 10
        accumulator: torch.Tensor = torch.zeros((1, 9, 9, num_tokens), device=device)
        # Initially, the final decision is simply the current board.
        final_decision: torch.Tensor = current_board.clone()
        decision_made: torch.Tensor = (clue_mask == 1)
        trajectory: List[np.ndarray] = [current_board.cpu().numpy()]

        # Temperature for soft gating. Lower values make the gating function steeper.
        tau: float = 0.05

        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor: torch.Tensor = torch.tensor([[t]], device=device, dtype=torch.float)
            logits: torch.Tensor = model(current_board, t_tensor, clue_mask)  # (1, 9, 9, num_tokens)
            probs: torch.Tensor = torch.softmax(logits, dim=-1)             # (1, 9, 9, num_tokens)

            # Update the evidence accumulator for non-clue and not-yet-decided cells.
            update_mask: torch.Tensor = (clue_mask == 0) & (~decision_made)
            accumulator[update_mask] += probs[update_mask]

            # Compute maximum accumulated evidence and corresponding indices.
            max_values, max_indices = accumulator.max(dim=-1)  # (1, 9, 9)

            # Compute a continuous commit gate (values between 0 and 1) using a sigmoid.
            commit_gate: torch.Tensor = torch.sigmoid((max_values - threshold) / tau)

            # Define a "hard commit" if the gate is sufficiently high.
            hard_commit_mask: torch.Tensor = (commit_gate > 0.8) & update_mask

            # For cells with a high commit gate, fix the decision.
            final_decision[hard_commit_mask] = max_indices[hard_commit_mask]
            decision_made[hard_commit_mask] = True

            # For remaining non-decided cells, continue with fallback sampling.
            fallback_mask: torch.Tensor = (clue_mask == 0) & (~decision_made)
            if fallback_mask.any():
                fallback_probs: torch.Tensor = probs[fallback_mask]  # (num_fallback, num_tokens)
                sampled_tokens: torch.Tensor = torch.multinomial(fallback_probs, num_samples=1).squeeze(-1)
                final_decision[fallback_mask] = sampled_tokens

            # Update current board for the next iteration.
            current_board = final_decision.clone()
            trajectory.append(current_board.cpu().numpy())

        return trajectory



def run_inference_ddm(
    model: nn.Module,
    solved_board: torch.Tensor,
    clue_mask: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    threshold: float = 0.9,
) -> List[np.ndarray]:
    """
    Prepares the initial board from a solved board and its clue mask, applies the forward
    noising process, and then runs the DDM-based reverse diffusion inference to obtain a
    trajectory of board states.

    Args:
        model (nn.Module): The trained denoiser model.
        solved_board (torch.Tensor): Solved board of shape (1, 9, 9).
        clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
        num_timesteps (int): Total number of diffusion timesteps.
        device (torch.device): Device for inference.
        threshold (float, optional): Evidence threshold for DDM-based decision-making.
                                     Defaults to 0.9.

    Returns:
        List[np.ndarray]: A list of board states (as numpy arrays) representing the
                          reverse diffusion trajectory.
    """
    # Create the initial board: clue cells remain; non-clue cells are set to the noise token (0).
    initial_board: torch.Tensor = solved_board.clone()
    initial_board[clue_mask == 0] = 0

    # Apply the forward process to add extra noise.
    t_full: torch.Tensor = torch.full((1, 1), num_timesteps, device=device, dtype=torch.float)
    initial_board = forward_process(initial_board, t_full, num_timesteps, clue_mask)

    # Run the DDM-based reverse diffusion inference.
    trajectory: List[np.ndarray] = reverse_diffusion_inference_ddm(
        model, initial_board, clue_mask, num_timesteps, device, threshold
    )
    return trajectory
