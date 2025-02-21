"""
dot_inference.py

This module implements a Diffusion-of-Thought (DoT) approach for the discrete diffusion
Sudoku solver. In the DoT paradigm, multiple reverse diffusion trajectories are generated
and then aggregated (using a majority vote) to produce a final board. This aggregation
helps ensure global consistency and robust decisions.

Functions:
    - run_inference_dot: Generates multiple inference trajectories and aggregates the final boards.
"""

from typing import List, Tuple
import torch
import numpy as np

# Import the standard reverse diffusion inference function from your existing code.
# This function is assumed to be defined in your "inference.py" file.
from inference import run_inference


def run_inference_dot(
    model: torch.nn.Module,
    solved_board: torch.Tensor,
    clue_mask: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    num_trajectories: int = 5
) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
    """
    Runs reverse diffusion inference multiple times (i.e. generates multiple trajectories)
    and aggregates their final board states to produce a consensus solution.

    Args:
        model (torch.nn.Module): The trained denoiser model.
        solved_board (torch.Tensor): Solved board of shape (1, 9, 9).
        clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
        num_timesteps (int): Total number of diffusion timesteps.
        device (torch.device): Device on which inference is performed.
        num_trajectories (int, optional): Number of trajectories to generate. Defaults to 5.

    Returns:
        Tuple[np.ndarray, List[List[np.ndarray]]]:
            - final_board (np.ndarray): Aggregated final board of shape (9, 9).
            - trajectories (List[List[np.ndarray]]): A list containing each trajectory, where each
              trajectory is itself a list of board states (numpy arrays) over time.
    """
    trajectories: List[List[np.ndarray]] = []
    final_boards: List[torch.Tensor] = []

    # Generate multiple inference trajectories.
    for _ in range(num_trajectories):
        # Call the existing reverse diffusion inference (from inference.py)
        trajectory: List[np.ndarray] = run_inference(model, solved_board, clue_mask, num_timesteps, device)
        trajectories.append(trajectory)
        # The final board from this trajectory is the last element.
        final_boards.append(torch.tensor(trajectory[-1]))

    # Stack all final boards; shape becomes (num_trajectories, 9, 9).
    boards_tensor: torch.Tensor = torch.stack(final_boards, dim=0)
    # Compute the mode (majority vote) along the first dimension.
    # For each cell, this gives the most frequently predicted digit.
    final_board: np.ndarray = torch.mode(boards_tensor, dim=0).values.cpu().numpy()

    return final_board, trajectories
