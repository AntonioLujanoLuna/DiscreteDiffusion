"""
Loss functions for the Discrete Diffusion project.

This module provides loss strategies for different diffusion training approaches.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

from diffusion import forward_process

class LossStrategy:
    """
    Base class for loss computation strategies.
    
    Different diffusion training approaches (Standard, DDM, DoT) require
    different loss computations. This class provides an interface for
    implementing these different strategies.
    """
    
    def compute_losses(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor], 
        t: torch.Tensor, 
        num_timesteps: int, 
        device: torch.device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for a batch of data.
        
        Args:
            model (torch.nn.Module): The model
            batch (Dict[str, torch.Tensor]): Batch of data
            t (torch.Tensor): Timestep tensor
            num_timesteps (int): Total diffusion timesteps
            device (torch.device): Device for computation
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        raise NotImplementedError("Subclasses must implement compute_losses")
    
    def get_metric_keys(self) -> List[str]:
        """
        Get the keys for metrics to track.
        
        Returns:
            List[str]: List of metric keys
        """
        raise NotImplementedError("Subclasses must implement get_metric_keys")


class StandardLossStrategy(LossStrategy):
    """
    Standard diffusion loss computation strategy.
    
    Computes the standard cross-entropy loss for predicting the clean token
    from a noisy board, along with an optional constraint loss for enforcing
    Sudoku rules.
    """
    
    def compute_losses(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor], 
        t: torch.Tensor, 
        num_timesteps: int, 
        device: torch.device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute standard diffusion losses.
        
        Args:
            model (torch.nn.Module): The denoiser model
            batch (Dict[str, torch.Tensor]): Batch containing "solved_board", "puzzle_board", and "clue_mask"
            t (torch.Tensor): Timestep tensor of shape (B, 1)
            num_timesteps (int): Total diffusion timesteps
            device (torch.device): Computation device
            **kwargs: Additional arguments, including:
                - noise_schedule_fn (Optional[Any]): Function for noise scheduling
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss components
        """
        solved_board = batch["solved_board"]
        puzzle_board = batch["puzzle_board"]
        clue_mask = batch["clue_mask"]
        
        # Get the noise schedule function from kwargs
        noise_schedule_fn = kwargs.get("noise_schedule_fn")
        
        # Apply forward process to get noisy board
        x_t = forward_process(puzzle_board, t, num_timesteps, clue_mask, noise_schedule_fn=noise_schedule_fn)
        
        # Get model predictions
        logits = model(x_t, t, clue_mask)
        
        # Compute cross-entropy loss
        batch_size = solved_board.size(0)
        logits_flat = logits.view(batch_size, -1, model.num_tokens)
        solved_board_flat = solved_board.view(batch_size, -1)
        clue_mask_flat = clue_mask.view(batch_size, -1)
        
        ce_loss = F.cross_entropy(
            logits_flat.view(-1, model.num_tokens),
            solved_board_flat.view(-1),
            reduction="none"
        )
        ce_loss = (ce_loss * (1 - clue_mask_flat.view(-1))).mean()
        
        # Compute constraint loss
        constraint_loss = compute_constraint_loss(logits)
        
        return {
            "total_loss": ce_loss,
            "ce_loss": ce_loss,
            "constraint_loss": constraint_loss
        }
    
    def get_metric_keys(self) -> List[str]:
        """
        Get the keys for metrics to track.
        
        Returns:
            List[str]: List of metric keys
        """
        return ["total_loss", "ce_loss", "constraint_loss"]


class DDMLossStrategy(LossStrategy):
    """
    Diffusion Decision Model (DDM) loss computation strategy.
    
    In addition to the standard cross-entropy and constraint losses,
    this strategy includes an evidence accumulation loss that encourages
    the model to make consistent decisions over time.
    """
    
    def compute_losses(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor], 
        t: torch.Tensor, 
        num_timesteps: int, 
        device: torch.device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for DDM training.
        
        Args:
            model (torch.nn.Module): The denoiser model
            batch (Dict[str, torch.Tensor]): Batch containing "solved_board", "puzzle_board", and "clue_mask"
            t (torch.Tensor): Timestep tensor of shape (B, 1)
            num_timesteps (int): Total diffusion timesteps
            device (torch.device): Computation device
            **kwargs: Additional arguments, including:
                - noise_schedule_fn (Optional[Any]): Function for noise scheduling
                - threshold (float): Evidence threshold for reverse diffusion
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss components
        """
        solved_board = batch["solved_board"]
        puzzle_board = batch["puzzle_board"]
        clue_mask = batch["clue_mask"]
        
        # Get additional parameters from kwargs
        noise_schedule_fn = kwargs.get("noise_schedule_fn")
        threshold = kwargs.get("threshold", 0.9)
        
        # Apply forward process to get noisy board
        x_t = forward_process(puzzle_board, t, num_timesteps, clue_mask, noise_schedule_fn=noise_schedule_fn)
        
        # Get model predictions
        logits = model(x_t, t, clue_mask)
        
        # Compute cross-entropy loss
        batch_size = solved_board.size(0)
        logits_flat = logits.view(batch_size, -1, model.num_tokens)
        solved_board_flat = solved_board.view(batch_size, -1)
        clue_mask_flat = clue_mask.view(batch_size, -1)
        
        ce_loss = F.cross_entropy(
            logits_flat.view(-1, model.num_tokens),
            solved_board_flat.view(-1),
            reduction="none"
        )
        ce_loss = (ce_loss * (1 - clue_mask_flat.view(-1))).mean()
        
        # Compute constraint loss
        constraint_loss = compute_constraint_loss(logits)
        
        # Simulate reverse diffusion to get the evidence accumulator
        from ddm_utils import simulate_reverse_diffusion_ddm, compute_evidence_loss
        _, accumulator = simulate_reverse_diffusion_ddm(
            model, x_t, clue_mask, num_timesteps, device, threshold=threshold
        )
        evidence_loss = compute_evidence_loss(accumulator, solved_board, clue_mask)
        
        return {
            "total_loss": ce_loss,
            "ce_loss": ce_loss,
            "constraint_loss": constraint_loss,
            "evidence_loss": evidence_loss
        }
    
    def get_metric_keys(self) -> List[str]:
        """
        Get the keys for metrics to track.
        
        Returns:
            List[str]: List of metric keys
        """
        return ["total_loss", "ce_loss", "constraint_loss", "evidence_loss"]


class DoTLossStrategy(LossStrategy):
    """
    Diffusion of Thought (DoT) loss computation strategy.
    
    In addition to the standard cross-entropy and constraint losses,
    this strategy includes a trajectory consistency loss that encourages
    multiple generation trajectories to converge to similar solutions.
    """
    
    def compute_losses(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor], 
        t: torch.Tensor, 
        num_timesteps: int, 
        device: torch.device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for DoT training.
        
        Args:
            model (torch.nn.Module): The denoiser model
            batch (Dict[str, torch.Tensor]): Batch containing "solved_board", "puzzle_board", and "clue_mask"
            t (torch.Tensor): Timestep tensor of shape (B, 1)
            num_timesteps (int): Total diffusion timesteps
            device (torch.device): Computation device
            **kwargs: Additional arguments, including:
                - noise_schedule_fn (Optional[Any]): Function for noise scheduling
                - num_trajectories (int): Number of trajectories to simulate
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss components
        """
        solved_board = batch["solved_board"]
        puzzle_board = batch["puzzle_board"]
        clue_mask = batch["clue_mask"]
        
        # Get additional parameters from kwargs
        noise_schedule_fn = kwargs.get("noise_schedule_fn")
        num_trajectories = kwargs.get("num_trajectories", 5)
        
        # Apply forward process to get noisy board
        x_t = forward_process(puzzle_board, t, num_timesteps, clue_mask, noise_schedule_fn=noise_schedule_fn)
        
        # Get model predictions
        logits = model(x_t, t, clue_mask)
        
        # Compute cross-entropy loss
        batch_size = solved_board.size(0)
        logits_flat = logits.view(batch_size, -1, model.num_tokens)
        solved_board_flat = solved_board.view(batch_size, -1)
        clue_mask_flat = clue_mask.view(batch_size, -1)
        
        ce_loss = F.cross_entropy(
            logits_flat.view(-1, model.num_tokens),
            solved_board_flat.view(-1),
            reduction="none"
        )
        ce_loss = (ce_loss * (1 - clue_mask_flat.view(-1))).mean()
        
        # Compute constraint loss
        constraint_loss = compute_constraint_loss(logits)
        
        # Simulate multiple reverse diffusion trajectories
        from dot_utils import simulate_reverse_diffusion_dot, compute_trajectory_consistency_loss
        trajectories = simulate_reverse_diffusion_dot(
            model, x_t, clue_mask, num_timesteps, device, num_trajectories
        )
        trajectory_loss = compute_trajectory_consistency_loss(trajectories)
        
        return {
            "total_loss": ce_loss,
            "ce_loss": ce_loss,
            "constraint_loss": constraint_loss,
            "trajectory_loss": trajectory_loss
        }
    
    def get_metric_keys(self) -> List[str]:
        """
        Get the keys for metrics to track.
        
        Returns:
            List[str]: List of metric keys
        """
        return ["total_loss", "ce_loss", "constraint_loss", "trajectory_loss"]


def compute_constraint_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Highly optimized constraint loss computation that handles all constraints
    in a single operation for maximum efficiency.
    
    Args:
        logits (torch.FloatTensor): Logits of shape (B, 9, 9, num_tokens).
    
    Returns:
        torch.FloatTensor: Scalar constraint loss.
    """
    # Exclude the noise token (index 0) and compute softmax over tokens 1-9
    p = torch.softmax(logits, dim=-1)[..., 1:10]  # shape: (B, 9, 9, 9)
    batch_size = p.size(0)
    
    # --- Create masks for all constraints at once ---
    # We'll handle all constraints with tensor operations
    
    # Row constraints (already properly arranged)
    # Sum across cells in each row for each digit
    row_sums = p.sum(dim=2)  # (B, 9, 9) - sum along columns
    
    # Column constraints
    # Sum across cells in each column for each digit
    col_sums = p.sum(dim=1)  # (B, 9, 9) - sum along rows
    
    # Block constraints
    # Reshape to group by 3×3 blocks
    reshaped = p.view(batch_size, 3, 3, 3, 3, 9)  # (B, 3, 3, 3, 3, 9)
    # Sum across cells in each block for each digit
    block_sums = reshaped.sum(dim=(2, 4))  # (B, 3, 3, 9)
    block_sums = block_sums.view(batch_size, 9, 9)  # (B, 9, 9)
    
    # Stack all constraints together
    all_sums = torch.cat([
        row_sums.view(batch_size * 9, 9),
        col_sums.view(batch_size * 9, 9),
        block_sums.view(batch_size * 9, 9)
    ], dim=0)  # (B*27, 9)
    
    # Compute squared error from target (should sum to 1)
    # This computes loss for all constraints at once
    loss = ((all_sums - 1.0) ** 2).mean()
    
    return loss

class ConstraintLoss(torch.nn.Module):
    """
    Computes a constraint loss for Sudoku by enforcing that the sum of probabilities 
    for each digit in every row, column, and block is close to 1. The losses for rows,
    columns, and blocks are weighted by learnable parameters.
    """
    def __init__(self, init_lambda_row=1.0, init_lambda_col=1.0, init_lambda_block=1.0):
        super(ConstraintLoss, self).__init__()
        # Learnable weights for each type of constraint
        self.lambda_row = torch.nn.Parameter(torch.tensor(init_lambda_row, dtype=torch.float32))
        self.lambda_col = torch.nn.Parameter(torch.tensor(init_lambda_col, dtype=torch.float32))
        self.lambda_block = torch.nn.Parameter(torch.tensor(init_lambda_block, dtype=torch.float32))

    def forward(self, logits):
        """
        Args:
            logits (torch.FloatTensor): Logits of shape (B, 9, 9, num_tokens).
        Returns:
            torch.FloatTensor: Scalar loss value.
        """
        # Convert logits to probabilities (exclude noise token at index 0)
        p = F.softmax(logits, dim=-1)[..., 1:10]  # shape: (B, 9, 9, 9)
        loss_row = 0.0
        loss_col = 0.0
        loss_block = 0.0

        # Row loss
        for i in range(9):
            row_sum = p[:, i, :, :].sum(dim=1)  # shape: (B, 9)
            loss_row += ((row_sum - 1) ** 2).mean()

        # Column loss
        for j in range(9):
            col_sum = p[:, :, j, :].sum(dim=1)  # shape: (B, 9)
            loss_col += ((col_sum - 1) ** 2).mean()

        # Block loss
        for bi in range(3):
            for bj in range(3):
                block = p[:, bi*3:(bi+1)*3, bj*3:(bj+1)*3, :]  # shape: (B, 3, 3, 9)
                block_sum = block.view(block.size(0), -1, 9).sum(dim=1)  # shape: (B, 9)
                loss_block += ((block_sum - 1) ** 2).mean()

        total_loss = self.lambda_row * loss_row + self.lambda_col * loss_col + self.lambda_block * loss_block
        return total_loss


def compute_uniqueness_loss(logits: torch.Tensor, clue_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes an entropy loss for non-clue cells that encourages the model to be confident in its predictions.
    This acts as a soft surrogate for the uniqueness requirement.
    
    Args:
        logits (torch.Tensor): Logits of shape (B, 9, 9, num_tokens).
        clue_mask (torch.Tensor): Binary mask of shape (B, 9, 9) with 1 for clue cells.
    
    Returns:
        torch.Tensor: Scalar loss value (lower entropy for non-clue cells is better).
    """
    # Compute probabilities over tokens 1-9.
    p = torch.softmax(logits, dim=-1)[..., 1:10]  # shape: (B, 9, 9, 9)
    # Create a mask for non-clue cells (unsqueeze to broadcast over digit dimension).
    non_clue_mask = (clue_mask == 0).unsqueeze(-1)  # shape: (B, 9, 9, 1)
    # For non-clue cells only, compute the entropy.
    p_nonclue = p * non_clue_mask  # zero out clues.
    entropy = - (p_nonclue * torch.log(p_nonclue + 1e-8)).sum(dim=-1)  # shape: (B, 9, 9)
    # Return the mean entropy as the uniqueness loss.
    return entropy.mean()