"""
dot_training.py

This module extends the training of the discrete diffusion Sudoku solver by incorporating a
Diffusion-of-Thought (DoT)-inspired mechanism. In this paradigm, multiple reverse diffusion
trajectories are simulated in a differentiable (soft) manner. A trajectory consistency loss
is computed to encourage agreement among the different trajectories. The overall training
objective is a weighted sum of the standard cross-entropy loss (ignoring clue cells), the
constraint loss (enforcing Sudoku rules), and the trajectory consistency loss.

Functions:
    - train_diffusion_dot: The main training loop that integrates the standard losses with the
      DoT-based trajectory consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffusion import forward_process, compute_constraint_loss
from dot_utils import compute_trajectory_consistency_loss, simulate_reverse_diffusion_dot
from validate import validate_dot_model

def train_diffusion_dot(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_timesteps: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    val_dataloader: torch.utils.data.DataLoader = None,
    num_epochs: int = 100,
    initial_lambda_constraint: float = 1.0,
    lambda_trajectory: float = 0.5,
    num_trajectories: int = 5,
) -> None:
    """
    Trains the discrete diffusion Sudoku solver using a DoT-inspired objective. For each batch,
    multiple reverse diffusion trajectories are simulated in a differentiable manner. The total loss
    is the weighted sum of:
      - Standard cross-entropy loss (ignoring clue cells).
      - Constraint loss enforcing Sudoku rules.
      - Trajectory consistency loss enforcing agreement among the multiple reverse diffusion trajectories.

    Args:
        model (nn.Module): The denoiser model.
        dataloader (torch.utils.data.DataLoader): Provides batches from the SudokuDataset.
        num_timesteps (int): Total diffusion timesteps.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device for training.
        num_epochs (int, optional): Number of epochs to train. Defaults to 100.
        initial_lambda_constraint (float, optional): Initial weight for the constraint loss. Defaults to 1.0.
        lambda_trajectory (float, optional): Weight for the trajectory consistency loss. Defaults to 0.5.
        num_trajectories (int, optional): Number of reverse diffusion trajectories to simulate. Defaults to 5.
    """
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    for epoch in range(num_epochs):
        lambda_constraint = initial_lambda_constraint * (0.95 ** epoch)

        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_constraint_loss = 0.0
        epoch_trajectory_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            solved_board: torch.Tensor = batch["solved_board"].to(device)  # (B, 9, 9)
            clue_mask: torch.Tensor = batch["clue_mask"].to(device)           # (B, 9, 9)
            batch_size: int = solved_board.size(0)

            # Sample random timesteps for each sample.
            t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=device).float()

            # Generate a noisy board using the forward process.
            x_t: torch.Tensor = forward_process(solved_board, t, num_timesteps, clue_mask)

            # Standard forward pass: compute logits from the noisy board.
            logits: torch.Tensor = model(x_t, t, clue_mask)  # (B, 9, 9, num_tokens)
            logits_flat = logits.view(batch_size, -1, logits.size(-1))
            solved_board_flat = solved_board.view(batch_size, -1)
            clue_mask_flat = clue_mask.view(batch_size, -1)

            loss_ce: torch.Tensor = F.cross_entropy(
                logits_flat.view(-1, logits.size(-1)),
                solved_board_flat.view(-1),
                reduction="none",
            )
            loss_ce = (loss_ce * (1 - clue_mask_flat.view(-1))).mean()

            loss_constraint: torch.Tensor = compute_constraint_loss(logits)

            # Simulate multiple reverse diffusion trajectories (soft predictions).
            trajectories: torch.Tensor = simulate_reverse_diffusion_dot(
                model, x_t, clue_mask, num_timesteps, device, num_trajectories
            )  # (B, T, 9, 9, num_tokens)

            # Compute the trajectory consistency loss.
            loss_trajectory: torch.Tensor = compute_trajectory_consistency_loss(trajectories)

            total_loss: torch.Tensor = loss_ce + lambda_constraint * loss_constraint + lambda_trajectory * loss_trajectory

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_constraint_loss += loss_constraint.item()
            epoch_trajectory_loss += loss_trajectory.item()

            progress_bar.set_postfix({
                "Total Loss": total_loss.item(),
                "CE Loss": loss_ce.item(),
                "Constraint Loss": loss_constraint.item(),
                "Trajectory Loss": loss_trajectory.item()
            })

        avg_loss = epoch_loss / len(dataloader)
        avg_ce_loss = epoch_ce_loss / len(dataloader)
        avg_constraint_loss = epoch_constraint_loss / len(dataloader)
        avg_trajectory_loss = epoch_trajectory_loss / len(dataloader)

        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"CE Loss: {avg_ce_loss:.4f}, Constraint Loss: {avg_constraint_loss:.4f}, "
              f"Trajectory Loss: {avg_trajectory_loss:.4f}, Lambda Constraint: {lambda_constraint:.4f}")

        if val_dataloader is not None:
            val_loss, val_ce_loss, val_constraint_loss, val_trajectory_loss = validate_dot_model(
                model, val_dataloader, num_timesteps, device, lambda_constraint, lambda_trajectory, num_trajectories
            )
            print(f"Validation: Loss: {val_loss:.4f}, CE Loss: {val_ce_loss:.4f}, "
                f"Constraint Loss: {val_constraint_loss:.4f}, Trajectory Loss: {val_trajectory_loss:.4f}")
