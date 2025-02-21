"""
ddm_training.py

This module extends the training of the discrete diffusion Sudoku solver to incorporate
a Diffusion Decision Model (DDM)-inspired mechanism. In this paradigm, during training we
simulate the reverse diffusion process while maintaining an evidence accumulator for each
cell. An auxiliary evidence consistency loss is computed to encourage the accumulator to
converge toward the ground truth digit distribution.

Functions:
    - simulate_reverse_diffusion_ddm: Runs a differentiable simulation of reverse diffusion,
      updating an evidence accumulator over multiple timesteps.
    - compute_evidence_loss: Computes an auxiliary loss that measures how well the accumulated
      evidence matches the target (solved) board.
    - train_diffusion_ddm: The main training loop that integrates the standard losses with the
      evidence consistency loss.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import the forward process and constraint loss from your diffusion module.
from diffusion import forward_process, compute_constraint_loss
from validate import validate_ddm_model
from ddm_utils import compute_evidence_loss, simulate_reverse_diffusion_ddm
from dataset import get_curriculum_clue_ratio

def train_diffusion_ddm(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_timesteps: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    initial_lambda_constraint: float = 1.0,
    lambda_evidence: float = 0.5,
    val_dataloader: torch.utils.data.DataLoader = None, 
    start_ratio=0.9, 
    end_ratio=0.1
) -> None:
    """
    Trains the discrete diffusion Sudoku solver using an extended objective that incorporates
    a DDM-inspired evidence consistency loss. In each batch, the reverse diffusion process is
    simulated with evidence accumulation. The total loss is a weighted sum of:
      - Standard cross-entropy loss (ignoring clue cells).
      - Constraint loss enforcing Sudoku rules.
      - Evidence consistency loss comparing the accumulated evidence with the ground truth.

    Args:
        model (nn.Module): The denoiser model.
        dataloader (torch.utils.data.DataLoader): Provides batches from the SudokuDataset.
        num_timesteps (int): Total diffusion timesteps.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device for training.
        num_epochs (int, optional): Number of epochs to train. Defaults to 100.
        initial_lambda_constraint (float, optional): Initial weight for the constraint loss.
                                                     Defaults to 1.0.
        lambda_evidence (float, optional): Weight for the evidence consistency loss.
                                           Defaults to 0.5.
        start_ratio (float): Start clue ratio for training 
        end_ratio (float): Ending clue ratio for training 
    """
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    for epoch in range(num_epochs):
        ratio = get_curriculum_clue_ratio(epoch, num_epochs, start_ratio, end_ratio)
        dataloader.dataset.set_epoch_ratio(ratio)

        lambda_constraint = initial_lambda_constraint * (0.95 ** epoch)
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_constraint_loss = 0.0
        epoch_evidence_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            solved_board: torch.Tensor = batch["solved_board"].to(device)
            clue_mask: torch.Tensor = batch["clue_mask"].to(device)
            batch_size: int = solved_board.size(0)

            t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=device).float()
            x_t: torch.Tensor = forward_process(solved_board, t, num_timesteps, clue_mask)
            logits: torch.Tensor = model(x_t, t, clue_mask)
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

            # Simulate reverse diffusion to get the evidence accumulator.
            _, accumulator = simulate_reverse_diffusion_ddm(
                model, x_t, clue_mask, num_timesteps, device, threshold=0.9
            )
            loss_evidence: torch.Tensor = compute_evidence_loss(accumulator, solved_board, clue_mask)

            total_loss: torch.Tensor = loss_ce + lambda_constraint * loss_constraint + lambda_evidence * loss_evidence

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_constraint_loss += loss_constraint.item()
            epoch_evidence_loss += loss_evidence.item()

            progress_bar.set_postfix({
                "Total Loss": total_loss.item(),
                "CE Loss": loss_ce.item(),
                "Constraint Loss": loss_constraint.item(),
                "Evidence Loss": loss_evidence.item()
            })

        avg_loss = epoch_loss / len(dataloader)
        avg_ce_loss = epoch_ce_loss / len(dataloader)
        avg_constraint_loss = epoch_constraint_loss / len(dataloader)
        avg_evidence_loss = epoch_evidence_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"CE Loss: {avg_ce_loss:.4f}, Constraint Loss: {avg_constraint_loss:.4f}, "
              f"Evidence Loss: {avg_evidence_loss:.4f}, Lambda Constraint: {lambda_constraint:.4f}")

        # Run validation if provided.
        if val_dataloader is not None:
            val_loss, val_ce_loss, val_constraint_loss, val_evidence_loss = validate_ddm_model(
                model, val_dataloader, num_timesteps, device, lambda_constraint, lambda_evidence
            )
            print(f"Validation: Loss: {val_loss:.4f}, CE Loss: {val_ce_loss:.4f}, "
                  f"Constraint Loss: {val_constraint_loss:.4f}, Evidence Loss: {val_evidence_loss:.4f}")

