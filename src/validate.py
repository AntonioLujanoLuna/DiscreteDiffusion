import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusion import compute_constraint_loss
from ddm_inference import forward_process
from ddm_utils import compute_evidence_loss, simulate_reverse_diffusion_ddm
from dot_utils import simulate_reverse_diffusion_dot, compute_trajectory_consistency_loss

def validate_model(model, dataloader, num_timesteps, device):
    """
    Computes the average validation loss over the given dataloader.
    
    Args:
        model (torch.nn.Module): The trained denoiser.
        dataloader (DataLoader): Validation dataloader.
        num_timesteps (int): Total diffusion timesteps.
        device (torch.device): Device for computation.
        
    Returns:
        tuple: (average total loss, average CE loss, average constraint loss)
    """
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            solved_board = batch["solved_board"].to(device)
            clue_mask = batch["clue_mask"].to(device)
            batch_size = solved_board.size(0)
            
            t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=device).float()
            x_t = forward_process(solved_board, t, num_timesteps, clue_mask)
            logits = model(x_t, t, clue_mask)
            
            logits_flat = logits.view(batch_size, -1, model.num_tokens)
            solved_board_flat = solved_board.view(batch_size, -1)
            clue_mask_flat = clue_mask.view(batch_size, -1)
            
            loss_ce = F.cross_entropy(
                logits_flat.view(-1, model.num_tokens),
                solved_board_flat.view(-1),
                reduction='none'
            )
            loss_ce = (loss_ce * (1 - clue_mask_flat.view(-1))).mean()
            loss_constraint = compute_constraint_loss(logits)
            
            total_loss += (loss_ce + loss_constraint).item()
            total_ce_loss += loss_ce.item()
            total_constraint_loss += loss_constraint.item()
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_constraint_loss = total_constraint_loss / len(dataloader)
    return avg_loss, avg_ce_loss, avg_constraint_loss

def validate_ddm_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_timesteps: int,
    device: torch.device,
    lambda_constraint: float,
    lambda_evidence: float,
) -> tuple:
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    total_evidence_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            solved_board = batch["solved_board"].to(device)
            clue_mask = batch["clue_mask"].to(device)
            batch_size = solved_board.size(0)

            t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=device).float()
            x_t = forward_process(solved_board, t, num_timesteps, clue_mask)
            logits = model(x_t, t, clue_mask)
            logits_flat = logits.view(batch_size, -1, logits.size(-1))
            solved_board_flat = solved_board.view(batch_size, -1)
            clue_mask_flat = clue_mask.view(batch_size, -1)

            loss_ce = F.cross_entropy(
                logits_flat.view(-1, logits.size(-1)),
                solved_board_flat.view(-1),
                reduction="none"
            )
            loss_ce = (loss_ce * (1 - clue_mask_flat.view(-1))).mean()
            loss_constraint = compute_constraint_loss(logits)

            _, accumulator = simulate_reverse_diffusion_ddm(
                model, x_t, clue_mask, num_timesteps, device, threshold=0.9
            )
            loss_evidence = compute_evidence_loss(accumulator, solved_board, clue_mask)

            batch_loss = loss_ce + lambda_constraint * loss_constraint + lambda_evidence * loss_evidence
            total_loss += batch_loss.item()
            total_ce_loss += loss_ce.item()
            total_constraint_loss += loss_constraint.item()
            total_evidence_loss += loss_evidence.item()

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_constraint_loss = total_constraint_loss / len(dataloader)
    avg_evidence_loss = total_evidence_loss / len(dataloader)
    return avg_loss, avg_ce_loss, avg_constraint_loss, avg_evidence_loss

def validate_dot_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_timesteps: int,
    device: torch.device,
    lambda_constraint: float,
    lambda_trajectory: float,
    num_trajectories: int,
) -> tuple:
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    total_trajectory_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            solved_board = batch["solved_board"].to(device)
            clue_mask = batch["clue_mask"].to(device)
            batch_size = solved_board.size(0)

            t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=device).float()
            x_t = forward_process(solved_board, t, num_timesteps, clue_mask)
            logits = model(x_t, t, clue_mask)
            logits_flat = logits.view(batch_size, -1, logits.size(-1))
            solved_board_flat = solved_board.view(batch_size, -1)
            clue_mask_flat = clue_mask.view(batch_size, -1)

            loss_ce = F.cross_entropy(
                logits_flat.view(-1, logits.size(-1)),
                solved_board_flat.view(-1),
                reduction="none"
            )
            loss_ce = (loss_ce * (1 - clue_mask_flat.view(-1))).mean()
            loss_constraint = compute_constraint_loss(logits)

            trajectories = simulate_reverse_diffusion_dot(
                model, x_t, clue_mask, num_timesteps, device, num_trajectories
            )
            loss_trajectory = compute_trajectory_consistency_loss(trajectories)

            batch_loss = loss_ce + lambda_constraint * loss_constraint + lambda_trajectory * loss_trajectory
            total_loss += batch_loss.item()
            total_ce_loss += loss_ce.item()
            total_constraint_loss += loss_constraint.item()
            total_trajectory_loss += loss_trajectory.item()

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_constraint_loss = total_constraint_loss / len(dataloader)
    avg_trajectory_loss = total_trajectory_loss / len(dataloader)
    return avg_loss, avg_ce_loss, avg_constraint_loss, avg_trajectory_loss
