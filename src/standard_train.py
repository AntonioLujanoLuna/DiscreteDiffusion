import torch
import torch.nn.functional as F
from diffusion import forward_process, compute_constraint_loss
from validate import validate_model
from dataset import get_curriculum_clue_ratio
from tqdm import tqdm

def train_diffusion(model, dataloader, num_timesteps, optimizer, device, num_epochs=100, 
    initial_lambda_constraint=1.0, val_dataloader=None, start_ratio=0.9, end_ratio=0.1):
    """
    Trains the diffusion model and optionally evaluates on a validation set.
    
    Args:
        model (torch.nn.Module): The denoiser.
        dataloader (DataLoader): Training dataloader.
        num_timesteps (int): Total diffusion timesteps.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Training device.
        num_epochs (int): Number of training epochs.
        initial_lambda_constraint (float): Initial weight for the constraint loss.
        val_dataloader (DataLoader, optional): Validation dataloader.
        start_ratio (float): Start clue ratio for training 
        end_ratio (float): Ending clue ratio for training 
    """
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    for epoch in range(num_epochs):
        # 1) Update ratio at start of epoch
        ratio = get_curriculum_clue_ratio(epoch, num_epochs, start_ratio, end_ratio)
        dataloader.dataset.set_epoch_ratio(ratio)
        
        # 2) Standard steps
        lambda_constraint = initial_lambda_constraint * (0.95 ** epoch)
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_constraint_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
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
            total_loss = loss_ce + lambda_constraint * loss_constraint
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_constraint_loss += loss_constraint.item()
            
            progress_bar.set_postfix({
                'Total Loss': total_loss.item(),
                'CE Loss': loss_ce.item(),
                'Constraint Loss': loss_constraint.item()
            })
        
        avg_loss = epoch_loss / len(dataloader)
        avg_ce_loss = epoch_ce_loss / len(dataloader)
        avg_constraint_loss = epoch_constraint_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, Constraint Loss: {avg_constraint_loss:.4f}, Lambda: {lambda_constraint:.4f}")
        
        # Run validation if provided.
        if val_dataloader is not None:
            val_loss, val_ce_loss, val_constraint_loss = validate_model(model, val_dataloader, num_timesteps, device)
            print(f"Validation: Loss: {val_loss:.4f}, CE Loss: {val_ce_loss:.4f}, Constraint Loss: {val_constraint_loss:.4f}")


def train_diffusion_curriculum_learning(model, dataloader, num_timesteps, optimizer, device, num_epochs=100, 
        lambda_constraint=1.0, curriculum_epochs=5, val_dataloader=None, start_ratio=0.95, end_ratio=0.05):
    """
    Trains the diffusion model using a curriculum strategy: starting with only the cross-entropy loss
    for a set number of epochs, then adding the constraint loss.
    
    Args:
        model (nn.Module): The denoiser.
        dataloader (DataLoader): Provides batches from the SudokuDataset.
        num_timesteps (int): Total diffusion timesteps.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Training device.
        num_epochs (int): Number of epochs.
        lambda_constraint (float): Weight for the constraint loss.
        curriculum_epochs (int): Number of epochs to train without the constraint loss.
        val_dataloader (DataLoader, optional): Validation dataloader.
        start_ratio (float): Start clue ratio for training 
        end_ratio (float): Ending clue ratio for training 
    """
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    for epoch in range(num_epochs):
        ratio = get_curriculum_clue_ratio(epoch, num_epochs, start_ratio, end_ratio)
        dataloader.dataset.set_epoch_ratio(ratio)

        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_constraint_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
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
            
            if epoch < curriculum_epochs:
                total_loss = loss_ce
            else:
                total_loss = loss_ce + lambda_constraint * loss_constraint
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_constraint_loss += loss_constraint.item()
            
            progress_bar.set_postfix({
                'Total Loss': total_loss.item(), 
                'CE Loss': loss_ce.item(), 
                'Constraint Loss': loss_constraint.item()
            })
        
        avg_loss = epoch_loss / len(dataloader)
        avg_ce_loss = epoch_ce_loss / len(dataloader)
        avg_constraint_loss = epoch_constraint_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, "
              f"Constraint Loss: {avg_constraint_loss:.4f}")

        # Run validation if provided.
        if val_dataloader is not None:
            val_loss, val_ce_loss, val_constraint_loss = validate_model(model, val_dataloader, num_timesteps, device)
            print(f"Validation: Loss: {val_loss:.4f}, CE Loss: {val_ce_loss:.4f}, Constraint Loss: {val_constraint_loss:.4f}")

