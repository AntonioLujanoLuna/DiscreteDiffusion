import math
import torch
import torch.nn.functional as F

def forward_process(x0, t, num_timesteps, clue_mask):
    """
    Applies the discrete forward noising process with a cosine-based noise schedule.
    
    Args:
        x0 (torch.LongTensor): Solved board of shape (B, 9, 9) with tokens in {1,...,9}.
        t (torch.Tensor): Timestep tensor of shape (B, 1) with values in [0, num_timesteps].
        num_timesteps (int): Total diffusion timesteps.
        clue_mask (torch.FloatTensor): Clue mask of shape (B, 9, 9).
        
    Returns:
        x_t (torch.LongTensor): Noisy board of shape (B, 9, 9) with tokens in {0,...,9}.
    """
    # Compute noise probability with shape (B, 1)
    noise_prob = 1 - torch.cos((t / num_timesteps) * (math.pi / 2))
    
    # Reshape to (B, 1, 1) so it can be expanded to (B, 9, 9)
    noise_prob = noise_prob.view(-1, 1, 1)
    noise_prob_grid = noise_prob.expand_as(x0)
    
    random_tensor = torch.rand_like(x0, dtype=torch.float)
    noise_mask = (random_tensor < noise_prob_grid) & (clue_mask == 0)
    
    random_tokens = torch.randint(0, 10, x0.shape, device=x0.device)
    x_t = x0.clone()
    x_t[noise_mask] = random_tokens[noise_mask]
    return x_t


def compute_constraint_loss(logits):
    """
    Computes an auxiliary loss to enforce Sudoku constraints.
    For each row, column, and 3x3 block, the sum of predicted probabilities (for digits 1-9) should be 1.
    
    Args:
        logits (torch.FloatTensor): Logits of shape (B, 9, 9, num_tokens).
        
    Returns:
        loss (torch.FloatTensor): Scalar constraint loss.
    """
    p = F.softmax(logits, dim=-1)[..., 1:10]  # (B, 9, 9, 9)
    loss = 0.0
    batch_size = logits.size(0)
    
    # Row constraints.
    for i in range(9):
        row_sum = p[:, i, :, :].sum(dim=1)     # (B, 9)
        loss += ((row_sum - 1) ** 2).mean()
    
    # Column constraints.
    for j in range(9):
        col_sum = p[:, :, j, :].sum(dim=1)     # (B, 9)
        loss += ((col_sum - 1) ** 2).mean()
    
    # Block constraints.
    for bi in range(3):
        for bj in range(3):
            block = p[:, bi*3:bi*3+3, bj*3:bj*3+3, :]  # (B, 3, 3, 9)
            block_sum = block.reshape(batch_size, -1, 9).sum(dim=1)  # (B, 9)
            loss += ((block_sum - 1) ** 2).mean()
    
    return loss

def reverse_diffusion_inference(model, initial_board, clue_mask, num_timesteps, device):
    """
    Performs reverse diffusion inference to solve a Sudoku puzzle.
    Starting from an initial board (with noise in non-clue positions),
    iteratively denoises the board by sampling from the model at decreasing timesteps.
    
    Args:
        model (nn.Module): Trained denoiser model.
        initial_board (torch.LongTensor): Board of shape (1, 9, 9) with clues filled and noise in others.
        clue_mask (torch.FloatTensor): Clue mask of shape (1, 9, 9).
        num_timesteps (int): Total diffusion timesteps.
        device (torch.device): Device for inference.
    
    Returns:
        trajectory (list of np.ndarray): List of board states (as numpy arrays) showing the evolution.
    """
    model.eval()
    with torch.no_grad():
        current_board = initial_board.clone()  # (1,9,9)
        trajectory = [current_board.cpu().numpy()]
        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor = torch.tensor([[t]], device=device, dtype=torch.float)
            logits = model(current_board, t_tensor, clue_mask)  # (1,9,9,num_tokens)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, 9, 9)
            new_board = current_board.clone()
            non_clue = (clue_mask == 0)
            new_board[non_clue] = sampled[non_clue]
            current_board = new_board
            trajectory.append(current_board.cpu().numpy())
        return trajectory
