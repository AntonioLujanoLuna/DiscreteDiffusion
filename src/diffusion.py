import math
import torch
import torch.nn.functional as F

def forward_process(x0, t, num_timesteps, clue_mask, gamma: float = 1.0):
    """
    Applies the discrete forward noising process with a parameterized cosine-based noise schedule.
    
    Args:
        x0 (torch.LongTensor): Solved board of shape (B, 9, 9) with tokens in {1,...,9}.
        t (torch.Tensor): Timestep tensor of shape (B, 1) with values in [0, num_timesteps].
        num_timesteps (int): Total diffusion timesteps.
        clue_mask (torch.FloatTensor): Clue mask of shape (B, 9, 9).
        gamma (float, optional): Exponent to modulate the noise schedule. 
                                 gamma=1.0 recovers the original cosine schedule.
    
    Returns:
        torch.LongTensor: Noisy board of shape (B, 9, 9) with tokens in {0,...,9}.
    """
    # Adjust time fraction using the gamma parameter for schedule flexibility.
    time_fraction = (t / num_timesteps) ** gamma
    noise_prob = 1 - torch.cos(time_fraction * (math.pi / 2))  # (B, 1)
    
    # Reshape and expand noise probability to match x0 dimensions.
    noise_prob = noise_prob.view(-1, 1, 1)
    noise_prob_grid = noise_prob.expand_as(x0)
    
    # Generate a random mask for non-clue cells.
    random_tensor = torch.rand_like(x0, dtype=torch.float)
    noise_mask = (random_tensor < noise_prob_grid) & (clue_mask == 0)
    
    # Replace non-clue positions with random tokens from 0 to 9.
    random_tokens = torch.randint(0, 10, x0.shape, device=x0.device)
    x_t = x0.clone()
    x_t[noise_mask] = random_tokens[noise_mask]
    return x_t


def compute_constraint_loss(logits):
    """
    Computes a rigorous constraint loss that enforces Sudoku rules by penalizing
    pairwise conflicts in rows, columns, and 3x3 blocks. For each group (row, column,
    block) and for each digit (1-9), the conflict is computed as:
    
        conflict = ( (sum of probabilities)^2 - sum of squared probabilities ) / 2

    In an ideal Sudoku row/column/block, exactly one cell should assign probability 1 
    to a given digit and the rest 0, leading to a conflict of 0.
    
    Args:
        logits (torch.FloatTensor): Logits of shape (B, 9, 9, num_tokens).
    
    Returns:
        torch.FloatTensor: Scalar constraint loss.
    """
    # Convert logits to probabilities (exclude the noise token at index 0)
    p = F.softmax(logits, dim=-1)[..., 1:10]  # shape: (B, 9, 9, 9)
    batch_size = logits.size(0)
    loss = 0.0

    # Row conflicts: iterate over each row index.
    for i in range(9):
        # p_row: (B, 9, 9) -> for row i, over 9 columns and 9 digits.
        p_row = p[:, i, :, :]
        # For each digit, compute conflict = ( (sum over cells)^2 - (sum of squares) ) / 2.
        sum_over_cols = p_row.sum(dim=1)          # shape: (B, 9)
        sum_sq = (p_row ** 2).sum(dim=1)            # shape: (B, 9)
        conflict = (sum_over_cols ** 2 - sum_sq) / 2
        loss += conflict.mean()

    # Column conflicts: iterate over each column index.
    for j in range(9):
        # p_col: (B, 9, 9) -> for column j, over 9 rows and 9 digits.
        p_col = p[:, :, j, :]
        sum_over_rows = p_col.sum(dim=1)           # shape: (B, 9)
        sum_sq = (p_col ** 2).sum(dim=1)             # shape: (B, 9)
        conflict = (sum_over_rows ** 2 - sum_sq) / 2
        loss += conflict.mean()

    # Block conflicts: iterate over each 3x3 block.
    for bi in range(3):
        for bj in range(3):
            # Extract block: (B, 3, 3, 9) then reshape to (B, 9, 9) (9 cells, 9 digits)
            block = p[:, bi*3:(bi+1)*3, bj*3:(bj+1)*3, :].reshape(batch_size, -1, 9)
            sum_over_block = block.sum(dim=1)        # shape: (B, 9)
            sum_sq = (block ** 2).sum(dim=1)           # shape: (B, 9)
            conflict = (sum_over_block ** 2 - sum_sq) / 2
            loss += conflict.mean()

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
