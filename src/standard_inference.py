import torch
from diffusion import forward_process, reverse_diffusion_inference

def run_inference(model, solved_board, clue_mask, num_timesteps, device):
    """
    Prepares the initial board from a solved board and its clue mask, then runs reverse diffusion inference.
    
    Args:
        model (nn.Module): The trained denoiser.
        solved_board (torch.Tensor): Solved board of shape (1, 9, 9).
        clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
        num_timesteps (int): Total diffusion timesteps.
        device (torch.device): Device for inference.
        
    Returns:
        trajectory (list): List of board states showing the denoising trajectory.
    """
    # Create initial board: clues remain; non-clue cells set to noise token (0).
    initial_board = solved_board.clone()
    initial_board[clue_mask == 0] = 0
    # Optionally add extra noise.
    t_full = torch.full((1, 1), num_timesteps, device=device).float()
    initial_board = forward_process(initial_board, t_full, num_timesteps, clue_mask)
    
    trajectory = reverse_diffusion_inference(model, initial_board, clue_mask, num_timesteps, device)
    return trajectory
