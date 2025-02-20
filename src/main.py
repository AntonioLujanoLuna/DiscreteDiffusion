import torch
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import SudokuDataset
from model import SudokuDenoiser, ImprovedSudokuDenoiser
from train import train_diffusion, train_diffusion_curriculum_learning
from inference import run_inference
from visualization import display_trajectory, display_trajectory_interactive

def main():
    # Set seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Hyperparameters.
    num_samples = 5000
    clue_ratio = 0.3
    batch_size = 32
    num_epochs = 30
    num_timesteps = 100
    learning_rate = 1e-4
    hidden_dim = 128
    num_layers = 8
    nhead = 8
    num_tokens = 10  # 0: noise token; 1-9: Sudoku digits.
    lambda_constraint = 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Starting Training ===")
    print(f"Device: {device}")
    print(f"Hyperparameters: num_samples={num_samples}, batch_size={batch_size}, num_epochs={num_epochs}, num_timesteps={num_timesteps}")
    
    # Create dataset and dataloader.
    dataset = SudokuDataset(num_samples=num_samples, clue_ratio=clue_ratio, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer.
    #model = SudokuDenoiser(num_tokens=num_tokens, hidden_dim=hidden_dim,
    #                       num_layers=num_layers, nhead=nhead, dropout=0.1).to(device)
    model = ImprovedSudokuDenoiser(num_tokens=num_tokens, hidden_dim=hidden_dim,
                       num_layers=num_layers, nhead=nhead, dropout=0.1).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Train the diffusion model.
    #train_diffusion(model, dataloader, num_timesteps, optimizer, device,
    #                num_epochs=num_epochs, lambda_constraint=lambda_constraint)
    
    train_diffusion_curriculum_learning(model, dataloader, num_timesteps, optimizer, device,
                num_epochs=num_epochs, lambda_constraint=lambda_constraint)

    print("=== Training Complete ===\n")
    
    # ---- Inference and Visualization ----
    print("=== Starting Inference ===")
    # For inference, take one sample from the dataset.
    sample = dataset[0]
    solved_board = sample["solved_board"].unsqueeze(0).to(device)  # (1, 9, 9)
    clue_mask = sample["clue_mask"].unsqueeze(0).to(device)          # (1, 9, 9)
    
    trajectory = run_inference(model, solved_board, clue_mask, num_timesteps, device)
    print("=== Inference Complete ===\n")
    
    # Visualize the reverse diffusion trajectory.
    # display_trajectory(trajectory)
    display_trajectory_interactive(trajectory, clue_mask=sample["clue_mask"].cpu().numpy())

if __name__ == '__main__':
    main()
