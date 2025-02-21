"""
main.py

This script selects and runs the training and inference procedures based on the MODE variable.
MODE can be:
    - "Standard": uses the standard diffusion training/inference.
    - "DDM": uses the DDM-inspired training/inference (evidence accumulation).
    - "DoT": uses the DoT-inspired training/inference (multiple trajectories and aggregation).

After training, a single sample is selected from the dataset for inference, and the reverse diffusion
trajectory is visualized.
"""

from typing import List
import torch
import random
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

# Import dataset and model.
from dataset import SudokuDataset
from model import ImprovedSudokuDenoiser  # Using the improved version.
from visualization import display_trajectory_interactive

# Set MODE here: "Standard", "DDM", or "DoT".
# MODE: str = "Standard"
MODE: str = "DDM"
# MODE: str = "DoT"

def main() -> None:
    # Set seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Hyperparameters.
    num_samples: int = 2000
    clue_ratio: float = 0.3
    batch_size: int = 32
    num_epochs: int = 30
    num_timesteps: int = 50
    learning_rate: float = 1e-4
    hidden_dim: int = 128
    num_layers: int = 6
    nhead: int = 8
    num_tokens: int = 10  # 0: noise token; 1-9: Sudoku digits.
    lambda_constraint: float = 1.0
    
    # Additional hyperparameters for DDM and DoT.
    lambda_evidence: float = 0.5   # Used in DDM training.
    lambda_trajectory: float = 0.5   # Used in DoT training.
    num_trajectories: int = 5        # For DoT training and inference.
    threshold: float = 0.9           # Used in DDM inference.
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=== Starting Training ===")
    print(f"Device: {device}")
    print(f"Hyperparameters: num_samples={num_samples}, batch_size={batch_size}, num_epochs={num_epochs}, num_timesteps={num_timesteps}")
    
    # Create dataset and dataloader.
    dataset: SudokuDataset = SudokuDataset(num_samples=num_samples, clue_ratio=clue_ratio, augment=True)
    dataloader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = SudokuDataset(num_samples=1000, clue_ratio=clue_ratio, augment=True) 
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and optimizer.
    model = ImprovedSudokuDenoiser(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nhead=nhead,
        dropout=0.1
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Choose training routine based on MODE.
    if MODE == "Standard":
        print("Training with standard diffusion training...")
        from standard_train import train_diffusion_curriculum_learning as train_fn
        train_fn(model, dataloader, num_timesteps, optimizer, device,
                 num_epochs=num_epochs, lambda_constraint=lambda_constraint, val_dataloader=val_dataloader)
    elif MODE == "DDM":
        from ddm_train import train_diffusion_ddm as train_fn
        train_fn(model, dataloader, num_timesteps, optimizer, device,
                num_epochs=num_epochs, initial_lambda_constraint=lambda_constraint,
                lambda_evidence=lambda_evidence, val_dataloader=val_dataloader)
    elif MODE == "DoT":
        from dot_train import train_diffusion_dot as train_fn
        train_fn(model, dataloader, num_timesteps, optimizer, device,
                num_epochs=num_epochs, initial_lambda_constraint=lambda_constraint,
                lambda_trajectory=lambda_trajectory, num_trajectories=num_trajectories,
                val_dataloader=val_dataloader)
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")
    
    print("=== Training Complete ===\n")
    
    # ---- Inference and Visualization ----
    print("=== Starting Inference ===")
    # Select one sample from the dataset for inference.
    sample = dataset[0]
    solved_board = sample["solved_board"].unsqueeze(0).to(device)  # (1, 9, 9)
    clue_mask = sample["clue_mask"].unsqueeze(0).to(device)          # (1, 9, 9)
    
    # Choose inference routine based on MODE.
    if MODE == "Standard":
        from standard_inference import run_inference as inference_fn
        print("Running standard reverse diffusion inference...")
        trajectory: List[np.ndarray] = inference_fn(model, solved_board, clue_mask, num_timesteps, device)
    elif MODE == "DDM":
        from ddm_inference import run_inference_ddm as inference_fn
        print("Running DDM-based inference...")
        trajectory: List[np.ndarray] = inference_fn(model, solved_board, clue_mask, num_timesteps, device, threshold=threshold)
    elif MODE == "DoT":
        from dot_inference import run_inference_dot as inference_fn
        print("Running DoT-based inference...")
        # run_inference_dot returns a tuple: final aggregated board and the list of trajectories.
        _, trajectory = inference_fn(model, solved_board, clue_mask, num_timesteps, device, num_trajectories=num_trajectories)
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")
    
    print("=== Inference Complete ===\n")
    
    # Visualize the reverse diffusion trajectory.
    display_trajectory_interactive(trajectory, clue_mask=sample["clue_mask"].cpu().numpy())
    
if __name__ == "__main__":
    main()
