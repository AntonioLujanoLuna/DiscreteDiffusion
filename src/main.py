"""
main.py

The main entry point for training and evaluating Sudoku discrete diffusion models.
Uses a configuration-based approach for experiment management and provides
a unified interface for different training modes (Standard, DDM, DoT).
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any

# Import configuration handling
from config import (
    ExperimentConfig, ModelType, TrainingMode,
    get_default_config, get_ddm_config, get_dot_config
)

# Import from refactored modules
from models import ImprovedSudokuDenoiser, HybridSudokuDenoiser
from data import SudokuDataset, get_curriculum_clue_ratio
from utils import (
    set_seeds, 
    LearnedNoiseSchedule, 
    create_model_from_config,
    display_trajectory_interactive, 
    export_trajectory_json
)
from training import (
    run_training_loop,
    LossStrategy,
    StandardLossStrategy,
    DDMLossStrategy,
    DoTLossStrategy
)
from inference import run_inference
from checkpoint import CheckpointManager, load_checkpoint
from logger import get_logger


def run_experiment(config: ExperimentConfig, resume_from: Optional[str] = None) -> None:
    """
    Run a complete experiment based on the provided configuration.
    
    Args:
        config (ExperimentConfig): Configuration for the experiment
        resume_from (str, optional): Path to checkpoint to resume from
    """
    # Initialize logging and device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger = get_logger(
        experiment_name=config.experiment_name,
        log_dir=config.logging.log_dir,
        use_tensorboard=config.logging.use_tensorboard
    )
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Using device: {device}")
    logger.log_config(config)
    
    # Set seeds for reproducibility
    set_seeds(config.seed)
    logger.info(f"Random seed set to {config.seed}")
    
    # Create model
    model = create_model_from_config(config.model.__dict__, device)
    
    # Create datasets and dataloaders
    train_dataset = SudokuDataset(
        num_samples=config.data.num_samples,
        clue_ratio=config.data.clue_ratio,
        augment=config.data.augment,
        ensure_unique=config.data.ensure_unique
    )
    
    val_dataset = SudokuDataset(
        num_samples=config.data.val_samples,
        clue_ratio=config.data.clue_ratio,
        augment=config.data.augment,
        ensure_unique=config.data.ensure_unique
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    
    # Initialize noise schedule if enabled
    noise_schedule_fn = None
    if config.training.use_learned_noise:
        noise_schedule_fn = LearnedNoiseSchedule(hidden_dim=16).to(device)
        logger.info("Using learned noise schedule")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.logging.checkpoint_dir,
        experiment_name=config.experiment_name,
        max_to_keep=5
    )
    checkpoint_manager.save_experiment_metadata(config)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from:
        checkpoint = load_checkpoint(
            checkpoint_path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Select appropriate loss strategy based on training mode
    if config.training.mode == TrainingMode.STANDARD:
        loss_strategy = StandardLossStrategy()
        auxiliary_loss_weights = None
        logger.info("Using standard diffusion training")
    elif config.training.mode == TrainingMode.DDM:
        loss_strategy = DDMLossStrategy()
        auxiliary_loss_weights = {"evidence_loss": config.training.lambda_evidence}
        logger.info("Using DDM (Diffusion Decision Model) training")
    elif config.training.mode == TrainingMode.DOT:
        loss_strategy = DoTLossStrategy()
        auxiliary_loss_weights = {"trajectory_loss": config.training.lambda_trajectory}
        logger.info("Using DoT (Diffusion-of-Thought) training")
    else:
        raise ValueError(f"Unsupported training mode: {config.training.mode}")
    
    # Define a function to update dataset's clue ratio (for curriculum learning)
    def set_epoch_ratio(ratio):
        train_dataset.set_epoch_ratio(ratio)
        val_dataset.set_epoch_ratio(ratio)
    
    # Run training loop
    logger.info("Starting training")
    metrics_history = run_training_loop(
        model=model,
        dataloader=train_loader,
        num_timesteps=config.training.num_timesteps,
        optimizer=optimizer,
        device=device,
        num_epochs=config.training.num_epochs,
        loss_strategy=loss_strategy,
        use_amp=True,
        set_epoch_ratio=set_epoch_ratio,
        get_curriculum_clue_ratio=get_curriculum_clue_ratio,
        auxiliary_loss_weights=auxiliary_loss_weights,
        initial_lambda_constraint=config.training.lambda_constraint,
        start_ratio=config.data.start_ratio,
        end_ratio=config.data.end_ratio,
        start_epoch=start_epoch,
        noise_schedule_fn=noise_schedule_fn,
        scheduler=scheduler,
        val_dataloader=val_loader,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        log_freq=config.logging.log_freq,
        save_freq=config.logging.save_freq,
        # Additional parameters for specific training modes
        threshold=config.training.threshold,
        num_trajectories=config.training.num_trajectories
    )
    
    # Run inference on a sample
    logger.info("Running inference on a sample")
    sample = val_dataset[0]
    solved_board = sample["solved_board"].unsqueeze(0).to(device)
    clue_mask = sample["clue_mask"].unsqueeze(0).to(device)
    
    # Select appropriate inference mode based on training mode
    inference_mode = config.training.mode.value.lower()
    
    # Run inference
    logger.info(f"Running {inference_mode} inference")
    if inference_mode == "dot":
        # DoT returns a tuple (final_board, trajectories)
        final_board, trajectories = run_inference(
            model=model,
            solved_board=solved_board,
            clue_mask=clue_mask,
            num_timesteps=config.training.num_timesteps,
            device=device,
            mode=inference_mode,
            noise_schedule_fn=noise_schedule_fn,
            num_trajectories=config.training.num_trajectories,
            threshold=config.training.threshold
        )
        # Use the first trajectory for visualization
        trajectory = trajectories[0]
    else:
        # Standard and DDM return a list of board states
        trajectory = run_inference(
            model=model,
            solved_board=solved_board,
            clue_mask=clue_mask,
            num_timesteps=config.training.num_timesteps,
            device=device,
            mode=inference_mode,
            noise_schedule_fn=noise_schedule_fn,
            threshold=config.training.threshold
        )
    
    # Export and visualize inference results
    if config.logging.export_inference:
        export_trajectory_json(trajectory)
        logger.info("Exported trajectory to JSON")
    
    if config.logging.visualize_inference:
        display_trajectory_interactive(trajectory, clue_mask=sample["clue_mask"].cpu().numpy())
        logger.info("Displayed interactive trajectory visualization")
    
    logger.info("Experiment completed successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sudoku Discrete Diffusion Training")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file"
    )
    parser.add_argument(
        "--mode", type=str, choices=["Standard", "DDM", "DoT"], default="Standard",
        help="Training mode: Standard, DDM (Diffusion Decision Model), or DoT (Diffusion-of-Thought)"
    )
    parser.add_argument(
        "--model", type=str, choices=["Base", "Hybrid"], default="Base",
        help="Model architecture: Base or Hybrid"
    )
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Experiment name"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        # Select default config based on mode
        if args.mode == "DDM":
            config = get_ddm_config()
        elif args.mode == "DoT":
            config = get_dot_config()
        else:
            config = get_default_config()
    
    # Override with command line arguments
    if args.name:
        config.experiment_name = args.name
    
    config.training.mode = TrainingMode(args.mode)
    config.model.model_type = ModelType(args.model)
    
    # Run the experiment
    run_experiment(config, resume_from=args.resume)


if __name__ == "__main__":
    main()