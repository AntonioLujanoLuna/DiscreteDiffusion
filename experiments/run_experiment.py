#!/usr/bin/env python3
"""
Script to run Sudoku diffusion experiments with different configurations.

This script loads a configuration file, sets up the experiment, and runs the training
and evaluation processes. It provides a command-line interface for controlling
various experiment parameters.
"""

import argparse
import os
import sys
import torch
from datetime import datetime
from pathlib import Path

# Add src to the path
sys.path.insert(0, os.path.abspath("src"))

from src.config import ExperimentConfig, TrainingMode, ModelType
from src.main import run_experiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Sudoku diffusion experiments")
    
    parser.add_argument("--config", type=str, required=True,
                      help="Path to the configuration file")
    
    parser.add_argument("--name", type=str, default=None,
                      help="Experiment name (overrides config)")
    
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu, or cuda:n)")
    
    parser.add_argument("--batch_size", type=int, default=None,
                      help="Batch size (overrides config)")
    
    parser.add_argument("--epochs", type=int, default=None,
                      help="Number of epochs (overrides config)")
    
    parser.add_argument("--lr", type=float, default=None,
                      help="Learning rate (overrides config)")
    
    parser.add_argument("--clue_ratio", type=float, default=None,
                      help="Clue ratio (overrides config)")
    
    parser.add_argument("--timesteps", type=int, default=None,
                      help="Number of diffusion timesteps (overrides config)")
    
    parser.add_argument("--resume", type=str, default=None,
                      help="Path to checkpoint to resume from")
    
    parser.add_argument("--mode", type=str, default=None, choices=["Standard", "DDM", "DoT"],
                      help="Training mode (overrides config)")
    
    parser.add_argument("--model", type=str, default=None, choices=["Base", "Hybrid"],
                      help="Model type (overrides config)")
    
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed (overrides config)")
    
    parser.add_argument("--no_vis", action="store_true",
                      help="Disable visualization")
    
    parser.add_argument("--export_config", action="store_true",
                      help="Export the effective configuration after applying overrides")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from file."""
    try:
        return ExperimentConfig.load(config_path)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {str(e)}")
        sys.exit(1)


def apply_overrides(config, args):
    """Apply command-line overrides to configuration."""
    if args.name:
        config.experiment_name = args.name
    
    if args.device:
        config.device = args.device
    
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    
    if args.lr:
        config.training.learning_rate = args.lr
    
    if args.clue_ratio:
        config.data.clue_ratio = args.clue_ratio
    
    if args.timesteps:
        config.training.num_timesteps = args.timesteps
    
    if args.mode:
        config.training.mode = TrainingMode(args.mode)
    
    if args.model:
        config.model.model_type = ModelType(args.model)
    
    if args.seed:
        config.seed = args.seed
    
    if args.no_vis:
        config.logging.visualize_inference = False
        config.logging.export_inference = False
    
    # Add timestamp to experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.experiment_name = f"{config.experiment_name}_{timestamp}"
    
    return config


def export_effective_config(config, args):
    """Export the effective configuration after applying overrides."""
    if args.export_config:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"effective_config_{timestamp}.json"
        
        try:
            config.save(export_path)
            print(f"Exported effective configuration to {export_path}")
        except Exception as e:
            print(f"Error exporting configuration: {str(e)}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply command-line overrides
    config = apply_overrides(config, args)
    
    # Export effective configuration if requested
    export_effective_config(config, args)
    
    # Print experiment details
    print(f"Running experiment: {config.experiment_name}")
    print(f"Training mode: {config.training.mode}")
    print(f"Model: {config.model.model_type}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Timesteps: {config.training.num_timesteps}")
    
    # Run the experiment
    try:
        run_experiment(config, resume_from=args.resume)
        print("Experiment completed successfully!")
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()