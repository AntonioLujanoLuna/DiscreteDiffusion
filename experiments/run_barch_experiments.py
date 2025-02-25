#!/usr/bin/env python3
"""
Script to run a batch of Sudoku diffusion experiments with different configurations.

This script allows running multiple experiments sequentially with different parameters
or a grid search over parameter combinations.
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from itertools import product
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch of Sudoku diffusion experiments")
    
    parser.add_argument("--batch_config", type=str, required=True,
                      help="Path to the batch configuration file")
    
    parser.add_argument("--output_dir", type=str, default="batch_results",
                      help="Directory to store experiment outputs")
    
    parser.add_argument("--sequential", action="store_true",
                      help="Run experiments sequentially instead of grid search")
    
    parser.add_argument("--dry_run", action="store_true",
                      help="Print experiment commands without executing")
    
    return parser.parse_args()


def load_batch_config(config_path):
    """Load batch configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading batch configuration from {config_path}: {str(e)}")
        sys.exit(1)


def run_grid_search(base_config, parameter_grid, output_dir, dry_run=False):
    """Run a grid search over parameter combinations."""
    # Extract parameter names and values
    param_names = list(parameter_grid.keys())
    param_values = [parameter_grid[name] for name in param_names]
    
    # Generate all combinations
    combinations = list(product(*param_values))
    total_experiments = len(combinations)
    
    print(f"Preparing to run {total_experiments} experiments in grid search")
    
    # Process each combination
    for i, combination in enumerate(combinations):
        params = dict(zip(param_names, combination))
        
        # Create a name for this experiment
        experiment_name = base_config.get("experiment_name", "batch_experiment")
        for name, value in params.items():
            # Use only the last part of the parameter path for the name
            param_short = name.split('.')[-1]
            experiment_name += f"_{param_short}_{value}"
        
        # Create config for this experiment
        experiment_config = base_config.copy()
        experiment_config["experiment_name"] = experiment_name
        
        # Apply parameters
        for param_path, value in params.items():
            # Handle nested parameters with dot notation
            parts = param_path.split('.')
            config_pointer = experiment_config
            for part in parts[:-1]:
                if part not in config_pointer:
                    config_pointer[part] = {}
                config_pointer = config_pointer[part]
            config_pointer[parts[-1]] = value
        
        # Save config to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(output_dir, f"config_{i}_{timestamp}.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        # Build command
        cmd = [
            "python", "run_experiment.py",
            "--config", config_path,
            "--export_config"
        ]
        
        # Execute or print the command
        if dry_run:
            print(f"[Experiment {i+1}/{total_experiments}] Command: {' '.join(cmd)}")
        else:
            print(f"[Experiment {i+1}/{total_experiments}] Running: {experiment_name}")
            subprocess.run(cmd, check=True)


def run_sequential_experiments(experiments, output_dir, dry_run=False):
    """Run a sequence of experiments with specific configurations."""
    total_experiments = len(experiments)
    
    print(f"Preparing to run {total_experiments} sequential experiments")
    
    for i, experiment in enumerate(experiments):
        # Get experiment configuration
        config = experiment.get("config", {})
        
        # Save config to temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_path = os.path.join(output_dir, f"config_{i}_{timestamp}.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Build command
        cmd = [
            "python", "run_experiment.py",
            "--config", config_path,
            "--export_config"
        ]
        
        # Add any additional command line arguments
        if "args" in experiment:
            for arg_name, arg_value in experiment["args"].items():
                cmd.append(f"--{arg_name}")
                if arg_value is not None:  # Skip for flags without values
                    cmd.append(str(arg_value))
        
        # Execute or print the command
        if dry_run:
            print(f"[Experiment {i+1}/{total_experiments}] Command: {' '.join(cmd)}")
        else:
            print(f"[Experiment {i+1}/{total_experiments}] Running with config: {config_path}")
            subprocess.run(cmd, check=True)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load batch configuration
    batch_config = load_batch_config(args.batch_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.sequential:
        # Run sequential experiments
        run_sequential_experiments(
            batch_config.get("experiments", []),
            args.output_dir,
            dry_run=args.dry_run
        )
    else:
        # Run grid search
        run_grid_search(
            batch_config.get("base_config", {}),
            batch_config.get("parameter_grid", {}),
            args.output_dir,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()