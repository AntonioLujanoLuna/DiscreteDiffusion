"""
Experiment configuration for the Discrete Diffusion project.

This module defines dataclasses for structured configuration management,
with type checking and serialization capabilities.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple


class ModelType(str, Enum):
    """Valid model architecture types."""
    BASE = "Base"
    HYBRID = "Hybrid"


class TrainingMode(str, Enum):
    """Valid training mode types."""
    STANDARD = "Standard"
    DDM = "DDM"  # Diffusion Decision Model
    DOT = "DoT"  # Diffusion of Thought


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_type: ModelType = ModelType.BASE
    num_tokens: int = 10  # 0: noise token; 1-9: Sudoku digits
    hidden_dim: int = 128
    num_layers: int = 6
    nhead: int = 8
    dropout: float = 0.1
    
    # Hybrid model specific params (only used if model_type is HYBRID)
    num_conv_layers: int = 2
    conv_kernel_size: int = 3


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    num_samples: int = 2000
    clue_ratio: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    val_samples: int = 500
    augment: bool = True
    ensure_unique: bool = False
    start_ratio: float = 0.9  # Start clue ratio for curriculum learning
    end_ratio: float = 0.1    # End clue ratio for curriculum learning


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    mode: TrainingMode = TrainingMode.STANDARD
    num_epochs: int = 20
    num_timesteps: int = 100
    learning_rate: float = 1e-4
    lambda_constraint: float = 1.0
    
    # DDM specific params
    lambda_evidence: float = 0.5
    
    # DoT specific params
    lambda_trajectory: float = 0.5
    num_trajectories: int = 5
    
    # Inference params
    threshold: float = 0.9  # Used in DDM inference
    use_learned_noise: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging and tracking."""
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5  # Save checkpoint every N epochs
    log_freq: int = 100  # Log metrics every N training steps
    use_tensorboard: bool = True
    visualize_inference: bool = True
    export_inference: bool = True


@dataclass
class ExperimentConfig:
    """Master configuration for an experiment."""
    experiment_name: str = "default_experiment"
    seed: int = 42
    device: str = "cuda"  # or "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def save(self, filepath: str) -> None:
        """Save config to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert nested dicts to appropriate dataclass objects
        model_config = ModelConfig(**config_dict.pop('model'))
        data_config = DataConfig(**config_dict.pop('data'))
        training_config = TrainingConfig(**config_dict.pop('training'))
        logging_config = LoggingConfig(**config_dict.pop('logging'))
        
        return cls(
            **config_dict,
            model=model_config,
            data=data_config,
            training=training_config,
            logging=logging_config
        )


def get_default_config() -> ExperimentConfig:
    """Return the default configuration."""
    return ExperimentConfig()


def get_ddm_config() -> ExperimentConfig:
    """Return a configuration for DDM training."""
    config = get_default_config()
    config.training.mode = TrainingMode.DDM
    return config


def get_dot_config() -> ExperimentConfig:
    """Return a configuration for DoT training."""
    config = get_default_config()
    config.training.mode = TrainingMode.DOT
    return config