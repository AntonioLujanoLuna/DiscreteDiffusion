"""
Configuration module for the Discrete Diffusion project.

This module provides configuration management for experiments.
"""

from .experiment import (
    ExperimentConfig,
    ModelType,
    TrainingMode,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    get_default_config,
    get_ddm_config,
    get_dot_config
)

__all__ = [
    'ExperimentConfig',
    'ModelType',
    'TrainingMode',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'LoggingConfig',
    'get_default_config',
    'get_ddm_config',
    'get_dot_config'
]