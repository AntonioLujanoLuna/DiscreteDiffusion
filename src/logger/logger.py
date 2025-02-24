"""
Logger implementation for the Discrete Diffusion project.

This module provides a structured logging system with console and file output,
as well as TensorBoard integration for metrics visualization.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Union, Any

# Conditionally import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Unified logging system that handles both text logging and metrics tracking.
    
    This class provides:
    - Console logging with color-coded levels
    - File logging to keep a permanent record
    - Optional TensorBoard integration for metrics visualization
    - Helper methods for logging common training events
    """
    
    def __init__(self, 
                 experiment_name: str, 
                 log_dir: str = "logs",
                 use_tensorboard: bool = True,
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG) -> None:
        """
        Initialize the logger.
        
        Args:
            experiment_name (str): Name of the experiment for log organization
            log_dir (str): Directory to store logs
            use_tensorboard (bool): Whether to use TensorBoard for metric logging
            console_level (int): Logging level for console output
            file_level (int): Logging level for file output
        """
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{timestamp}"
        
        # Create log directory
        self.run_log_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.run_log_dir, exist_ok=True)
        
        # Set up Python logger
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = os.path.join(self.run_log_dir, f"{self.run_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # TensorBoard writer
        self.writer = None
        if self.use_tensorboard:
            tensorboard_dir = os.path.join(self.run_log_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            self.info(f"TensorBoard writer initialized at {tensorboard_dir}")
        
        self.info(f"Logger initialized for experiment: {experiment_name}")
        self.info(f"Logs will be saved to: {self.run_log_dir}")
    
    def debug(self, msg: str) -> None:
        """Log a debug message."""
        self.logger.debug(msg)
    
    def info(self, msg: str) -> None:
        """Log an info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg)
    
    def critical(self, msg: str) -> None:
        """Log a critical message."""
        self.logger.critical(msg)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log metrics to TensorBoard and/or console.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            step (int): Current step/iteration
            prefix (str): Optional prefix for metric names (e.g., 'train/' or 'val/')
        """
        # Format for console output
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step} - {prefix} metrics: {metrics_str}")
        
        # Log to TensorBoard if available
        if self.writer is not None:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)
    
    def log_training_batch(self, metrics: Dict[str, float], epoch: int, batch_idx: int, num_batches: int) -> None:
        """
        Log metrics for a training batch.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            epoch (int): Current epoch number
            batch_idx (int): Current batch index
            num_batches (int): Total number of batches in the epoch
        """
        # Calculate global step
        step = epoch * num_batches + batch_idx
        
        # Add progress information
        progress = f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{num_batches}"
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"{progress} - {metrics_str}")
        
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"train/{name}", value, step)
    
    def log_epoch(self, metrics: Dict[str, float], epoch: int, prefix: str = "train") -> None:
        """
        Log metrics for a complete epoch.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            epoch (int): Current epoch number
            prefix (str): Prefix for metric names ('train' or 'val')
        """
        # Format for console output
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Epoch {epoch+1} - {prefix} metrics: {metrics_str}")
        
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{name}", value, epoch)
    
    def log_model_summary(self, model_summary: str) -> None:
        """Log a summary of the model architecture."""
        self.info(f"Model Summary:\n{model_summary}")
    
    def log_config(self, config: Any) -> None:
        """Log the configuration."""
        if hasattr(config, '__dict__'):
            config_str = str(config.__dict__)
        else:
            config_str = str(config)
        self.info(f"Configuration:\n{config_str}")
    
    def close(self) -> None:
        """Close the logger and any associated resources."""
        if self.writer is not None:
            self.writer.close()
        
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.info("Logger closed")


# Global logger instance
_LOGGER = None


def get_logger(experiment_name: str = None, log_dir: str = "logs", **kwargs) -> Logger:
    """
    Get or create the global logger instance.
    
    Args:
        experiment_name (str): Name of the experiment
        log_dir (str): Directory to store logs
        **kwargs: Additional arguments to pass to the Logger constructor
    
    Returns:
        Logger: The global logger instance
    """
    global _LOGGER
    if _LOGGER is None and experiment_name is not None:
        _LOGGER = Logger(experiment_name=experiment_name, log_dir=log_dir, **kwargs)
    return _LOGGER