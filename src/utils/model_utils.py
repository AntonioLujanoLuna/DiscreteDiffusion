"""
Model-related utility functions for the Discrete Diffusion project.

This module provides utility functions for creating, initializing, and
manipulating diffusion models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple

def create_model_from_config(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        config (Dict[str, Any]): Model configuration dictionary.
        device (torch.device): Device to place the model on.
        
    Returns:
        nn.Module: Instantiated model.
        
    Raises:
        ValueError: If an unsupported model type is specified.
    """
    model_type = config.get("model_type", "Base")
    num_tokens = config.get("num_tokens", 10)
    hidden_dim = config.get("hidden_dim", 128)
    num_layers = config.get("num_layers", 6)
    nhead = config.get("nhead", 8)
    dropout = config.get("dropout", 0.1)
    
    # Import here to avoid circular imports
    from models import ImprovedSudokuDenoiser, HybridSudokuDenoiser
    
    if model_type == "Base":
        model = ImprovedSudokuDenoiser(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        ).to(device)
    elif model_type == "Hybrid":
        num_conv_layers = config.get("num_conv_layers", 2)
        conv_kernel_size = config.get("conv_kernel_size", 3)
        model = HybridSudokuDenoiser(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            num_conv_layers=num_conv_layers,
            conv_kernel_size=conv_kernel_size,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def initialize_weights(model: nn.Module) -> None:
    """
    Initialize the weights of a model.
    
    Args:
        model (nn.Module): Model to initialize.
    """
    for name, p in model.named_parameters():
        if "weight" in name:
            if len(p.shape) >= 2:
                # For linear and convolutional layers
                nn.init.xavier_uniform_(p)
            else:
                # For 1D weights like bias terms
                nn.init.zeros_(p)
        elif "bias" in name:
            nn.init.zeros_(p)