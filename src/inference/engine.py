"""
Inference engine for the Discrete Diffusion project.

This module provides the core inference implementation that supports different 
inference strategies for diffusion models.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Any, Callable, Union, Tuple

from .strategies import (
    InferenceStrategy, 
    StandardInferenceStrategy, 
    DDMInferenceStrategy,
    DoTInferenceStrategy
)
from diffusion import forward_process

class InferenceEngine:
    """
    Inference engine for diffusion models.
    
    This class encapsulates the inference logic for different diffusion approaches,
    providing a unified interface for running inference with different strategies.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_timesteps: int,
        strategy: InferenceStrategy,
        noise_schedule_fn: Optional[Callable] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model (torch.nn.Module): The trained model.
            device (torch.device): Device for inference.
            num_timesteps (int): Number of diffusion timesteps.
            strategy (InferenceStrategy): Strategy for inference.
            noise_schedule_fn (Callable, optional): Function for noise scheduling.
        """
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        self.strategy = strategy
        self.noise_schedule_fn = noise_schedule_fn
    
    def prepare_initial_board(
        self, 
        solved_board: torch.Tensor, 
        clue_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare the initial board for inference by setting non-clue cells to the noise token.
        
        Args:
            solved_board (torch.Tensor): Solved board of shape (1, 9, 9).
            clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
            
        Returns:
            torch.Tensor: Initial board with clues preserved and non-clues set to noise token.
        """
        # Set non-clue cells to the noise token (0)
        initial_board = solved_board.clone()
        initial_board[clue_mask == 0] = 0
        
        # Apply the forward process to add noise
        t_full = torch.full((1, 1), self.num_timesteps, device=self.device, dtype=torch.float)
        noisy_board = forward_process(
            initial_board, 
            t_full, 
            self.num_timesteps, 
            clue_mask, 
            noise_schedule_fn=self.noise_schedule_fn
        )
        
        return noisy_board
    
    def run_inference(
        self, 
        solved_board: torch.Tensor, 
        clue_mask: torch.Tensor,
        **kwargs
    ) -> Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
        """
        Run inference with the selected strategy.
        
        Args:
            solved_board (torch.Tensor): Solved board of shape (1, 9, 9).
            clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
            **kwargs: Additional arguments for the strategy.
            
        Returns:
            Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
                Either a list of board states or a tuple containing the final board and trajectories.
        """
        self.model.eval()
        
        # Prepare the initial board
        initial_board = self.prepare_initial_board(solved_board, clue_mask)
        
        # Run inference with the strategy
        with torch.no_grad():
            return self.strategy.run_inference(
                self.model, 
                initial_board, 
                clue_mask, 
                self.num_timesteps, 
                self.device,
                noise_schedule_fn=self.noise_schedule_fn,
                **kwargs
            )


def run_inference(
    model: torch.nn.Module,
    solved_board: torch.Tensor,
    clue_mask: torch.Tensor,
    num_timesteps: int,
    device: torch.device,
    mode: str = "standard",
    noise_schedule_fn: Optional[Callable] = None,
    **kwargs
) -> Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
    """
    Run inference with a diffusion model using the specified strategy.
    
    Args:
        model (torch.nn.Module): The trained model.
        solved_board (torch.Tensor): Solved board of shape (1, 9, 9).
        clue_mask (torch.Tensor): Clue mask of shape (1, 9, 9).
        num_timesteps (int): Number of diffusion timesteps.
        device (torch.device): Device for inference.
        mode (str): Inference mode ("standard", "ddm", or "dot").
        noise_schedule_fn (Callable, optional): Function for noise scheduling.
        **kwargs: Additional arguments for the strategy.
        
    Returns:
        Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
            Either a list of board states or a tuple containing the final board and trajectories.
    """
    # Select the appropriate strategy based on mode
    if mode == "standard":
        strategy = StandardInferenceStrategy()
    elif mode == "ddm":
        strategy = DDMInferenceStrategy()
    elif mode == "dot":
        strategy = DoTInferenceStrategy()
    else:
        raise ValueError(f"Unsupported inference mode: {mode}")
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        device=device,
        num_timesteps=num_timesteps,
        strategy=strategy,
        noise_schedule_fn=noise_schedule_fn
    )
    
    # Run inference
    return engine.run_inference(solved_board, clue_mask, **kwargs)