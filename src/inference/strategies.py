"""
Inference strategies for the Discrete Diffusion project.

This module provides different strategies for running inference with diffusion models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Callable, Tuple, Any, Union

from diffusion import forward_process

class InferenceStrategy:
    """
    Base class for inference strategies using a template method pattern.
    
    This class defines the common algorithm structure for inference,
    while allowing subclasses to implement strategy-specific steps.
    """
    
    def run_inference(
        self,
        model: torch.nn.Module,
        initial_board: torch.Tensor,
        clue_mask: torch.Tensor,
        num_timesteps: int,
        device: torch.device,
        **kwargs
    ) -> Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
        """
        Template method that defines the inference algorithm structure.
        
        Args:
            model (torch.nn.Module): The trained model.
            initial_board (torch.Tensor): Initial noisy board.
            clue_mask (torch.Tensor): Clue mask.
            num_timesteps (int): Number of diffusion timesteps.
            device (torch.device): Device for inference.
            **kwargs: Additional arguments.
            
        Returns:
            Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
                Either a list of board states or a tuple containing the final board and trajectories.
        """
        # Initialize strategy-specific state
        state = self._initialize_state(model, initial_board, clue_mask, device, **kwargs)
        
        # Track the trajectory
        trajectory = [initial_board.cpu().numpy()]
        
        # Run reverse diffusion
        for t in reversed(range(1, num_timesteps + 1)):
            t_tensor = torch.tensor([[t]], device=device, dtype=torch.float)
            
            # Generate predictions for current timestep
            logits = model(state["current_board"], t_tensor, clue_mask)
            probs = F.softmax(logits, dim=-1)
            
            # Strategy-specific update step
            state = self._update_step(state, probs, t, clue_mask, **kwargs)
            
            # Add to trajectory
            trajectory.append(state["current_board"].cpu().numpy())
        
        # Finalize and return results
        return self._finalize_results(state, trajectory, **kwargs)
    
    def _initialize_state(
        self,
        model: torch.nn.Module,
        initial_board: torch.Tensor,
        clue_mask: torch.Tensor,
        device: torch.device,
        **kwargs
    ) -> Dict[str, Any]:
        """Initialize strategy-specific state. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_state")
    
    def _update_step(
        self,
        state: Dict[str, Any],
        probs: torch.Tensor,
        t: int,
        clue_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform a single reverse diffusion step. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _update_step")
    
    def _finalize_results(
        self,
        state: Dict[str, Any],
        trajectory: List[np.ndarray],
        **kwargs
    ) -> Union[List[np.ndarray], Tuple[np.ndarray, List[List[np.ndarray]]]]:
        """Finalize and return results. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _finalize_results")


class StandardInferenceStrategy(InferenceStrategy):
    """Standard diffusion inference strategy."""
    
    def _initialize_state(
        self,
        model: torch.nn.Module,
        initial_board: torch.Tensor,
        clue_mask: torch.Tensor,
        device: torch.device,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "current_board": initial_board.clone()
        }
    
    def _update_step(
        self,
        state: Dict[str, Any],
        probs: torch.Tensor,
        t: int,
        clue_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        # Sample from probability distribution
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[0], 9, 9)
        
        # Create new board, keeping clue cells unchanged
        new_board = state["current_board"].clone()
        non_clue = (clue_mask == 0)
        new_board[non_clue] = sampled[non_clue]
        
        return {"current_board": new_board}
    
    def _finalize_results(
        self,
        state: Dict[str, Any],
        trajectory: List[np.ndarray],
        **kwargs
    ) -> List[np.ndarray]:
        return trajectory


class DDMInferenceStrategy(InferenceStrategy):
    """Diffusion Decision Model (DDM) inference strategy."""
    
    def _initialize_state(
        self,
        model: torch.nn.Module,
        initial_board: torch.Tensor,
        clue_mask: torch.Tensor,
        device: torch.device,
        **kwargs
    ) -> Dict[str, Any]:
        num_tokens = model.num_tokens
        return {
            "current_board": initial_board.clone(),
            "accumulator": torch.zeros((initial_board.size(0), 9, 9, num_tokens), device=device),
            "decision_made": (clue_mask == 1),
            "final_decision": initial_board.clone()
        }
    
    def _update_step(
        self,
        state: Dict[str, Any],
        probs: torch.Tensor,
        t: int,
        clue_mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        # Get parameters from kwargs
        threshold = kwargs.get("threshold", 0.9)
        beta = kwargs.get("beta", 0.9)
        tau = kwargs.get("tau", 0.05)
        
        # Get state variables
        current_board = state["current_board"]
        accumulator = state["accumulator"]
        decision_made = state["decision_made"]
        final_decision = state["final_decision"]
        
        # Update accumulator with momentum
        update_mask = (clue_mask == 0) & (~decision_made)
        accumulator[update_mask] = beta * accumulator[update_mask] + (1 - beta) * probs[update_mask]
        
        # Compute adaptive threshold
        adaptive_threshold = threshold * (0.5 + 0.5 * (t / kwargs.get("num_timesteps", 100)))
        
        # Find max evidence and corresponding indices
        max_values, max_indices = accumulator.max(dim=-1)
        
        # Compute soft commit gate
        commit_gate = torch.sigmoid((max_values - adaptive_threshold) / tau)
        
        # Determine cells to commit
        hard_commit_mask = (commit_gate > 0.8) & update_mask
        
        # Update final decision
        final_decision[hard_commit_mask] = max_indices[hard_commit_mask]
        decision_made[hard_commit_mask] = True
        
        # For non-decided cells, sample from distribution
        fallback_mask = (clue_mask == 0) & (~decision_made)
        if fallback_mask.any():
            fallback_probs = probs[fallback_mask]
            sampled_tokens = torch.multinomial(fallback_probs, num_samples=1).squeeze(-1)
            final_decision[fallback_mask] = sampled_tokens
        
        return {
            "current_board": final_decision.clone(),
            "accumulator": accumulator,
            "decision_made": decision_made,
            "final_decision": final_decision
        }
    
    def _finalize_results(
        self,
        state: Dict[str, Any],
        trajectory: List[np.ndarray],
        **kwargs
    ) -> List[np.ndarray]:
        return trajectory


class DoTInferenceStrategy(InferenceStrategy):
    """Diffusion of Thought (DoT) inference strategy."""
    
    def run_inference(
        self,
        model: torch.nn.Module,
        initial_board: torch.Tensor,
        clue_mask: torch.Tensor,
        num_timesteps: int,
        device: torch.device,
        **kwargs
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        """
        Override the template method for DoT since it requires multiple trajectories.
        """
        num_trajectories = kwargs.get("num_trajectories", 5)
        trajectories = []
        final_boards = []
        
        # Generate multiple trajectories using the standard strategy
        standard_strategy = StandardInferenceStrategy()
        for _ in range(num_trajectories):
            trajectory = standard_strategy.run_inference(
                model, initial_board, clue_mask, num_timesteps, device
            )
            trajectories.append(trajectory)
            final_boards.append(torch.tensor(trajectory[-1]))
        
        # Compute mode/majority vote
        boards_tensor = torch.stack(final_boards, dim=0)
        final_board = torch.mode(boards_tensor, dim=0).values.cpu().numpy()
        
        return final_board, trajectories