"""
Training engine for the Discrete Diffusion project.

This module provides the core training and validation loop implementations
that support different diffusion training paradigms.
"""

import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Callable, Optional, Any, List, Union

from .losses import LossStrategy, StandardLossStrategy

class TrainingEngine:
    """
    Training engine for diffusion models.
    
    This class encapsulates the training logic for different diffusion approaches,
    providing a unified interface for training loops, validation, and checkpointing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_strategy: LossStrategy,
        device: torch.device,
        logger: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        checkpoint_manager: Optional[Any] = None
    ):
        """
        Initialize the training engine.
        
        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            loss_strategy (LossStrategy): Strategy for computing losses.
            device (torch.device): Device for training.
            logger (Any, optional): Logger for tracking metrics.
            scheduler (Any, optional): Learning rate scheduler.
            checkpoint_manager (Any, optional): Manager for saving/loading checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_strategy = loss_strategy
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.metrics_history = {}
    
    def train_epoch(
    self,
    dataloader: DataLoader,
    epoch: int,
    num_epochs: int,
    num_timesteps: int,
    lambda_constraint: float,
    noise_schedule_fn: Optional[Any] = None,
    log_freq: int = 10,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False
) -> Dict[str, float]:
        """
        Train for one epoch with optional mixed precision.
        
        Args:
            dataloader (DataLoader): Training dataloader.
            epoch (int): Current epoch number.
            num_epochs (int): Total number of epochs.
            num_timesteps (int): Total diffusion timesteps.
            lambda_constraint (float): Weight for constraint loss.
            noise_schedule_fn (Any, optional): Function for noise scheduling.
            log_freq (int): Frequency of logging training metrics (in batches).
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
            use_amp (bool): Whether to use Automatic Mixed Precision.
            
        Returns:
            Dict[str, float]: Dictionary of average metrics for the epoch.
        """
        self.model.train()
        epoch_metrics = self._init_metrics()
        
        # Create progress bar
        if self.logger:
            progress_description = f"Epoch {epoch+1}/{num_epochs}"
        else:
            progress_description = f"Epoch {epoch+1}/{num_epochs} [λ: {lambda_constraint:.2f}]"
        
        progress_bar = tqdm(dataloader, desc=progress_description, leave=False)
        epoch_start_time = time.time()
        
        # Batch loop
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            batch_size = batch["solved_board"].size(0)
            
            # Sample timesteps
            t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=self.device).float()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute losses using strategy (with or without mixed precision)
            if use_amp and scaler is not None:
                # Use autocast for mixed precision
                with torch.cuda.amp.autocast():
                    loss_dict = self.loss_strategy.compute_losses(
                        self.model, batch, t, num_timesteps, self.device,
                        noise_schedule_fn=noise_schedule_fn
                    )
                    
                    # Apply constraint loss weight
                    total_loss = loss_dict["total_loss"]
                    if "constraint_loss" in loss_dict:
                        total_loss = total_loss + lambda_constraint * loss_dict["constraint_loss"]
                
                # Use scaler for backward and optimizer step
                scaler.scale(total_loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                # Standard precision training
                loss_dict = self.loss_strategy.compute_losses(
                    self.model, batch, t, num_timesteps, self.device,
                    noise_schedule_fn=noise_schedule_fn
                )
                
                # Apply constraint loss weight
                total_loss = loss_dict["total_loss"]
                if "constraint_loss" in loss_dict:
                    total_loss = total_loss + lambda_constraint * loss_dict["constraint_loss"]
                
                # Standard backward and optimizer step
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update metrics
            self._update_batch_metrics(epoch_metrics, loss_dict, total_loss)
            
            # Update progress bar
            progress_dict = self._create_progress_dict(loss_dict, total_loss)
            progress_bar.set_postfix(progress_dict)
            
            # Log batch metrics
            self.global_step += 1
            if self.logger and batch_idx % log_freq == 0:
                metrics_to_log = {
                    **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()},
                    "total_loss": total_loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "lambda_constraint": lambda_constraint
                }
                self.logger.log_metrics(metrics_to_log, self.global_step, prefix="train")
        
        # Compute epoch average metrics
        avg_metrics = self._compute_avg_metrics(epoch_metrics, len(dataloader))
        
        # Log epoch metrics
        if self.logger:
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            self.logger.log_epoch(avg_metrics, epoch, prefix="train")
        else:
            # Print metrics if logger not available
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items())
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {metrics_str}")
        
        return avg_metrics
    
    def validate(
    self,
    dataloader: DataLoader,
    num_timesteps: int,
    lambda_constraint: float,
    noise_schedule_fn: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False
) -> Dict[str, float]:
        """
        Run validation on the given dataloader with optional mixed precision.
        
        Args:
            dataloader (DataLoader): Validation dataloader.
            num_timesteps (int): Total diffusion timesteps.
            lambda_constraint (float): Weight for constraint loss.
            noise_schedule_fn (Any, optional): Function for noise scheduling.
            scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
            use_amp (bool): Whether to use Automatic Mixed Precision.
            
        Returns:
            Dict[str, float]: Dictionary of validation metrics.
        """
        self.model.eval()
        
        # Initialize metrics
        val_metrics = self._init_metrics()
        num_batches = len(dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                batch = self._move_to_device(batch)
                batch_size = batch["solved_board"].size(0)
                
                # Sample timesteps
                t = torch.randint(1, num_timesteps + 1, (batch_size, 1), device=self.device).float()
                
                # Compute losses using strategy (with or without mixed precision)
                if use_amp and scaler is not None:
                    # Use autocast for mixed precision
                    with torch.cuda.amp.autocast():
                        loss_dict = self.loss_strategy.compute_losses(
                            self.model, batch, t, num_timesteps, self.device,
                            noise_schedule_fn=noise_schedule_fn
                        )
                        
                        # Apply constraint loss weight
                        total_loss = loss_dict["total_loss"]
                        if "constraint_loss" in loss_dict:
                            total_loss = total_loss + lambda_constraint * loss_dict["constraint_loss"]
                else:
                    # Standard precision
                    loss_dict = self.loss_strategy.compute_losses(
                        self.model, batch, t, num_timesteps, self.device,
                        noise_schedule_fn=noise_schedule_fn
                    )
                    
                    # Apply constraint loss weight
                    total_loss = loss_dict["total_loss"]
                    if "constraint_loss" in loss_dict:
                        total_loss = total_loss + lambda_constraint * loss_dict["constraint_loss"]
                
                # Update metrics
                self._update_batch_metrics(val_metrics, loss_dict, total_loss)
        
        # Compute average metrics
        avg_metrics = self._compute_avg_metrics(val_metrics, num_batches)
        
        self.model.train()
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics_history: Dict[str, List[float]], is_best: bool = False) -> None:
        """
        Save a checkpoint if checkpoint manager is provided.
        
        Args:
            epoch (int): Current epoch number.
            metrics_history (Dict[str, List[float]]): History of metrics.
            is_best (bool): Whether this is the best model so far.
        """
        if self.checkpoint_manager:
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics_history": metrics_history,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss
            }
            
            if self.scheduler:
                state_dict["scheduler_state_dict"] = self.scheduler.state_dict()
            
            self.checkpoint_manager.save(
                state_dict=state_dict,
                epoch=epoch,
                step=self.global_step,
                metric_value=self.best_val_loss,
                is_best=is_best
            )
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Move all tensors in batch to device and preprocess data for training.
        This combines multiple operations to minimize device transfers.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch.
            device (torch.device): Target device.
            
        Returns:
            Dict[str, torch.Tensor]: Processed batch on device.
        """
        processed_batch = {}
        
        # Move tensors to device and preprocess in one pass
        for key, tensor in batch.items():
            # Handle different tensor types appropriately
            if key == "solved_board" or key == "puzzle_board":
                # For boards, ensure correct dtype for embedding lookup
                processed_batch[key] = tensor.to(device, dtype=torch.long)
            elif key == "clue_mask":
                # For masks, ensure float dtype for multiplication
                processed_batch[key] = tensor.to(device, dtype=torch.float)
            else:
                # Default handling for other tensors
                processed_batch[key] = tensor.to(device)
                
        return processed_batch
    
    def _init_metrics(self) -> Dict[str, float]:
        """Initialize metrics dictionary with zeros."""
        metrics = {
            "total_loss": 0.0,
        }
        # Add strategy-specific metrics
        for key in self.loss_strategy.get_metric_keys():
            metrics[key] = 0.0
        return metrics
    
    def _update_batch_metrics(
        self, 
        metrics: Dict[str, float], 
        loss_dict: Dict[str, torch.Tensor],
        total_loss: torch.Tensor
    ) -> None:
        """Update metrics with batch results."""
        metrics["total_loss"] += total_loss.item()
        for key, value in loss_dict.items():
            if key in metrics:
                metrics[key] += value.item()
    
    def _compute_avg_metrics(self, metrics: Dict[str, float], num_batches: int) -> Dict[str, float]:
        """Compute average metrics over the epoch."""
        return {key: value / num_batches for key, value in metrics.items()}
    
    def _create_progress_dict(
        self, 
        loss_dict: Dict[str, torch.Tensor],
        total_loss: torch.Tensor
    ) -> Dict[str, float]:
        """Create a dictionary for tqdm progress bar."""
        # Start with total loss
        progress_dict = {"loss": total_loss.item()}
        
        # Add other losses (with shortened keys)
        for key, value in loss_dict.items():
            if key != "total_loss":
                short_key = key[:6]  # Shorten key for display
                progress_dict[short_key] = value.item()
        
        return progress_dict


def run_training_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_timesteps: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    compute_losses: Optional[Callable] = None,
    loss_strategy: Optional[LossStrategy] = None,
    set_epoch_ratio: Optional[Callable[[float], None]] = None,
    get_curriculum_clue_ratio: Optional[Callable[[int, int, float, float], float]] = None,
    auxiliary_loss_weights: Optional[Dict[str, float]] = None,
    initial_lambda_constraint: float = 1.0,
    start_ratio: float = 0.9,
    end_ratio: float = 0.1,
    start_epoch: int = 0,
    noise_schedule_fn: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    val_dataloader: Optional[DataLoader] = None,
    checkpoint_manager: Optional[Any] = None,
    logger: Optional[Any] = None,
    log_freq: int = 10,
    save_freq: int = 1,
    use_amp: bool = False,  # New parameter to control AMP usage
    **compute_losses_kwargs
) -> Dict[str, List[float]]:
    """
    Enhanced training loop that supports different diffusion training paradigms,
    with logging, checkpointing, and validation capabilities.

    Args:
        model (torch.nn.Module): The denoiser model
        dataloader (DataLoader): Training dataloader
        num_timesteps (int): Total diffusion timesteps
        optimizer (torch.optim.Optimizer): Optimizer for training
        device (torch.device): Training device
        num_epochs (int): Number of epochs for training
        compute_losses (Callable, optional): Legacy function to compute losses
        loss_strategy (LossStrategy, optional): Strategy for computing losses (preferred over compute_losses)
        set_epoch_ratio (Callable, optional): Function to update dataset's clue ratio
        get_curriculum_clue_ratio (Callable, optional): Function to compute clue ratio for curriculum learning
        auxiliary_loss_weights (Dict[str, float], optional): Weights for auxiliary losses
        initial_lambda_constraint (float): Initial weight for constraint loss
        start_ratio (float): Starting clue ratio for curriculum learning
        end_ratio (float): Ending clue ratio for curriculum learning
        start_epoch (int): Starting epoch (for resuming training)
        noise_schedule_fn (Any, optional): Function for computing noise schedule
        scheduler (Any, optional): Learning rate scheduler
        val_dataloader (DataLoader, optional): Validation dataloader
        checkpoint_manager (Any, optional): Manager for saving/loading checkpoints
        logger (Any, optional): Logger for tracking metrics
        log_freq (int): Frequency of logging training metrics (in batches)
        save_freq (int): Frequency of saving checkpoints (in epochs)
        use_amp (bool): Whether to use Automatic Mixed Precision for training
        **compute_losses_kwargs: Additional arguments for the compute_losses function

    Returns:
        Dict[str, List[float]]: Dictionary of training metrics history
    """
    # Ensure we have a loss strategy
    if loss_strategy is None:
        if compute_losses is not None:
            # Wrapper for legacy compute_losses function
            loss_strategy = LossStrategyAdapter(compute_losses, **compute_losses_kwargs)
        else:
            # Default to standard loss strategy
            loss_strategy = StandardLossStrategy()
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_strategy=loss_strategy,
        device=device,
        logger=logger,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager
    )
    
    # Initialize AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Prepare metrics history
    metrics_history = _initialize_metrics_history(loss_strategy, val_dataloader is not None)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Update clue ratio for curriculum learning if provided
        if set_epoch_ratio and get_curriculum_clue_ratio:
            current_ratio = get_curriculum_clue_ratio(epoch, num_epochs, start_ratio, end_ratio)
            set_epoch_ratio(current_ratio)
            if logger:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Clue Ratio: {current_ratio:.3f}")
        
        # Decay lambda_constraint over epochs
        lambda_constraint = initial_lambda_constraint * (0.95 ** epoch)
        
        # Train for one epoch
        train_metrics = engine.train_epoch(
            dataloader=dataloader,
            epoch=epoch,
            num_epochs=num_epochs,
            num_timesteps=num_timesteps,
            lambda_constraint=lambda_constraint,
            noise_schedule_fn=noise_schedule_fn,
            log_freq=log_freq,
            scaler=scaler,  # Pass the scaler to the train_epoch function
            use_amp=use_amp  # Pass the use_amp flag
        )
        
        # Update metrics history with training metrics
        _update_metrics_history(metrics_history, train_metrics, prefix="train")
        
        # Run validation if dataloader provided
        is_best = False
        if val_dataloader:
            val_metrics = engine.validate(
                dataloader=val_dataloader,
                num_timesteps=num_timesteps,
                lambda_constraint=lambda_constraint,
                noise_schedule_fn=noise_schedule_fn,
                scaler=scaler,  # Pass the scaler for validation too
                use_amp=use_amp  # Pass the use_amp flag
            )
            
            # Update metrics history with validation metrics
            _update_metrics_history(metrics_history, val_metrics, prefix="val")
            
            # Log validation metrics
            if logger:
                logger.log_epoch(val_metrics, epoch, prefix="val")
            else:
                val_metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"Epoch {epoch+1}/{num_epochs} - Val: {val_metrics_str}")
            
            # Update learning rate scheduler if provided
            if scheduler:
                scheduler.step(val_metrics.get("total_loss", 0.0))
            
            # Check if this is the best model so far
            val_loss = val_metrics.get("total_loss", float("inf"))
            is_best = val_loss < engine.best_val_loss
            if is_best:
                engine.best_val_loss = val_loss
        
        # Save checkpoint if needed
        if (epoch + 1) % save_freq == 0:
            engine.save_checkpoint(epoch, metrics_history, is_best)
    
    # Return metrics history
    return metrics_history


def run_validation_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_timesteps: int,
    device: torch.device,
    compute_losses: Optional[Callable] = None,
    loss_strategy: Optional[LossStrategy] = None,
    use_amp: bool = False,  # New parameter for AMP
    **compute_losses_kwargs
) -> Dict[str, float]:
    """
    Run a validation loop on the given dataloader.

    Args:
        model (torch.nn.Module): The model to validate
        dataloader (DataLoader): Validation dataloader
        num_timesteps (int): Total diffusion timesteps
        device (torch.device): Device for computation
        compute_losses (Callable, optional): Legacy function to compute losses
        loss_strategy (LossStrategy, optional): Strategy for computing losses (preferred)
        use_amp (bool): Whether to use Automatic Mixed Precision
        **compute_losses_kwargs: Additional arguments for compute_losses

    Returns:
        Dict[str, float]: Dictionary of validation metrics
    """
    # Ensure we have a loss strategy
    if loss_strategy is None:
        if compute_losses is not None:
            # Wrapper for legacy compute_losses function
            loss_strategy = LossStrategyAdapter(compute_losses, **compute_losses_kwargs)
        else:
            # Default to standard loss strategy
            loss_strategy = StandardLossStrategy()
    
    # Initialize scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Create a temporary engine for validation
    engine = TrainingEngine(
        model=model,
        optimizer=None,  # Not needed for validation
        loss_strategy=loss_strategy,
        device=device
    )
    
    # Run validation
    return engine.validate(
        dataloader=dataloader,
        num_timesteps=num_timesteps,
        lambda_constraint=compute_losses_kwargs.get('lambda_constraint', 1.0),
        noise_schedule_fn=compute_losses_kwargs.get('noise_schedule_fn'),
        scaler=scaler,
        use_amp=use_amp
    )


class LossStrategyAdapter(LossStrategy):
    """Adapter for legacy compute_losses functions to the LossStrategy interface."""
    
    def __init__(self, compute_losses: Callable, **kwargs):
        """
        Initialize the adapter.
        
        Args:
            compute_losses (Callable): Legacy compute_losses function
            **kwargs: Additional arguments to pass to compute_losses
        """
        self.compute_losses = compute_losses
        self.kwargs = kwargs
    
    def compute_losses(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor], 
        t: torch.Tensor, 
        num_timesteps: int, 
        device: torch.device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses using the legacy function.
        
        Args:
            model (torch.nn.Module): The model
            batch (Dict[str, torch.Tensor]): Batch of data
            t (torch.Tensor): Timestep tensor
            num_timesteps (int): Total diffusion timesteps
            device (torch.device): Device for computation
            **kwargs: Additional arguments
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of loss components
        """
        # Merge kwargs with self.kwargs
        all_kwargs = {**self.kwargs, **kwargs}
        return self.compute_losses(model, batch, t, kwargs.get('noise_schedule_fn'), num_timesteps, device, **all_kwargs)
    
    def get_metric_keys(self) -> List[str]:
        """
        Get the keys for metrics to track.
        
        Returns:
            List[str]: List of metric keys
        """
        # Since we don't know what the compute_losses function returns,
        # we provide a default set of keys
        return ["total_loss", "ce_loss", "constraint_loss"]


def _initialize_metrics_history(
    loss_strategy: LossStrategy, 
    has_validation: bool
) -> Dict[str, List[float]]:
    """
    Initialize metrics history dictionary.
    
    Args:
        loss_strategy (LossStrategy): Strategy for computing losses
        has_validation (bool): Whether validation metrics should be included
        
    Returns:
        Dict[str, List[float]]: Empty metrics history
    """
    metrics_history = {}
    
    # Add training metrics
    for key in ["total_loss"] + loss_strategy.get_metric_keys():
        metrics_history[f"train_{key}"] = []
    
    # Add validation metrics if needed
    if has_validation:
        for key in ["total_loss"] + loss_strategy.get_metric_keys():
            metrics_history[f"val_{key}"] = []
    
    return metrics_history


def _update_metrics_history(
    metrics_history: Dict[str, List[float]],
    metrics: Dict[str, float],
    prefix: str
) -> None:
    """
    Update metrics history with new metrics.
    
    Args:
        metrics_history (Dict[str, List[float]]): Metrics history to update
        metrics (Dict[str, float]): New metrics to add
        prefix (str): Prefix for metric keys (e.g., 'train' or 'val')
    """
    for key, value in metrics.items():
        history_key = f"{prefix}_{key}"
        if history_key in metrics_history:
            metrics_history[history_key].append(value)

def _create_noise_cache(
    noise_schedule_fn: Optional[Callable], 
    num_timesteps: int, 
    device: torch.device
) -> Dict[int, torch.Tensor]:
    """
    Create a cache for noise schedule values if a function is provided.
    
    Args:
        noise_schedule_fn (Callable, optional): Function to compute noise probability.
        num_timesteps (int): Total diffusion timesteps.
        device (torch.device): Device for computation.
        
    Returns:
        Dict[int, torch.Tensor]: Cache of noise probabilities, or empty dict if no function.
    """
    noise_cache = {}
    if noise_schedule_fn is not None:
        for t_val in range(1, num_timesteps + 1):
            t_tensor = torch.tensor([[t_val]], device=device, dtype=torch.float)
            noise_cache[t_val] = noise_schedule_fn(t_tensor, num_timesteps)
    return noise_cache


def _get_cached_noise_fn(noise_cache: Dict[int, torch.Tensor]) -> Optional[Callable]:
    """
    Create a function that uses the cached noise values.
    
    Args:
        noise_cache (Dict[int, torch.Tensor]): Cache of noise probabilities.
        
    Returns:
        Callable or None: Function that returns cached values, or None if cache is empty.
    """
    if not noise_cache:
        return None
        
    def cached_noise_fn(t_tensor: torch.Tensor, num_timesteps: int) -> torch.Tensor:
        return noise_cache[int(t_tensor.item())]
        
    return cached_noise_fn

