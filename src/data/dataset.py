"""
Dataset class for the Discrete Diffusion project.

This module provides the SudokuDataset class for generating and managing Sudoku puzzles.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Tuple

from .sudoku_utils import (
    generate_full_sudoku,
    augment_sudoku,
    create_sudoku_puzzle,
    create_unique_sudoku_puzzle
)

class SudokuDataset(Dataset):
    """
    Dataset for generating Sudoku puzzles and their solutions.
    
    Generates full solved Sudoku boards and creates puzzles with a specified
    ratio of clue cells. Supports curriculum learning by adjusting the clue
    ratio during training.
    """
    
    def __init__(self, num_samples=1000, clue_ratio=0.3, augment=True, ensure_unique=False):
        """
        Initialize the Sudoku dataset.
        
        Args:
            num_samples (int): Number of Sudoku puzzles to generate.
            clue_ratio (float): Fraction of cells to keep as clues (0-1).
            augment (bool): Whether to apply data augmentation to the boards.
            ensure_unique (bool): Whether to ensure puzzles have unique solutions.
        """
        self.num_samples = num_samples
        self.initial_clue_ratio = clue_ratio   # We'll treat this as a 'default' or 'initial' ratio
        self.clue_ratio = clue_ratio           # The *current* ratio we will actually use
        self.augment = augment
        self.ensure_unique = ensure_unique

        # Generate all boards *once* and store them
        self.boards = []
        for _ in range(num_samples):
            board = generate_full_sudoku()  # returns a 9x9 solved board
            if self.augment:
                board = augment_sudoku(board)  # returns a 9x9 np array
            self.boards.append(board)

    def set_epoch_ratio(self, ratio: float) -> None:
        """
        Called externally (e.g., by the training loop at the start of each epoch)
        to update the clue ratio used to build the clue mask.
        
        Args:
            ratio (float): New clue ratio to use.
        """
        self.clue_ratio = ratio

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "solved_board": Full solved board (target)
                - "puzzle_board": Puzzle board (input)
                - "clue_mask": Binary mask indicating clue cells
        """
        board = self.boards[idx]
        board = board.copy()  # Ensure a contiguous array (fixes negative strides issue)
        
        # Generate a puzzle and clue mask using the current clue_ratio.
        if self.ensure_unique:
            # Assume self.target_num_clues is defined (or compute from clue_ratio)
            puzzle_board, clue_mask = create_unique_sudoku_puzzle(
                board, 
                target_num_clues=int(round(self.clue_ratio * 81))
            )
        else:
            puzzle_board, clue_mask = create_sudoku_puzzle(
                board, 
                clue_ratio=self.clue_ratio
            )
        
        # Convert arrays to torch tensors.
        solved_board_tensor = torch.tensor(board, dtype=torch.long)       # Full solved board (target)
        puzzle_board_tensor = torch.tensor(puzzle_board, dtype=torch.long)  # Puzzle board (input)
        clue_mask_tensor  = torch.tensor(clue_mask, dtype=torch.float32)    # Clue mask

        return {
            "solved_board": solved_board_tensor,
            "puzzle_board": puzzle_board_tensor,
            "clue_mask": clue_mask_tensor
        }