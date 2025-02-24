"""
Data module for the Discrete Diffusion project.

This module provides dataset classes and utilities for Sudoku data generation and manipulation.
"""

from .dataset import SudokuDataset
from .sudoku_utils import (
    generate_full_sudoku,
    create_sudoku_puzzle,
    create_unique_sudoku_puzzle,
    get_curriculum_clue_ratio
)

__all__ = [
    'SudokuDataset',
    'generate_full_sudoku',
    'create_sudoku_puzzle',
    'create_unique_sudoku_puzzle',
    'get_curriculum_clue_ratio'
]