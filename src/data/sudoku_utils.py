"""
Utility functions for Sudoku generation and validation.

This module provides functions for generating Sudoku boards, creating puzzles,
and validating board properties.
"""

import random
import numpy as np
from typing import Tuple, List, Union, Optional

def count_solutions(board: np.ndarray) -> int:
    """
    Counts the number of solutions for a given Sudoku puzzle using backtracking.
    The function stops counting once more than one solution is found.
    
    Args:
        board (np.ndarray): A 9x9 numpy array representing a Sudoku puzzle 
                         (with 0 for empty cells).
    
    Returns:
        int: The number of solutions found (typically 0, 1, or >1).
    """
    count = [0]  # Using a list to allow modification within the nested function

    def is_valid_move(b, row, col, num):
        if num in b[row]:
            return False
        for r in range(9):
            if b[r][col] == num:
                return False
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if b[r][c] == num:
                    return False
        return True

    def solve(b):
        # Find the first empty cell
        for i in range(9):
            for j in range(9):
                if b[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid_move(b, i, j, num):
                            b[i][j] = num
                            solve(b)
                            b[i][j] = 0
                    return  # Backtrack if no valid number was found
        count[0] += 1
        # Early exit if more than one solution is found
        if count[0] > 1:
            return

    board_copy = board.copy().tolist()  # Work on a mutable copy
    solve(board_copy)
    return count[0]

def meets_minimum_distribution(puzzle: np.ndarray, min_clues: int = 2) -> bool:
    """
    Checks whether every row, column, and 3x3 block in the puzzle has at least `min_clues` clues.
    
    Args:
        puzzle (np.ndarray): A 9x9 numpy array representing the Sudoku puzzle (0 for empty).
        min_clues (int, optional): Minimum number of non-zero clues per row, column, and block.
    
    Returns:
        bool: True if the puzzle meets the minimum distribution, False otherwise.
    """
    # Check rows
    for row in puzzle:
        if np.count_nonzero(row) < min_clues:
            return False
    # Check columns
    for col in puzzle.T:
        if np.count_nonzero(col) < min_clues:
            return False
    # Check 3x3 blocks
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = puzzle[i:i+3, j:j+3]
            if np.count_nonzero(block) < min_clues:
                return False
    return True

def create_unique_sudoku_puzzle(
    solved_board: np.ndarray, 
    target_num_clues: int, 
    min_clues_per_unit: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a Sudoku puzzle with a unique solution by removing clues one-by-one from a solved board.
    The process stops when the number of clues reaches target_num_clues, and it also checks that every row,
    column, and block has at least `min_clues_per_unit` clues.
    
    Args:
        solved_board (np.ndarray): A fully solved 9x9 Sudoku board.
        target_num_clues (int): Desired number of clues in the final puzzle (typically between 17 and 81).
        min_clues_per_unit (int, optional): Minimum clues required in every row, column, and block.
    
    Returns:
        tuple: (puzzle_board, clue_mask)
            - puzzle_board (np.ndarray): A 9x9 puzzle board with non-clue cells set to 0.
            - clue_mask (np.ndarray): A 9x9 binary mask where 1 indicates a clue cell.
    """
    puzzle = solved_board.copy()
    positions = list(range(81))
    np.random.shuffle(positions)

    # Try to remove clues one by one
    for pos in positions:
        row, col = pos // 9, pos % 9
        if puzzle[row, col] != 0:
            backup = puzzle[row, col]
            puzzle[row, col] = 0
            # Check uniqueness and distribution; if not unique or distribution is too sparse, revert removal.
            if count_solutions(puzzle) != 1 or not meets_minimum_distribution(puzzle, min_clues_per_unit):
                puzzle[row, col] = backup
        # Stop if we have reached the target number of clues
        if np.count_nonzero(puzzle) <= target_num_clues:
            break

    clue_mask = (puzzle != 0).astype(np.float32)
    return puzzle, clue_mask

def create_sudoku_puzzle(
    solved_board: np.ndarray, 
    clue_ratio: Optional[float] = None, 
    num_clues: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a fully solved Sudoku board into a puzzle by removing clues based on difficulty.
    
    Either `clue_ratio` (a float between 0 and 1) or `num_clues` (an integer between 17 and 81)
    must be provided. If both are provided, `num_clues` takes precedence.
    
    Args:
        solved_board (np.ndarray): A 9x9 numpy array representing a fully solved Sudoku board.
        clue_ratio (float, optional): Fraction of cells to keep as clues (e.g., 0.3 means ~30% clues).
        num_clues (int, optional): Exact number of clues to retain in the puzzle.
        
    Returns:
        tuple: (puzzle_board, clue_mask)
            - puzzle_board (np.ndarray): A 9x9 array where non-clue cells are set to 0.
            - clue_mask (np.ndarray): A 9x9 binary mask with 1s at positions of clues and 0s elsewhere.
    """
    if num_clues is None and clue_ratio is None:
        raise ValueError("Either clue_ratio or num_clues must be provided.")
    
    # Determine the number of clues to keep.
    if num_clues is None:
        num_clues = int(round(clue_ratio * 81))
    
    # Ensure at least one clue is kept in case of extremely low difficulty.
    num_clues = max(1, num_clues)
    
    # Flatten indices and shuffle them.
    indices = np.arange(81)
    np.random.shuffle(indices)
    
    # Select the first num_clues indices to keep as clues.
    clue_indices = indices[:num_clues]
    
    # Create a binary mask of shape (9, 9): 1 indicates a clue cell, 0 indicates removal.
    mask_flat = np.zeros(81, dtype=np.float32)
    mask_flat[clue_indices] = 1.0
    clue_mask = mask_flat.reshape((9, 9))
    
    # Generate the puzzle board: keep clues and set non-clues to 0.
    puzzle_board = solved_board.copy()
    puzzle_board[clue_mask == 0] = 0
    
    return puzzle_board, clue_mask

def generate_full_sudoku() -> List[List[int]]:
    """
    Generates a fully solved Sudoku board using backtracking.
    
    Returns:
        board (list of lists): A 9x9 list with integers 1-9.
    """
    board = [[0 for _ in range(9)] for _ in range(9)]
    
    def is_valid(board, row, col, num):
        if num in board[row]:
            return False
        for r in range(9):
            if board[r][col] == num:
                return False
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if board[r][c] == num:
                    return False
        return True

    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if solve():
                                return True
                            board[i][j] = 0
                    return False
        return True

    solve()
    return board

def augment_sudoku(board: Union[List[List[int]], np.ndarray]) -> np.ndarray:
    """
    Applies data augmentation to a fully solved Sudoku board.
    
    Augmentations include:
      - A random permutation of the digits.
      - Random row swaps within each 3-row band.
      - Random column swaps within each 3-column stack.
      - A random rotation (0, 90, 180, or 270 degrees).
      - Random horizontal and vertical flips.
    
    Args:
        board (list of lists or np.ndarray): 9x9 solved Sudoku board.
        
    Returns:
        new_board (np.ndarray): Augmented 9x9 Sudoku board.
    """
    # Convert to numpy array if not already.
    board_np = np.array(board, dtype=np.int64)
    
    # ---------------------------
    # 1) Digit Permutation:
    # ---------------------------
    perm = np.random.permutation(np.arange(1, 10))
    digit_map = {i + 1: perm[i] for i in range(9)}
    aug_board = np.vectorize(lambda x: digit_map[x])(board_np)
    
    # ---------------------------
    # 2) Row Swap within Bands:
    # ---------------------------
    for band in range(3):
        rows = list(range(band * 3, band * 3 + 3))
        random.shuffle(rows)
        aug_board[band * 3:band * 3 + 3, :] = aug_board[rows, :]
    
    # ---------------------------
    # 3) Column Swap within Stacks:
    # ---------------------------
    for stack in range(3):
        cols = list(range(stack * 3, stack * 3 + 3))
        random.shuffle(cols)
        aug_board[:, stack * 3:stack * 3 + 3] = aug_board[:, cols]
    
    # ---------------------------
    # 4) Random Rotation:
    # ---------------------------
    # Rotate the board by 0, 90, 180, or 270 degrees
    k = np.random.randint(0, 4)
    aug_board = np.rot90(aug_board, k=k)
    
    # ---------------------------
    # 5) Random Flips:
    # ---------------------------
    # With probability 0.5, perform a horizontal flip.
    if np.random.rand() < 0.5:
        aug_board = np.fliplr(aug_board)
    # With probability 0.5, perform a vertical flip.
    if np.random.rand() < 0.5:
        aug_board = np.flipud(aug_board)
    
    return aug_board

def get_curriculum_clue_ratio(epoch: int, num_epochs: int, start_ratio: float = 0.9, end_ratio: float = 0.1) -> float:
    """
    Returns a clue_ratio for the given epoch that linearly moves from start_ratio 
    to end_ratio over num_epochs.
    
    Args:
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        start_ratio (float, optional): Initial clue ratio (typically higher).
        end_ratio (float, optional): Final clue ratio (typically lower).
        
    Returns:
        float: The interpolated clue ratio for the current epoch.
    """
    progress = epoch / (num_epochs - 1) if num_epochs > 1 else 1.0  # goes from 0.0 -> 1.0
    ratio = (1.0 - progress) * start_ratio + progress * end_ratio
    return ratio