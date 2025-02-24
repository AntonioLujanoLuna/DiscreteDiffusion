"""
test_dataset.py

Unit tests for the dataset components of the Discrete Diffusion project.
Ensures that Sudoku dataset creation, augmentation, and puzzle generation
work correctly.
"""

import torch
import numpy as np
import unittest
import sys
import os

# Add the src directory to the path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dataset import (
    SudokuDataset, generate_full_sudoku, augment_sudoku, create_sudoku_puzzle,
    count_solutions, meets_minimum_distribution, create_unique_sudoku_puzzle
)


class TestSudokuDataset(unittest.TestCase):
    """Test suite for Sudoku dataset functionality."""
    
    def test_generate_full_sudoku(self):
        """Test generate_full_sudoku function."""
        board = generate_full_sudoku()
        
        # Check shape
        self.assertEqual(len(board), 9)
        self.assertEqual(len(board[0]), 9)
        
        # Check that all values are valid Sudoku digits
        for row in board:
            for cell in row:
                self.assertTrue(1 <= cell <= 9, f"Cell value {cell} is not a valid Sudoku digit (1-9)")
        
        # Check that rows have no duplicates
        for row in board:
            self.assertEqual(len(set(row)), 9, "Row has duplicate digits")
        
        # Check that columns have no duplicates
        for col in range(9):
            column = [board[row][col] for row in range(9)]
            self.assertEqual(len(set(column)), 9, "Column has duplicate digits")
        
        # Check that 3x3 blocks have no duplicates
        for block_row in range(0, 9, 3):
            for block_col in range(0, 9, 3):
                block = [board[block_row + i][block_col + j] 
                        for i in range(3) for j in range(3)]
                self.assertEqual(len(set(block)), 9, "Block has duplicate digits")
    
    def test_augment_sudoku(self):
        """Test augment_sudoku function."""
        # Generate a board
        original_board = generate_full_sudoku()
        
        # Augment it
        augmented_board = augment_sudoku(original_board)
        
        # Check that it's still a valid board
        self.assertEqual(augmented_board.shape, (9, 9))
        
        # Check that all values are valid Sudoku digits
        for row in augmented_board:
            for cell in row:
                self.assertTrue(1 <= cell <= 9, f"Cell value {cell} is not a valid Sudoku digit (1-9)")
        
        # Check that rows have no duplicates
        for row in augmented_board:
            self.assertEqual(len(set(row)), 9, "Row has duplicate digits")
        
        # Check that columns have no duplicates
        for col in range(9):
            column = [augmented_board[row, col] for row in range(9)]
            self.assertEqual(len(set(column)), 9, "Column has duplicate digits")
        
        # Check that 3x3 blocks have no duplicates
        for block_row in range(0, 9, 3):
            for block_col in range(0, 9, 3):
                block = [augmented_board[block_row + i, block_col + j] 
                        for i in range(3) for j in range(3)]
                self.assertEqual(len(set(block)), 9, "Block has duplicate digits")
    
    def test_create_sudoku_puzzle(self):
        """Test create_sudoku_puzzle function."""
        # Generate a solved board
        solved_board = np.array(generate_full_sudoku())
        
        # Create a puzzle with 40% clues
        clue_ratio = 0.4
        puzzle_board, clue_mask = create_sudoku_puzzle(solved_board, clue_ratio=clue_ratio)
        
        # Check shapes
        self.assertEqual(puzzle_board.shape, (9, 9))
        self.assertEqual(clue_mask.shape, (9, 9))
        
        # Check that clue_mask is binary
        np.testing.assert_array_equal(np.unique(clue_mask), np.array([0.0, 1.0]))
        
        # Check that clue ratio is approximately correct
        num_clues = np.count_nonzero(clue_mask)
        expected_clues = int(round(clue_ratio * 81))
        self.assertAlmostEqual(num_clues, expected_clues, delta=1)
        
        # Check that clues match the solved board
        clue_positions = np.where(clue_mask == 1)
        for i, j in zip(*clue_positions):
            self.assertEqual(puzzle_board[i, j], solved_board[i, j])
        
        # Check that non-clues are zeros
        non_clue_positions = np.where(clue_mask == 0)
        for i, j in zip(*non_clue_positions):
            self.assertEqual(puzzle_board[i, j], 0)
    
    def test_count_solutions(self):
        """Test count_solutions function."""
        # Generate a solved board
        solved_board = np.array(generate_full_sudoku())
        
        # A full board should have exactly one solution
        solutions = count_solutions(solved_board)
        self.assertEqual(solutions, 1)
        
        # Create a puzzle with many empty cells
        zero_board = np.zeros((9, 9), dtype=int)
        solutions = count_solutions(zero_board)
        self.assertTrue(solutions > 1)  # Should have many solutions
    
    def test_meets_minimum_distribution(self):
        """Test meets_minimum_distribution function."""
        # Generate a solved board
        solved_board = np.array(generate_full_sudoku())
        
        # A full board should meet any minimum clue requirement
        self.assertTrue(meets_minimum_distribution(solved_board, min_clues=9))
        
        # Create an unbalanced puzzle
        unbalanced_board = np.zeros((9, 9), dtype=int)
        unbalanced_board[0, :] = solved_board[0, :]  # Fill only first row
        
        # Should pass with min_clues=1 for rows
        self.assertTrue(meets_minimum_distribution(unbalanced_board, min_clues=1))
        
        # Should fail with min_clues=2 for rows
        self.assertFalse(meets_minimum_distribution(unbalanced_board, min_clues=2))
    
    def test_dataset_creation(self):
        """Test SudokuDataset creation and basic functionality."""
        # Create a small dataset
        dataset = SudokuDataset(num_samples=5, clue_ratio=0.3)
        
        # Check that the dataset has the correct number of samples
        self.assertEqual(len(dataset), 5)
        
        # Get a sample
        sample = dataset[0]
        
        # Check that the sample has the expected keys
        self.assertIn("solved_board", sample)
        self.assertIn("puzzle_board", sample)
        self.assertIn("clue_mask", sample)
        
        # Check tensor shapes
        self.assertEqual(sample["solved_board"].shape, (9, 9))
        self.assertEqual(sample["puzzle_board"].shape, (9, 9))
        self.assertEqual(sample["clue_mask"].shape, (9, 9))
        
        # Check that solved_board has valid Sudoku digits
        solved_board = sample["solved_board"].numpy()
        for row in solved_board:
            for cell in row:
                self.assertTrue(1 <= cell <= 9, f"Cell value {cell} is not a valid Sudoku digit (1-9)")
        
        # Check that clue_mask is binary
        clue_mask = sample["clue_mask"].numpy()
        np.testing.assert_array_equal(np.unique(clue_mask), np.array([0.0, 1.0]))
        
        # Check that puzzle_board has zeros for non-clue cells
        puzzle_board = sample["puzzle_board"].numpy()
        non_clue_positions = np.where(clue_mask == 0)
        for i, j in zip(*non_clue_positions):
            self.assertEqual(puzzle_board[i, j], 0)
        
        # Check that clue cells match the solved board
        clue_positions = np.where(clue_mask == 1)
        for i, j in zip(*clue_positions):
            self.assertEqual(puzzle_board[i, j], solved_board[i, j])
    
    def test_set_epoch_ratio(self):
        """Test setting the clue ratio for an epoch."""
        # Create a small dataset
        dataset = SudokuDataset(num_samples=5, clue_ratio=0.5)
        
        # Check initial ratio
        self.assertEqual(dataset.clue_ratio, 0.5)
        
        # Set a new ratio
        new_ratio = 0.2
        dataset.set_epoch_ratio(new_ratio)
        
        # Check that the ratio was updated
        self.assertEqual(dataset.clue_ratio, new_ratio)
        
        # Get a sample and check the number of clues
        sample = dataset[0]
        clue_mask = sample["clue_mask"].numpy()
        num_clues = np.count_nonzero(clue_mask)
        expected_clues = int(round(new_ratio * 81))
        self.assertAlmostEqual(num_clues, expected_clues, delta=3)


if __name__ == '__main__':
    unittest.main()