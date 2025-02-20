import random
import numpy as np
import torch
from torch.utils.data import Dataset

def generate_full_sudoku():
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

def augment_sudoku(board):
    """
    Applies data augmentation to a solved Sudoku board.
    Augmentations include:
      - A random permutation of the digits.
      - Random row swaps within each 3-row band.
      - Random column swaps within each 3-column stack.
      
    Args:
        board (list of lists): 9x9 solved Sudoku.
        
    Returns:
        new_board (np.ndarray): Augmented 9x9 Sudoku board.
    """
    board_np = np.array(board, dtype=np.int64)
    # Random digit permutation.
    perm = np.random.permutation(np.arange(1, 10))
    digit_map = {i + 1: perm[i] for i in range(9)}
    aug_board = np.vectorize(lambda x: digit_map[x])(board_np)
    
    # Random row swap within each band.
    for band in range(3):
        rows = list(range(band * 3, band * 3 + 3))
        random.shuffle(rows)
        aug_board[band * 3:band * 3 + 3, :] = aug_board[rows, :]
    
    # Random column swap within each stack.
    for stack in range(3):
        cols = list(range(stack * 3, stack * 3 + 3))
        random.shuffle(cols)
        aug_board[:, stack * 3:stack * 3 + 3] = aug_board[:, cols]
    
    return aug_board

class SudokuDataset(Dataset):
    """
    PyTorch Dataset for generating augmented solved Sudoku boards and corresponding clue masks.
    Each sample is a dictionary with:
      - 'solved_board': a torch.LongTensor of shape (9, 9) with values in {1,...,9}.
      - 'clue_mask': a torch.FloatTensor of shape (9, 9) with 1.0 for clue cells.
    """
    def __init__(self, num_samples=1000, clue_ratio=0.3, augment=True):
        self.num_samples = num_samples
        self.clue_ratio = clue_ratio
        self.augment = augment
        self.samples = []
        for _ in range(num_samples):
            board = generate_full_sudoku()
            if self.augment:
                board = augment_sudoku(board)
            # board is a 9x9 numpy array with digits 1-9.
            mask = np.random.rand(9, 9) < clue_ratio
            if not mask.any():
                mask[random.randint(0, 8), random.randint(0, 8)] = True
            mask = mask.astype(np.float32)
            self.samples.append((board, mask))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        board, mask = self.samples[idx]
        solved_board = torch.tensor(board, dtype=torch.long)   # Shape: (9, 9)
        clue_mask = torch.tensor(mask, dtype=torch.float32)      # Shape: (9, 9)
        return {"solved_board": solved_board, "clue_mask": clue_mask}
