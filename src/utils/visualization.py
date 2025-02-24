"""
Visualization utilities for the Discrete Diffusion project.

This module provides functions for visualizing Sudoku boards and trajectories.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import plotly.graph_objects as go
import json
from typing import List, Optional, Union, Dict, Any, Tuple

from .common import validate_trajectory

def display_sudoku_board(
    board: np.ndarray, 
    title: str = "Sudoku Board",
    clue_mask: Optional[np.ndarray] = None,
    highlight_cells: Optional[List[Tuple[int, int]]] = None,
    highlight_color: str = "red",
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Display a Sudoku board with optional highlighting.
    
    Args:
        board (np.ndarray): The Sudoku board as a 9x9 numpy array.
        title (str): Title for the plot.
        clue_mask (np.ndarray, optional): Binary mask where 1 indicates clue cells.
        highlight_cells (List[Tuple[int, int]], optional): List of (row, col) cells to highlight.
        highlight_color (str): Color to use for highlighting cells.
        cmap (str): Colormap for cell background.
        
    Returns:
        plt.Figure: The matplotlib figure containing the visualization.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title(title)
    
    # Create a colored array for the background
    color_array = np.zeros_like(board, dtype=float)
    if clue_mask is not None:
        color_array = clue_mask.astype(float) * 0.5
    
    # Create the heatmap
    cmap = plt.cm.get_cmap(cmap)
    cmap.set_bad('white')
    im = ax.imshow(color_array, cmap=cmap, alpha=0.3)
    
    # Add grid lines
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(-.5, 9, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
    ax.set_xticks(np.arange(0, 9, 3))
    ax.set_yticks(np.arange(0, 9, 3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add the numbers
    for i in range(9):
        for j in range(9):
            val = board[i, j]
            if val != 0:
                ax.text(j, i, str(int(val)), ha='center', va='center', fontsize=12)
    
    # Highlight specific cells if requested
    if highlight_cells:
        for i, j in highlight_cells:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                edgecolor=highlight_color, linewidth=3)
            ax.add_patch(rect)
    
    # Final adjustments
    plt.tight_layout()
    return fig

def export_trajectory_json(trajectory: List[np.ndarray], filename: str = "trajectory.json") -> None:
    """
    Exports a trajectory to a JSON file. Each board in the trajectory is converted to a list of lists.
    
    Args:
        trajectory (List[np.ndarray]): List of board states (each a numpy array).
        filename (str): Output filename for the JSON.
    """
    # Convert each board in the trajectory to a list if it's a NumPy array
    trajectory_converted = [
        board.tolist() if isinstance(board, np.ndarray) else board
        for board in trajectory
    ]
    
    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(trajectory_converted, f)

def display_trajectory(trajectory: List[np.ndarray], interval: int = 500) -> None:
    """
    Create an animation to display a trajectory of Sudoku boards.
    
    Args:
        trajectory (List[np.ndarray]): List of board states.
        interval (int): Time in milliseconds between frames.
    """
    # Validate and normalize the trajectory
    trajectory = validate_trajectory(trajectory)
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.axis('tight')
    ax.axis('off')
    
    def board_to_text(board: np.ndarray) -> List[List[str]]:
        """Convert board numbers to strings; show an empty string for token 0."""
        return [[("" if cell == 0 else str(int(cell))) for cell in row] for row in board]
    
    # Initialize with the first board
    table = ax.table(cellText=board_to_text(trajectory[0]), loc='center', cellLoc='center')
    plt.title("Reverse Diffusion Inference")
    
    # Update function for animation
    def update(frame: int) -> None:
        ax.clear()
        ax.axis('tight')
        ax.axis('off')
        board = trajectory[frame]
        board_text = board_to_text(board)
        ax.table(cellText=board_text, loc='center', cellLoc='center')
        plt.title(f"Step {frame}/{len(trajectory)-1}")
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=interval, repeat=False)
    plt.close()  # To prevent duplicate showing
    
    return ani

def display_trajectory_interactive(trajectory: List[np.ndarray], clue_mask: Optional[np.ndarray] = None) -> None:
    """
    Create an interactive Plotly visualization for a trajectory of Sudoku boards.
    
    Args:
        trajectory (List[np.ndarray]): List of board states.
        clue_mask (np.ndarray, optional): Binary mask where 1 indicates clue cells.
    """
    # Validate trajectory
    trajectory = validate_trajectory(trajectory)
    if clue_mask is not None:
        clue_mask = np.array(clue_mask)
        if clue_mask.shape != (9, 9):
            raise ValueError(f"Clue mask has invalid shape {clue_mask.shape}. Expected (9, 9).")
    
    # Create grid lines
    line_shapes = []
    for i in range(10):
        # Thicker at multiples of 3
        line_width = 4 if (i % 3 == 0) else 1
        
        # Horizontal line
        line_shapes.append(dict(
            type="line", x0=0, x1=9, y0=i, y1=i,
            line=dict(color="black", width=line_width), layer="below"
        ))
        
        # Vertical line
        line_shapes.append(dict(
            type="line", x0=i, x1=i, y0=0, y1=9,
            line=dict(color="black", width=line_width), layer="below"
        ))
    
    # Prepare frames
    frames = []
    for frame_idx, board in enumerate(trajectory):
        # Build annotations
        frame_annotations = []
        for row in range(9):
            for col in range(9):
                val = board[row, col]
                # Display an empty string for 0
                cell_value = "" if val == 0 else str(int(val))
                
                # Handle cell background color
                if clue_mask is not None and clue_mask[row, col] == 1:
                    cell_bg = "lightblue"
                else:
                    cell_bg = "white"
                
                # Add annotation
                frame_annotations.append(dict(
                    x=col, y=row, text=cell_value, showarrow=False,
                    font=dict(color="black", size=16),
                    align="center", valign="middle",
                    bgcolor=cell_bg, bordercolor="black", borderwidth=1
                ))
        
        # Create frame
        frames.append(go.Frame(
            data=[], name=str(frame_idx),
            layout=go.Layout(annotations=frame_annotations)
        ))
    
    # Create figure
    fig = go.Figure(
        data=[],
        layout=go.Layout(
            title="Sudoku Reverse Diffusion",
            xaxis=dict(visible=False, range=[-0.5, 9.5], showgrid=False),
            yaxis=dict(visible=False, range=[9.5, -0.5], scaleanchor="x", showgrid=False),
            width=600, height=600, shapes=line_shapes,
            margin=dict(l=20, r=20, t=60, b=20),
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.1, xanchor="right", y=1.07, yanchor="top",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 700, "redraw": True},
                                     "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True},
                                       "mode": "immediate", "transition": {"duration": 0}}])
                ]
            )]
        ),
        frames=frames
    )
    
    # Add slider
    fig.update_layout(
        sliders=[{
            "currentvalue": {"prefix": "Step: "},
            "pad": {"t": 60}, "len": 0.9, "x": 0.1,
            "steps": [
                {
                    "args": [[frame.name], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate"}],
                    "label": frame.name, "method": "animate"
                }
                for frame in frames
            ]
        }]
    )
    
    # Set initial annotations
    fig.layout.annotations = frames[0].layout.annotations
    
    # Show the figure
    fig.show()