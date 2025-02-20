import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import plotly.graph_objects as go

def validate_trajectory(trajectory):
    """
    Validates that each board in the trajectory is a 2D array with shape (9, 9).
    If a board has an extra singleton dimension (i.e., shape (1, 9, 9)), it is squeezed to (9, 9).

    Args:
        trajectory (list): List of numpy arrays representing board states.
    
    Raises:
        ValueError: If a board does not have the expected shape after squeezing.
    """
    for idx, board in enumerate(trajectory):
        board = np.array(board)
        # If board is of shape (1, 9, 9), squeeze it.
        if board.ndim == 3 and board.shape[0] == 1:
            trajectory[idx] = board.squeeze(0)
        elif board.ndim != 2 or board.shape != (9, 9):
            raise ValueError(
                f"Board at index {idx} has invalid shape {board.shape}. Expected (9, 9)."
            )


def display_trajectory(trajectory):
    """
    Visualizes the evolution of the Sudoku board during reverse diffusion.
    
    Args:
        trajectory (list of np.ndarray): List of board states (each of shape (9,9)).
    """
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    def board_to_text(board):
        # If board has shape (1, 9, 9), remove the extra dimension.
        if board.ndim == 3 and board.shape[0] == 1:
            board = board.squeeze(0)
        # Convert board numbers to strings; show an empty string for token 0.
        return [[("" if cell == 0 else str(cell)) for cell in row] for row in board]

    table = ax.table(cellText=board_to_text(trajectory[0]),
                     loc='center', cellLoc='center')
    plt.title("Reverse Diffusion Inference")
    
    def update(frame):
        ax.clear()
        ax.axis('tight')
        ax.axis('off')
        board = trajectory[frame]
        board_text = board_to_text(board)
        ax.table(cellText=board_text, loc='center', cellLoc='center')
        plt.title(f"Step {frame}/{len(trajectory)-1}")
    
    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=500, repeat=False)
    plt.show()

def display_trajectory_interactive(trajectory, clue_mask=None):
    """
    Visualizes the evolution of the Sudoku board during reverse diffusion
    with an interactive Plotly animation. Clue cells are color-coded.
    
    Args:
        trajectory (list of np.ndarray): List of board states. Each board must be of shape (9, 9).
        clue_mask (np.ndarray or None): A binary mask of shape (9, 9) indicating clue cells.
                                        If provided, cells with clues are highlighted in 'lightblue',
                                        while others use 'lightgreen'.
    """
    # Validate board shapes.
    validate_trajectory(trajectory)
    
    # If clue_mask is provided, validate its shape.
    if clue_mask is not None:
        clue_mask = np.array(clue_mask)
        if clue_mask.ndim != 2 or clue_mask.shape != (9, 9):
            raise ValueError(
                f"Clue mask has invalid shape {clue_mask.shape}. Expected (9, 9)."
            )
    
    frames = []
    num_frames = len(trajectory)
    
    # Generate frames for each board state.
    for idx, board in enumerate(trajectory):
        board = np.array(board)
        frame_annotations = []
        for i in range(9):
            for j in range(9):
                # Display an empty string for a 0 (noise) token.
                cell_value = "" if board[i, j] == 0 else str(int(board[i, j]))
                # Use color-coding if clue_mask is provided.
                if clue_mask is not None:
                    color = "lightblue" if clue_mask[i, j] else "lightgreen"
                else:
                    color = "lightgreen"
                frame_annotations.append(dict(
                    x=j, y=i, text=cell_value, showarrow=False,
                    font=dict(color="black", size=16),
                    align="center",
                    bgcolor=color,
                    bordercolor="black"
                ))
        frames.append(go.Frame(data=[], layout=dict(annotations=frame_annotations), name=str(idx)))
    
    # Create the initial figure using the first frame's annotations.
    fig = go.Figure(
        data=[],
        layout=go.Layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, autorange="reversed"),
            title="Sudoku Reverse Diffusion Inference",
            width=500,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        ),
        frames=frames
    )
    
    # Set initial frame annotations.
    fig.layout.annotations = frames[0].layout.annotations
    
    # Add a slider to navigate through frames.
    fig.update_layout(
        sliders=[{
            "steps": [
                {
                    "args": [[frame.name],
                             {"frame": {"duration": 500, "redraw": True},
                              "mode": "immediate"}],
                    "label": frame.name,
                    "method": "animate"
                } for frame in frames
            ],
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "x": 0.1,
            "len": 0.9
        }]
    )
    
    fig.show()
