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
    An enhanced Plotly-based interactive visualization for Sudoku reverse-diffusion trajectories,
    with clearer Sudoku grid lines and improved annotation styling.

    Args:
        trajectory (list of np.ndarray): 
            List of board states. Each state must be a 2D array of shape (9, 9).
        clue_mask (np.ndarray or None): 
            A binary mask of shape (9, 9) where 1 indicates a clue cell. If provided, 
            those cells are shown with a different background color.
    """

    # ---------------------------
    # 1) Validate and squeeze boards
    # ---------------------------
    def validate_trajectory(trajectory_list):
        for idx, board_arr in enumerate(trajectory_list):
            board_np = np.array(board_arr)
            # If shape is (1, 9, 9), squeeze out the extra dim
            if board_np.ndim == 3 and board_np.shape[0] == 1:
                board_np = board_np.squeeze(0)
                trajectory_list[idx] = board_np
            if board_np.shape != (9, 9):
                raise ValueError(
                    f"Board at index {idx} has shape {board_np.shape}. Expected (9, 9)."
                )

    validate_trajectory(trajectory)

    if clue_mask is not None:
        clue_mask = np.array(clue_mask)
        if clue_mask.shape != (9, 9):
            raise ValueError(
                f"Clue mask has invalid shape {clue_mask.shape}. Expected (9, 9)."
            )

    # ---------------------------
    # 2) Generate shapes for Sudoku grid lines
    # ---------------------------
    # We'll create horizontal and vertical lines from x=0..9, y=0..9.
    # Thicker lines at multiples of 3 (0, 3, 6, 9).
    line_shapes = []
    for i in range(10):
        # Thicker at multiples of 3
        line_width = 4 if (i % 3 == 0) else 1

        # Horizontal line (from x=0 to x=9 at y=i)
        line_shapes.append(
            dict(
                type="line",
                x0=0, x1=9,
                y0=i, y1=i,
                line=dict(color="black", width=line_width),
                layer="below"   # so it doesn't overlap text
            )
        )
        # Vertical line (from y=0 to y=9 at x=i)
        line_shapes.append(
            dict(
                type="line",
                x0=i, x1=i,
                y0=0, y1=9,
                line=dict(color="black", width=line_width),
                layer="below"
            )
        )

    # ---------------------------
    # 3) Prepare frames (one for each board in the trajectory)
    # ---------------------------
    frames = []
    for frame_idx, board in enumerate(trajectory):
        # Build up the list of annotations for this frame
        frame_annotations = []
        for row in range(9):
            for col in range(9):
                val = board[row, col]
                # Display an empty string for 0
                cell_value = "" if val == 0 else str(int(val))

                # If clue_mask given, highlight clues differently
                if clue_mask is not None and clue_mask[row, col] == 1:
                    cell_bg = "lightblue"
                else:
                    cell_bg = "white"

                # Each cell is placed by annotation at x=col, y=row
                # We'll invert y in the final figure by reversing the axis range.
                frame_annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=cell_value,
                        showarrow=False,
                        font=dict(color="black", size=16),
                        align="center",
                        valign="middle",
                        bgcolor=cell_bg,
                        bordercolor="black",
                        borderwidth=1
                    )
                )

        # Create a frame with no "data", just layout updates to the annotations
        frames.append(
            go.Frame(
                data=[],
                name=str(frame_idx),
                layout=go.Layout(annotations=frame_annotations)
            )
        )

    # ---------------------------
    # 4) Create the figure with initial layout
    # ---------------------------
    fig = go.Figure(
        data=[],
        layout=go.Layout(
            title="Sudoku Reverse Diffusion (Enhanced)",
            xaxis=dict(
                visible=False,
                range=[-0.5, 9.5],  # small padding
                showgrid=False
            ),
            yaxis=dict(
                visible=False,
                # We want row=0 at the top, row=8 at the bottom
                range=[9.5, -0.5],
                scaleanchor="x",  # Force square cells
                showgrid=False
            ),
            width=600,
            height=600,
            shapes=line_shapes,  # The grid lines
            margin=dict(l=20, r=20, t=60, b=20),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.1, xanchor="right",
                    y=1.07, yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 700, "redraw": True},
                                         "fromcurrent": True}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None],
                                  {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}]
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )

    # Set initial annotations
    fig.layout.annotations = frames[0].layout.annotations

    # ---------------------------
    # 5) Add a slider for manual stepping
    # ---------------------------
    fig.update_layout(
        sliders=[
            {
                "currentvalue": {"prefix": "Step: "},
                "pad": {"t": 60},
                "len": 0.9,
                "x": 0.1,
                "steps": [
                    {
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate"
                            }
                        ],
                        "label": frame.name,
                        "method": "animate",
                    }
                    for frame in frames
                ],
            }
        ]
    )

    # Finally, show
    fig.show()
