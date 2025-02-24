# Discrete Diffusion for Sudoku

This repository implements various discrete diffusion models for solving Sudoku puzzles. The project explores different approaches to the reverse diffusion process, including standard diffusion, Diffusion Decision Models (DDM), and Diffusion-of-Thought (DoT).

## Project Structure

```
discrete-diffusion/
├── src/                  # Source code
│   ├── __init__.py
│   ├── config.py         # Configuration system
│   ├── dataset.py        # Sudoku dataset generation
│   ├── model.py          # Model architectures
│   ├── diffusion.py      # Diffusion process
│   ├── loss.py           # Loss functions
│   ├── training.py       # Training loops
│   ├── standard_train.py # Standard diffusion training
│   ├── ddm_train.py      # DDM training
│   ├── dot_train.py      # DoT training
│   ├── standard_inference.py # Standard inference
│   ├── ddm_inference.py  # DDM inference
│   ├── dot_inference.py  # DoT inference
│   ├── utils.py          # Utility functions
│   ├── checkpoint.py     # Checkpoint management
│   ├── logger.py         # Logging system
│   └── visualization.py  # Visualization tools
├── tests/                # Unit tests
├── configs/              # Configuration files
├── logs/                 # Log files
├── checkpoints/          # Model checkpoints
└── README.md             # This file
```

## Features

- **Discrete Diffusion Models**: Implements diffusion models for discrete data (Sudoku puzzles)
- **Multiple Training Paradigms**:
  - Standard diffusion training
  - Diffusion Decision Model (DDM) with evidence accumulation
  - Diffusion-of-Thought (DoT) with multi-trajectory consistency
- **Advanced Architectures**:
  - Improved Sudoku Denoiser with transformer-based architecture
  - Hybrid Sudoku Denoiser with convolutional + transformer architecture
- **Curriculum Learning**: Gradually increases puzzle difficulty during training
- **Visualization**: Interactive visualization of the reverse diffusion process
- **Flexible Configuration**: Dataclass-based configuration system
- **Logging and Checkpointing**: Robust tracking and model saving

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/discrete-diffusion.git
cd discrete-diffusion
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

Train a model using the default configuration:

```bash
python src/main.py --name my_experiment
```

Specify a different training mode:

```bash
python src/main.py --mode DDM --name ddm_experiment
```

Use a specific model architecture:

```bash
python src/main.py --model Hybrid --name hybrid_experiment
```

Resume training from a checkpoint:

```bash
python src/main.py --resume checkpoints/my_experiment/checkpoint_epoch005.pt
```

### Using a Configuration File

Create a custom configuration file:

```bash
# Create a default config file
python -c "from src.config import get_default_config; config = get_default_config(); config.save('my_config.json')"
```

Edit the configuration file and use it for training:

```bash
python src/main.py --config my_config.json
```

### Running Tests

```bash
python -m unittest discover tests
```

## Advanced Features

### Learned Noise Schedule

The system includes a learned noise schedule that can adapt to the specific characteristics of Sudoku puzzles.

### Curriculum Learning

Training starts with easier puzzles (more clues) and gradually progresses to harder puzzles (fewer clues).

### Evidence Accumulation (DDM)

In the DDM approach, evidence is accumulated over multiple denoising steps, allowing for more robust decisions about cell values.

### Multi-trajectory Consistency (DoT)

The DoT approach runs multiple reverse diffusion trajectories and encourages consistency between them, leading to more robust solutions.

## Visualization

The project includes visualization tools for the reverse diffusion process:

```python
from src.visualization import display_trajectory_interactive

display_trajectory_interactive(trajectory, clue_mask)
```

## Citation

If you use this code for your research, please cite:

```
@software{discrete-diffusion,
  author = Antonio Lujano Luna,
  title = {Discrete Diffusion for Sudoku},
  year = {2025},
  url = {https://github.com/AntonioLujanoLuna/discrete-diffusion}
}
```

## License

MIT