# DiscreteDiffusion

This project implements a discrete diffusion model for solving Sudoku puzzles using PyTorch. The model leverages a diffusion process that gradually denoises a noisy Sudoku grid, while enforcing Sudoku constraints via an auxiliary loss.

## Project Structure

```
DiscreteDiffusion/
├── README.md           # This file, providing an overview and instructions.
├── requirements.txt    # Python package dependencies.
└── src/
    └── main.py         # Main Python module containing data generation, model, training loop, and inference visualization.
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AntonioLujanoLuna/DiscreteDiffusion.git
   cd DiscreteDiffusion
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main module is located in the `src/` directory. To train the model and visualize inference, simply run:

```bash
python src/main.py
```

The script will:
- Generate a dataset of Sudoku puzzles with data augmentation.
- Train the discrete diffusion model with auxiliary constraint loss.
- Perform reverse diffusion inference on a sample puzzle.
- Visualize the evolution of the board using a matplotlib animation.

## Further Improvements

Possible enhancements include:
- Self-conditioning and advanced reverse sampling strategies.
- Additional constraint enforcement or post-processing steps.
- Exploring hybrid architectures that better capture 2D grid structure.

Feel free to fork and contribute!

## License

This project is licensed under the MIT License.
