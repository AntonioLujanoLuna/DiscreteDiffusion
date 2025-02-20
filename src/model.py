import torch
import torch.nn as nn
import math

class SudokuDenoiser(nn.Module):
    """
    Denoiser network for the discrete diffusion model on Sudoku.
    Uses 2D positional embeddings, token embeddings, time conditioning, and a transformer encoder.
    """
    def __init__(self, num_tokens=10, hidden_dim=128, num_layers=6, nhead=8, dropout=0.1):
        """
        Args:
            num_tokens (int): Total tokens (0 for noise; 1-9 for Sudoku digits).
            hidden_dim (int): Embedding dimension.
            num_layers (int): Number of transformer layers.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(SudokuDenoiser, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Token embedding.
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)
        # Learnable 2D positional embedding for the 9x9 grid.
        self.position_embedding = nn.Parameter(torch.randn(9, 9, hidden_dim))
        # Time embedding.
        self.time_embed = nn.Linear(1, hidden_dim)
        
        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to logits over tokens.
        self.output_layer = nn.Linear(hidden_dim, num_tokens)
    
    def forward(self, x, t, clue_mask):
        """
        Args:
            x (torch.LongTensor): Noisy board of shape (batch_size, 9, 9) with tokens in {0,...,9}.
            t (torch.FloatTensor): Timestep tensor of shape (batch_size, 1).
            clue_mask (torch.FloatTensor): Clue mask of shape (batch_size, 9, 9).
            
        Returns:
            logits (torch.FloatTensor): Logits of shape (batch_size, 9, 9, num_tokens).
        """
        batch_size = x.size(0)
        # Embed tokens and add 2D positional embeddings.
        x_emb = self.token_embedding(x)             # (B, 9, 9, hidden_dim)
        x_emb = x_emb + self.position_embedding       # (B, 9, 9, hidden_dim)
        
        # Flatten grid to sequence for the transformer: (B, 81, hidden_dim)
        x_emb = x_emb.view(batch_size, -1, self.hidden_dim)
        
        # Add time embedding.
        t_emb = self.time_embed(t)                    # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1)                    # (B, 1, hidden_dim)
        x_emb = x_emb + t_emb
        
        # With batch_first=True, no need to transpose.
        x_emb = self.transformer(x_emb)               # (B, 81, hidden_dim)
        
        logits = self.output_layer(x_emb)             # (B, 81, num_tokens)
        logits = logits.view(batch_size, 9, 9, self.num_tokens)
        
        # Force clue cells to match their known token by overriding logits.
        fixed_logits = torch.full_like(logits, -1e10)
        fixed_logits.scatter_(3, x.unsqueeze(-1), 1e10)
        clue_mask_expanded = clue_mask.unsqueeze(-1)  # (B, 9, 9, 1)
        logits = clue_mask_expanded * fixed_logits + (1 - clue_mask_expanded) * logits
        
        return logits


class SinusoidalTimeEmbedding(nn.Module):
    """
    Computes sinusoidal time embeddings similar to the positional encodings used in Transformers.
    
    Given a time tensor t of shape (B, 1), produces an embedding of shape (B, hidden_dim).
    """
    def __init__(self, hidden_dim):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t):
        """
        Args:
            t (torch.FloatTensor): Tensor of shape (B, 1) representing time steps.
            
        Returns:
            torch.FloatTensor: Sinusoidal embeddings of shape (B, hidden_dim).
        """
        half_dim = self.hidden_dim // 2
        # Compute a scaling factor for the frequencies.
        emb_scale = math.log(10000) / (half_dim - 1)
        # Create the frequency tensor.
        freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb_scale)
        # Multiply time tensor with frequencies.
        emb = t * freq  # (B, half_dim)
        # Concatenate sine and cosine embeddings.
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # If hidden_dim is odd, pad with zeros.
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=-1)
        return emb

class ImprovedSudokuDenoiser(nn.Module):
    """
    Improved Denoiser Network for Discrete Diffusion on Sudoku.
    
    This network incorporates:
      - A token embedding layer.
      - 2D sinusoidal positional embeddings computed for a 9x9 grid.
      - A sinusoidal time embedding module.
      - A transformer encoder for sequence processing.
      - An output projection that is post-processed to enforce the clues.
    """
    def __init__(self, num_tokens=10, hidden_dim=128, num_layers=8, nhead=8, dropout=0.1):
        """
        Args:
            num_tokens (int): Total number of tokens (0 for noise; 1-9 for Sudoku digits).
            hidden_dim (int): Dimensionality of token and time embeddings.
            num_layers (int): Number of transformer layers.
            nhead (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(ImprovedSudokuDenoiser, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Token embedding.
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)
        
        # Register fixed 2D sinusoidal positional embeddings for a 9x9 grid.
        self.register_buffer("pos_embedding", self.create_2d_sinusoidal_embeddings(9, 9, hidden_dim))
        
        # Time embedding using the sinusoidal method.
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        
        # Transformer encoder (with batch_first=True for easier integration).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layer mapping transformer features to token logits.
        self.output_layer = nn.Linear(hidden_dim, num_tokens)

    def create_2d_sinusoidal_embeddings(self, height, width, dim):
        """
        Creates fixed 2D sinusoidal positional embeddings for a grid.
        
        Args:
            height (int): Height of the grid (number of rows).
            width (int): Width of the grid (number of columns).
            dim (int): Embedding dimensionality.
        
        Returns:
            torch.FloatTensor: Positional embeddings of shape (height, width, dim).
        """
        pe = torch.zeros(height, width, dim)
        for i in range(height):
            for j in range(width):
                for k in range(0, dim, 2):
                    div_term = math.exp((k * -math.log(10000.0)) / dim)
                    pe[i, j, k] = math.sin(i * div_term) + math.sin(j * div_term)
                    if k + 1 < dim:
                        pe[i, j, k + 1] = math.cos(i * div_term) + math.cos(j * div_term)
        return pe

    def forward(self, x, t, clue_mask):
        """
        Forward pass for the improved denoiser.
        
        Args:
            x (torch.LongTensor): Noisy board of shape (B, 9, 9) with token values in {0,...,9}.
            t (torch.FloatTensor): Timestep tensor of shape (B, 1).
            clue_mask (torch.FloatTensor): Binary mask of shape (B, 9, 9) where 1 indicates a clue.
        
        Returns:
            torch.FloatTensor: Logits of shape (B, 9, 9, num_tokens).
        """
        batch_size = x.size(0)
        # Embed the tokens.
        x_emb = self.token_embedding(x)  # Shape: (B, 9, 9, hidden_dim)
        # Add fixed 2D positional embeddings.
        x_emb = x_emb + self.pos_embedding  # Broadcasting over batch.
        
        # Flatten the 9x9 grid to a sequence of length 81.
        x_emb = x_emb.view(batch_size, -1, self.hidden_dim)  # (B, 81, hidden_dim)
        
        # Compute the time embedding and add to token embeddings.
        t_emb = self.time_embedding(t)  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (B, 1, hidden_dim)
        x_emb = x_emb + t_emb
        
        # Process the sequence using the transformer encoder.
        x_emb = self.transformer(x_emb)  # (B, 81, hidden_dim)
        
        # Project transformer features to token logits.
        logits = self.output_layer(x_emb)  # (B, 81, num_tokens)
        logits = logits.view(batch_size, 9, 9, self.num_tokens)
        
        # Enforce that clue cells are fixed by overriding their logits.
        fixed_logits = torch.full_like(logits, -1e10)
        fixed_logits.scatter_(3, x.unsqueeze(-1), 1e10)
        clue_mask_expanded = clue_mask.unsqueeze(-1)  # (B, 9, 9, 1)
        logits = clue_mask_expanded * fixed_logits + (1 - clue_mask_expanded) * logits
        
        return logits

