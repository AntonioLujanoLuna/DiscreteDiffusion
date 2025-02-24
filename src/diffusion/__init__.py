"""
Diffusion module for the Discrete Diffusion project.

This module provides functionality for forward and reverse diffusion processes.
"""

from .process import (
    forward_process,
    reverse_diffusion_inference
)

__all__ = [
    'forward_process',
    'reverse_diffusion_inference'
]