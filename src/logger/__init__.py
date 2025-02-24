"""
Logging module for the Discrete Diffusion project.

This module provides unified logging functionality with optional TensorBoard integration.
"""

from .logger import (
    Logger,
    get_logger
)

__all__ = [
    'Logger',
    'get_logger'
]