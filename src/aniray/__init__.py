"""
aniray: A scientific computing package with JAX and Ray integration.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core.base import BaseCompute
from .core.config import Config

# Algorithm imports
from .algorithms.numerical import solve_linear_system

# Utility imports
from .utils.validation import validate_array
from .utils.logging import get_logger

__all__ = [
    "BaseCompute",
    "Config", 
    "solve_linear_system",
    "validate_array",
    "get_logger",
]