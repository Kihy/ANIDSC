"""
Type conversion system with auto-registered adapters.

Usage:
    from converters import auto_cast
    import torch
    import numpy as np
    
    @auto_cast
    def process(data: torch.Tensor, mask: np.ndarray):
        return data, mask
    
    # Just importing converters auto-registers all available adapters!
    result = process([[1, 2]], [True, False])
"""

from .base import TypeAdapter, AutoRegisterAdapter
from .registry import ConverterRegistry
from .decorator import auto_cast, auto_cast_method

# Import adapters - this triggers auto-registration
from . import adapters

__version__ = "1.0.0"

__all__ = [
    'TypeAdapter',
    'AutoRegisterAdapter',
    'ConverterRegistry',
    'auto_cast',
    'auto_cast_method',
]