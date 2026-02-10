"""Adapter for PyTorch tensors - AUTO-REGISTERED"""
from typing import Any
from ..base import AutoRegisterAdapter

try:
    import torch
    import numpy as np
    
    class TensorAdapter(AutoRegisterAdapter):
        """Handles all conversions TO torch.Tensor"""
        
        @staticmethod
        def target_type() -> type:
            return torch.Tensor
        
        @staticmethod
        def can_convert(value: Any) -> bool:
            return isinstance(value, (
                list, tuple, np.ndarray, torch.Tensor, int, float
            )) or hasattr(value, '__array__')
        
        @staticmethod
        def convert(value: Any) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value
            elif isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            elif isinstance(value, (list, tuple)):
                return torch.tensor(value)
            elif hasattr(value, '__array__'):
                return torch.tensor(np.array(value))
            else:
                return torch.tensor(value)

except ImportError:
    # PyTorch not available, skip this adapter
    pass