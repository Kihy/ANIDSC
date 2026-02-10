"""Adapter for NumPy arrays - AUTO-REGISTERED"""
from typing import Any
from ..base import AutoRegisterAdapter

try:
    import numpy as np
    
    class NumpyAdapter(AutoRegisterAdapter):
        """Handles all conversions TO np.ndarray"""
        
        @staticmethod
        def target_type() -> type:
            return np.ndarray
        
        @staticmethod
        def can_convert(value: Any) -> bool:
            return isinstance(value, (
                list, tuple, np.ndarray, int, float
            )) or hasattr(value, '__iter__') or hasattr(value, '__array__')
        
        @staticmethod
        def convert(value: Any) -> np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            elif isinstance(value, (list, tuple)):
                return np.array(value)
            else:
                try:
                    import torch
                    if isinstance(value, torch.Tensor):
                        return value.cpu().detach().numpy()
                except ImportError:
                    pass
                
                if hasattr(value, '__array__'):
                    return np.array(value)
                elif hasattr(value, '__iter__'):
                    return np.array(list(value))
                else:
                    return np.array(value)

except ImportError:
    # NumPy not available, skip this adapter
    pass