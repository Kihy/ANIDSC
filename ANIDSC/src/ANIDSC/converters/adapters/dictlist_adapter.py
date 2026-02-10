"""Adapter for RecordList (List[Dict]) - AUTO-REGISTERED"""
from typing import Any, List, Dict
from ..base import AutoRegisterAdapter
from ..types import RecordList

try:
    import pandas as pd
    import numpy as np
    
    class RecordListAdapter(AutoRegisterAdapter):
        """Handles conversions TO RecordList (List[Dict])"""
        
        @staticmethod
        def target_type() -> type:
            return RecordList
        
        @staticmethod
        def can_convert(value: Any) -> bool:
            return isinstance(value, (pd.DataFrame, list, dict, np.ndarray))
        
        @staticmethod
        def convert(value: Any) -> RecordList:
            if isinstance(value, pd.DataFrame):
                return RecordList(value.to_dict('records'))
            else:
                raise TypeError(f"Cannot convert {type(value)} to RecordList")

except ImportError:
    pass