import numpy as np 
from pytdigest import TDigest
import torch
from collections import deque

def compare_dicts(dict1, dict2):
    # Check if keys are the same
    if dict1.keys() != dict2.keys():
        return False

    # Compare values for each key
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        # Handle NumPy arrays
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not (isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray)):
                return key  # One is array, the other isn't
            if not np.array_equal(val1, val2):
                return key

        # Handle nested dictionaries recursively
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dicts(val1, val2):
                return key

        # Handle lists/tuples (if they contain arrays)
        elif isinstance(val1, (list, tuple, deque)) and isinstance(val2, (list, tuple, deque)):
            if len(val1) != len(val2):
                return key
            for v1, v2 in zip(val1, val2):
                if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                    if not (np.array_equal(v1, v2)):
                        return key
                else:
                    if str(v1)!=str(v2):
                        return key
                

        # Default comparison for non-array types
        elif isinstance(val1, torch.nn.Module) and isinstance(val2, torch.nn.Module):
            if str(v1)!=str(v2):
                return key
        else:
            if val1 != val2:
                return key

    return True