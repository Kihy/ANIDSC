from itertools import product
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from pytdigest import TDigest
import torch
from collections import deque
import yaml

import torch_geometric
import pandas as pd
import networkx as nx
from functools import singledispatch



# -------------------------
# Generic entry point
# -------------------------
def compare_dicts(d1, d2, ctx=None):
    if d1.keys() != d2.keys():
        print("different number of keys")
        return False

    for k in d1:
        if not compare(d1[k], d2[k], ctx):
            print(f"different key: {k} for {ctx}")
            print(f"val1 ({type(d1[k])}): {d1[k]}")
            print(f"val2 ({type(d2[k])}): {d2[k]}")
            return False
    return True


# -------------------------
# Generic comparator
# -------------------------
@singledispatch
def compare(a, b, ctx=None):
    return a == b


# -------------------------
# NumPy
# -------------------------
@compare.register(np.ndarray)
def _(a, b, ctx=None):
    return isinstance(b, np.ndarray) and np.array_equal(a, b)


# -------------------------
# Torch
# -------------------------
@compare.register(torch.Tensor)
def _(a, b, ctx=None):
    return isinstance(b, torch.Tensor) and torch.allclose(a, b, equal_nan=True)


@compare.register(torch.nn.Module)
def _(a, b, ctx=None):
    if not isinstance(b, torch.nn.Module):
        return False
    sdA, sdB = a.state_dict(), b.state_dict()
    return all(
        torch.allclose(sdA[k], sdB[k], equal_nan=True)
        for k in sdA
    )


# -------------------------
# pandas
# -------------------------
@compare.register(pd.Series)
def _(a, b, ctx=None):
    return isinstance(b, pd.Series) and a.equals(b)


# -------------------------
# Containers
# -------------------------
@compare.register(list)
@compare.register(tuple)
@compare.register(deque)
def _(a, b, ctx=None):
    if type(a) is not type(b) or len(a) != len(b):
        return False
    return all(compare(x, y, ctx) for x, y in zip(a, b))


@compare.register(dict)
def _(a, b, ctx=None):
    return compare_dicts(a, b, ctx)


# -------------------------
# NetworkX
# -------------------------
@compare.register(nx.Graph)
def _(a, b, ctx=None):
    if not isinstance(b, nx.Graph):
        return False
    return (
        list(a.nodes(data=True)) == list(b.nodes(data=True)) and
        list(a.edges(data=True)) == list(b.edges(data=True))
    )


# -------------------------
# PyG
# -------------------------
@compare.register(torch_geometric.data.Data)
def _(a, b, ctx=None):
    return compare_dicts(a.to_dict(), b.to_dict(), ctx)


def uniqueXT(
    x,
    sorted=True,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    occur_last=False,
    dim=None,
):
    if return_index or (not sorted and dim is not None):
        unique, inverse, counts = torch.unique(
            x, sorted=True, return_inverse=True, return_counts=True, dim=dim
        )
        inv_sorted, inv_argsort = inverse.flatten().sort(stable=True)

        if occur_last and return_index:
            tot_counts = (
                inverse.numel()
                - 1
                - torch.cat((counts.new_zeros(1), counts.flip(dims=[0]).cumsum(dim=0)))[
                    :-1
                ].flip(dims=[0])
            )
        else:
            tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]

        index = inv_argsort[tot_counts]

        if not sorted:
            index, idx_argsort = index.sort()
            unique = (
                unique[idx_argsort]
                if dim is None
                else torch.index_select(unique, dim, idx_argsort)
            )
            if return_inverse:
                idx_tmp = idx_argsort.argsort()
                inverse.flatten().index_put_((inv_argsort,), idx_tmp[inv_sorted])
            if return_counts:
                counts = counts[idx_argsort]

        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_inverse:
            ret += (inverse,)
        if return_counts:
            ret += (counts,)
        return ret if len(ret) > 1 else ret[0]

    else:
        return torch.unique(
            x,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )


def generate_cartesian_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate cartesian product of all array-valued elements in config.
    
    Args:
        config: Dictionary with array and non-array values
        
    Returns:
        List of dictionaries with all combinations
    """
    # Separate array and non-array fields
    array_fields = {}
    static_fields = {}
    
    for key, value in config.items():
        if isinstance(value, list):
            array_fields[key] = value
        else:
            static_fields[key] = value
    
    # If no array fields, return original config
    if not array_fields:
        return [config]
    
    # Generate cartesian product
    keys = list(array_fields.keys())
    values = [array_fields[k] for k in keys]
    
    configs = []
    for combination in product(*values):
        # Create new config with this combination
        new_config = static_fields.copy()
        new_config.update(dict(zip(keys, combination)))
        configs.append(new_config)
    
    return configs


def load_yaml(input_str):
    """
    If input_str points to a file, load YAML from file.
    Otherwise, parse input_str as YAML text.
    """
    
    if input_str.endswith(".yaml") or input_str.endswith(".yml"):
        path = Path(input_str)

        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"File exists but does not contain valid YAML: {input_str}") from e
        else:
            raise FileNotFoundError(f"Input file does not exist: {input_str}")
    else:
        try:
            return yaml.safe_load(input_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Input is not valid YAML text: {input_str}") from e 
        
            
    
def print_dictionary(d, indent=0):
    """Recursively prints a dictionary with indentation."""
    if len(d) == 0:
        print(None)
        return
    for key, value in d.items():
        print("  " * indent + f"{key}: ", end="")
        if isinstance(value, dict):
            print()  # Newline for nested dict
            print_dictionary(value, indent + 1)
        else:
            print(f"{value}")