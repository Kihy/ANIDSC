import numpy as np
from pytdigest import TDigest
import torch
from collections import deque

import torch_geometric
import networkx as nx


def compare_dicts(dict1, dict2, comp_class=None):
    # Check if keys are the same
    if dict1.keys() != dict2.keys():
        print("different number of keys")
        return False

    is_same = True
    # Compare values for each key
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        # Handle NumPy arrays
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not (isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray)):
                is_same = False
                break
            if not np.array_equal(val1, val2):
                is_same = False
                break

        # Handle nested dictionaries recursively
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dicts(val1, val2, comp_class):
                is_same = False
                break

        # Handle lists/tuples (if they contain arrays)
        elif isinstance(val1, (list, tuple, deque)) and isinstance(
            val2, (list, tuple, deque)
        ):
            if len(val1) != len(val2):
                is_same = False
                break
            for v1, v2 in zip(val1, val2):
                if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                    if not (np.array_equal(v1, v2)):
                        is_same = False
                        val1 = v1
                        val2 = v2
                        break
                elif callable(v1):
                    if v1.__qualname__ != v2.__qualname__:
                        is_same = False
                        val1 = v1
                        val2 = v2
                        break
                elif str(v1) != str(v2):
                    is_same = False
                    val1 = v1
                    val2 = v2
                    break

        # Default comparison for non-array types
        elif isinstance(val1, torch.nn.Module):
            sdA = val1.state_dict()
            sdB = val2.state_dict()
            for i in sdA:
                if not torch.isclose(sdA[i], sdB[i], equal_nan=True).all():
                    is_same = False
                    val1 = sdA[i]
                    val2 = sdB[i]
                    break
            if not is_same:
                break
        elif isinstance(val1, nx.Graph):
            if not (
                list(val1.nodes(data=True)) == list(val2.nodes(data=True))
                and list(val1.edges(data=True)) == list(val2.edges(data=True))
            ):
                
                is_same = False
                break

        elif isinstance(val1, torch_geometric.data.Data):
            compare_dicts(val1.to_dict(), val2.to_dict(), comp_class)

        elif isinstance(val1, torch.Tensor):
            return torch.allclose(val1, val2)

        elif val1 != val2:
            is_same = False
            break

    if not is_same:
        print(f"different {key} for {comp_class}")
        print(f"val1: {type(val1)} {val1}")
        print(f"val2: {type(val1)} {val2}")

    return is_same


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
