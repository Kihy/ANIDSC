import json
from pathlib import Path
from io import TextIOWrapper
import numpy as np 
import torch 
import pickle

def uniqueXT(x, sorted=True, return_index=False, return_inverse=False, return_counts=False,
             occur_last=False, dim=None):
    if return_index or (not sorted and dim is not None):
        unique, inverse, counts = torch.unique(x, sorted=True,
            return_inverse=True, return_counts=True, dim=dim)
        inv_sorted, inv_argsort = inverse.flatten().sort(stable=True)

        if occur_last and return_index:
            tot_counts = (inverse.numel() - 1 - 
                torch.cat((counts.new_zeros(1),
                counts.flip(dims=[0]).cumsum(dim=0)))[:-1].flip(dims=[0]))
        else:
            tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        
        index = inv_argsort[tot_counts]
        
        if not sorted:
            index, idx_argsort = index.sort()
            unique = (unique[idx_argsort] if dim is None else
                torch.index_select(unique, dim, idx_argsort))
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
        return ret if len(ret)>1 else ret[0]
    
    else:
        return torch.unique(x, sorted=sorted, return_inverse=return_inverse,
            return_counts=return_counts, dim=dim)


class LazyInitializationMixin:
    def lazy_init(self, **kwargs):
        
        for k, v in kwargs.items():
            if k in self.allowed:
                setattr(self, k, v)
            else:
                raise ValueError(f"{k} not allowed")
            setattr(self, k, v)
            self.allowed.remove(k)
            
    def start(self, **kwargs):
        assigned=list(self.allowed)
        for k, v in kwargs.items():
            if k in self.allowed:
                setattr(self, k, v)
            else:
                raise ValueError(f"{k} not allowed")
            assigned.remove(k)
        
        if len(assigned)>0:
            raise ValueError("Must assign the following variables",",".join(assigned))

        return self.entry()

    def __rrshift__(self, other):
        return self.start(**other)



class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, TextIOWrapper):
            return obj.name
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def to_numpy(x):
    return x.detach().cpu().numpy()

def to_tensor(x):
    return torch.tensor(x)

def load_dataset_info():
    with open("../../datasets/data_info.json", "r") as f:
        data_info = json.load(f)
    return data_info


def save_dataset_info(data_info):
    with open("../../datasets/data_info.json", "w") as f:
        json.dump(data_info, f, indent=4, cls=JSONEncoder)
