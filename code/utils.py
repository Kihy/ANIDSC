import json
from pathlib import Path
from io import TextIOWrapper
import numpy as np 
import torch 

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
    return x.numpy()

def to_tensor(x):
    return torch.tensor(x)

def load_dataset_info():
    with open("../../datasets/data_info.json", "r") as f:
        data_info = json.load(f)
    return data_info


def save_dataset_info(data_info):
    with open("../../datasets/data_info.json", "w") as f:
        json.dump(data_info, f, indent=4, cls=JSONEncoder)
