import json
from pathlib import Path
from io import TextIOWrapper
import numpy as np 
import torch 
import pickle
import scipy
from pytdigest import TDigest
from abc import ABC, abstractmethod
import pandas as pd

from datetime import datetime, timedelta, time
import pytz

    
class LivePercentile:
    def __init__(self, ndim=None):
        """ Constructs a LiveStream object
        """

        if isinstance(ndim, int):
            self.dims=[TDigest() for _ in range(ndim)]
            self.patience=0
            self.ndim=ndim
        elif isinstance(ndim, list):
            self.dims=self.of_centroids(ndim)
            self.ndim=len(ndim)
            self.patience=10

        else:
            raise ValueError("ndim must be int or list")
        

    def add(self, item):
        """ Adds another datum """
        item=item[:,:self.ndim]
        
        if isinstance(item, torch.Tensor):
            item=item.cpu().numpy()
        
        if self.ndim==1:
            self.dims[0].update(item)
        else:
            for i, n in enumerate(item.T):
                self.dims[i].update(n)
        
        self.patience+=1

    def reset(self):
        
        self.dims=[TDigest() for _ in range(self.ndim)]
        self.patience=0
    
    def quantiles(self, p):
        """ Returns a list of tuples of the quantile and its location """
        
        if self.ndim==0 or self.patience<1:
            return None 
        percentiles=np.zeros((len(p),self.ndim))
        
        for d in range(self.ndim):
            percentiles[:,d]=self.dims[d].inverse_cdf(p)
        
        return torch.tensor(percentiles).float()
    
    def to_centroids(self):
        return [i.get_centroids() for i in self.dims]
    
    def of_centroids(self, dim_list):

        return [TDigest.of_centroids(i) for i in dim_list]
    
    def __getstate__(self):
        state=self.__dict__.copy()
        state["dims"]=self.to_centroids()
        return state 
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dims=self.of_centroids(self.dims)


def calc_quantile(x, p):
    eps=1e-6
    x=np.log(np.array(x)+eps)
    mean=np.mean(x) 
    std=np.std(x)

    quantile=np.exp(mean+np.sqrt(2)*std*scipy.special.erfinv(2*p-1))
    return quantile

def is_stable(x, p=0.95, return_quantile=False):
    if len(x)!=x.maxlen:
        stability=False
        quantile=0.
    else:
        quantile=calc_quantile(x, p)
        stability=np.mean(np.array(x)<quantile)<p
        
    if return_quantile:
        return stability, quantile
    else:
        return stability

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


class LazyInitializer:
    """allows subclass to be lazily initialized. Allowable attributes are stored in allowed and children must implement entry() function

    """    
    def __init__(self, allowed:list[str])->None:
        """initialize 

        Args:
            allowed (list[str]): list of allowed variables
        """        
        self.allowed=allowed
    
    def set_attr(self, **kwargs):
        """sets attributes in kwargs

        Raises:
            ValueError: if key in kwargs is not in allowed
        """        
        for k, v in kwargs.items():
            if k in list(self.allowed):
                setattr(self, k, v)
            else:
                raise ValueError(f"{k} not allowed")
            setattr(self, k, v)
            self.allowed.remove(k)
            
    def start(self, **kwargs): 
        self.set_attr(**kwargs)
        
        if len(self.allowed)>0:
            raise ValueError("Must assign the following variables",",".join(self.allowed))

        self.entry_func()
        
    @abstractmethod
    def entry_func(self):
        pass
           
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


def to_tensor(x):
    return torch.tensor(x)

def load_dataset_info():
    with open("../datasets/data_info.json", "r") as f:
        data_info = json.load(f)
    return data_info




def find_concept_drift_times(dataset_name, fe_name, file_name, timezone, schedule):
    times = pd.read_csv(
        f"../datasets/{dataset_name}/{fe_name}/{file_name}.csv",
        skiprows=lambda x: x % 256 != 0,
    )
    times = times["timestamp"]
    timezone = pytz.timezone(timezone)
    idle = True
    drift_idx = []
    for idx, time in times.items():
        # find time in brisbane, and adjusted time period
        pkt_time = datetime.fromtimestamp(
            float(time), tz=timezone
        )  # -timedelta(hours=17)

        # weekday schedule
        prev_idle = idle

        conditions = schedule[pkt_time.weekday()]

        for c in conditions:
            # print(c[0], pkt_time.time(), c[1])
            if c[0] <= pkt_time.time() <= c[1]:
                idle = False
                break
            else:
                idle = True

        if idle != prev_idle:
            drift_idx.append(idx)
    print(drift_idx)

def save_dataset_info(data_info):
    with open("../datasets/data_info.json", "w") as f:
        json.dump(data_info, f, indent=4, cls=JSONEncoder)
        
def get_node_map(dataset_name, fe_name,file_name):
    try:
        with open(f"../datasets/{dataset_name}/{fe_name}/state/{file_name}.pkl", "rb") as pf:
            state=pickle.load(pf)
            
        return state["node_map"]
    except FileNotFoundError as e:
        print(e)
        return None

