import json
from pathlib import Path
from io import TextIOWrapper
import numpy as np 
import torch 
import pickle
import scipy
from pytdigest import TDigest
import robustats
import matplotlib

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


def to_tensor(x):
    return torch.tensor(x)

def load_dataset_info():
    with open("../../datasets/data_info.json", "r") as f:
        data_info = json.load(f)
    return data_info


def save_dataset_info(data_info):
    with open("../../datasets/data_info.json", "w") as f:
        json.dump(data_info, f, indent=4, cls=JSONEncoder)
        
def get_node_map(dataset_name, fe_name,file_name):
    try:
        with open(f"../../datasets/{dataset_name}/{fe_name}/state/{file_name}.pkl", "rb") as pf:
            state=pickle.load(pf)
            
        return state["node_map"]
    except FileNotFoundError as e:
        print(e)
        return None

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp