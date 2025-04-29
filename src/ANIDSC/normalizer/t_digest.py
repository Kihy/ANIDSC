from typing import List

from ..save_mixin.pickle import PickleSaveMixin
from .normalizer import BaseOnlineNormalizer
from pytdigest import TDigest
import numpy as np



class LivePercentile(PickleSaveMixin, BaseOnlineNormalizer):
    def __init__(self, p:List[float]=[0.25, 0.5, 0.75],  **kwargs):
        """normalizes input with percentile calculations with tdigest

        Args:
            p (List[float], optional): list of percentiles for extraction, in the order lower percentile, median, upper percentile. Defaults to [0.25, 0.5, 0.75].
        """        
        BaseOnlineNormalizer.__init__(self, **kwargs)
        self.p = p
        self.count = 0
        self.dims=[]
        self.preprocessors=[self.to_numpy]
        
        

    def to_numpy(self,X):
        return np.array(X)
        
        
    def setup(self):
        super().setup()
        self.context["scaler"]=self
        if not self.loaded_from_file:
            self.dims=[TDigest() for _ in range(self.ndim-self.skip)]
    
   
        

    def update(self, X):
        """Adds another datum"""
        
        for i, n in enumerate(X.T):
            self.dims[i].update(n)

        self.count += 1
        
    

    def process(self, X):
        self.current_batch=X
        percentiles = self.quantiles()

        no_scale=X[:,:self.skip]
        scale=X[:,self.skip:]
        
        if percentiles is None:
            percentiles = np.quantile(scale, self.p, axis=0)
            
        scaled_features = (scale - percentiles[1]) / (percentiles[2] - percentiles[0])
        scaled_features = np.nan_to_num(
            scaled_features, nan=0.0, posinf=0.0, neginf=0.0
        )

        return np.hstack((no_scale, scaled_features))

    def reset(self):
        self.dims = [TDigest() for _ in range(self.ndim-self.skip)]
        self.count = 0

    def quantiles(self):
        """Returns a list of tuples of the quantile and its location"""

        if self.ndim == 0 or self.count < self.warmup:
            return None
        percentiles = np.zeros((len(self.p), self.ndim-self.skip))

        for d in range(self.ndim-self.skip):
            percentiles[:, d] = self.dims[d].inverse_cdf(self.p)

        return percentiles

    def to_centroids(self):
        return [i.get_centroids() for i in self.dims]

    def of_centroids(self, dim_list):
        return [TDigest.of_centroids(np.array(i)) for i in dim_list]

    def __getstate__(self):
        state_dict=self.__dict__.copy()
        state_dict['dims']=self.to_centroids()
        
        return state_dict
        
    def __setstate__(self, state):
        state["dims"]=self.of_centroids(state['dims'])
        self.__dict__.update(state)
        
    