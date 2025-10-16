from typing import List

import torch

from ..save_mixin.pickle import PickleSaveMixin
from ..component.normalizer import BaseOnlineNormalizer
from pytdigest import TDigest
import numpy as np


class LivePercentile(PickleSaveMixin, BaseOnlineNormalizer):
    def __init__(self):
        """normalizes input with percentile calculations with tdigest

        Args:
            p (List[float], optional): list of percentiles for extraction, in the order lower percentile, median, upper percentile. Defaults to [0.25, 0.5, 0.75].
        """
        super().__init__()

        self.p = [0.16, 0.5, 0.84]
        self.count = 0
        self.ndim=None
    

    def setup(self):
        if self.ndim is None:
            self.ndim = self.request_attr("output_dim")
            self.dims = [TDigest() for _ in range(self.ndim)]
            
        
    def teardown(self):
        pass 

    def update(self, X):
        """Adds another datum"""

        for i, n in enumerate(X.T):
            self.dims[i].update(n)

        self.count += 1

    def process(self, X):
        
        # convert to numpy
        if isinstance(X, torch.Tensor):
            X=X.detach().cpu().numpy()
        else:
            X=np.array(X)
        
        
        percentiles = self.quantiles()


        if percentiles is None:
            percentiles = np.quantile(X, self.p, axis=0)

        scaled_features = (X - percentiles[1]) / (percentiles[2] - percentiles[0])
        scaled_features = np.nan_to_num(
            scaled_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        #update afterwards
        
        self.update(X)

        return scaled_features

    def reset(self):
        self.dims = [TDigest() for _ in range(self.ndim)]
        self.count = 0

    def quantiles(self):
        """Returns a list of tuples of the quantile and its location"""

        percentiles = np.zeros((len(self.p), self.ndim))

        for d in range(self.ndim ):
            percentiles[:, d] = self.dims[d].inverse_cdf(self.p)

        return percentiles

    def to_centroids(self):
        return [i.get_centroids() for i in self.dims]

    def of_centroids(self, dim_list):
        return [TDigest.of_centroids(np.array(i)) for i in dim_list]

    def __getstate__(self):
        state_dict = super().__getstate__().copy()

        state_dict["dims"] = self.to_centroids()

        return state_dict

    def __setstate__(self, state):
        state["dims"] = self.of_centroids(state["dims"])
        super().__setstate__(state)
