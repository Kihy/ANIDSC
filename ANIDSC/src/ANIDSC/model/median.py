

from ..utils.helper import compare_dicts
from ..save_mixin.pickle import PickleSaveMixin
import numpy as np

from pytdigest import TDigest
from ..converters.decorator import auto_cast_method


class MedianDetector(PickleSaveMixin):
    def __init__(self, ndims, **kwargs):
        """a simple boxplot
        """
        self.ndims=ndims
        self.model=[TDigest() for _ in range(self.ndims)]
        
    
    def get_total_params(self):
        return 0
    
    @auto_cast_method
    def predict_step(self, X: np.ndarray):
        return_score=[]
        for data in X:
            
            score=0
            for x, dim in zip(data, self.model):
                
                if np.isnan(dim.mean):
                    continue 
                    
                # relative tolerance test
                dim_score=np.abs((x-dim.mean)/(dim.inverse_cdf(0.84)-dim.inverse_cdf(0.16)))
                if dim_score>score:
                    score=dim_score

            return_score.append(score)
        return np.array(return_score)

    @auto_cast_method
    def train_step(self, X: np.ndarray):
        
            
        for data in X:

            
            for x, dim in zip(data, self.model):
                dim.update(x)
                
    def to_centroids(self):
        return [i.get_centroids() for i in self.model]

    def of_centroids(self, dim_list):
        return [TDigest.of_centroids(np.array(i)) for i in dim_list]

    def __getstate__(self):
        state_dict=self.__dict__.copy()
        
        state_dict["model"] = self.to_centroids()

        return state_dict

    def __setstate__(self, state):
        state["model"] = self.of_centroids(state["model"])
        self.__dict__.update(state)

    def __eq__(self, other):
        same_class=self.__class__==other.__class__ 
        if not same_class:
            return False
        
        
        
        # Create copies of the __dict__ to avoid modifying the original attributes
        self_attrs = self.__getstate__().copy()
        other_attrs = other.__getstate__().copy()

        return compare_dicts(self_attrs, other_attrs, self.__class__)        
              