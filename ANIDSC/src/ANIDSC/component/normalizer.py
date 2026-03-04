from abc import abstractmethod

from ..converters.decorator import auto_cast_method
from .pipeline_component import PipelineComponent
import numpy as np 
import copy 

class BaseOnlineNormalizer(PipelineComponent):
    
    def __init__(self, scale_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.scale_idx=scale_idx
        self.batch_processed=0
    
    @abstractmethod
    def update(self, X):
        pass 
    
    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def reset(self):
        pass 
    
    @auto_cast_method
    def process(self, X:np.ndarray):
        unscaled = X[:, :self.scale_idx]
        to_scale = X[:, self.scale_idx:]
        
        # check scaled shape is equal to self.ndim 
        if to_scale.shape[1] != self.ndim:
            raise ValueError(f"Expected input with {self.ndim} features to scale, but got {to_scale.shape[1]}")
        
        
        scaled = self.transform(to_scale)
        
        result = np.concatenate([unscaled, scaled], axis=1)
        self.batch_processed+=len(X)
        self.update(to_scale)
        return result