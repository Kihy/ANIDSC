from abc import abstractmethod

from ..converters.decorator import auto_cast_method
from .pipeline_component import PipelineComponent
import numpy as np 
import copy 

class BaseOnlineNormalizer(PipelineComponent):
    
    def __init__(self, warmup=1000, **kwargs):
        super().__init__(**kwargs)
        self.warmup=warmup
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
        if self.batch_processed<self.warmup:
            #update during warmup
            scaled=None
        else:
            scaled=self.transform(X)
        self.batch_processed+=len(X)
        self.update(X)
        return scaled