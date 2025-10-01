from abc import abstractmethod
from .pipeline_component import PipelineComponent

import copy 

class BaseOnlineNormalizer:
    def __init__(self, **kwargs):
        """base normalizer to normalize datastream online

        Args:
            warmup (int, optional): warm up period before the normlizer outputs value. Defaults to 10.
            ndim (int, optional): number of input dimensions. Defaults to 0.
            
        """        
        pass 
    
    @abstractmethod
    def process(self, X):
        pass
        
    @abstractmethod
    def update(self, X):
        pass 

    @abstractmethod
    def reset(self):
        pass 
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result