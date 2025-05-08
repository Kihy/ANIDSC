from abc import abstractmethod
from .pipeline_component import PipelineComponent

import copy 

class BaseOnlineNormalizer(PipelineComponent):
    def __init__(self, warmup:int=10, **kwargs):
        """base normalizer to normalize datastream online

        Args:
            warmup (int, optional): warm up period before the normlizer outputs value. Defaults to 10.
            ndim (int, optional): number of input dimensions. Defaults to 0.
            
        """        
        super().__init__(component_type="scaler",**kwargs)
        self.warmup=warmup
        self.current_batch=None
        

        
    def setup(self):
        super().setup()

        ndim=self.request_attr("data_source","ndim")
        
        self.ndim=ndim
        self.skip=self.request_attr("data_source", "skip")

        
    @abstractmethod
    def update(self, X):
        pass 
    
    def update_current(self):
        self.update(self.current_batch)

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