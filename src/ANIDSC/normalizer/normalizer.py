from abc import abstractmethod
from ..components.pipeline_component import PipelineComponent

import copy 

class BaseOnlineNormalizer(PipelineComponent):
    def __init__(self, warmup:int=10, ndim:int=0, **kwargs):
        """base normalizer to normalize datastream online

        Args:
            warmup (int, optional): warm up period before the normlizer outputs value. Defaults to 10.
            ndim (int, optional): number of input dimensions. Defaults to 0.
            skip (int, optional): number of input dimensions to skip scaling. Defaults to 0.
        """        
        super().__init__(component_type="scalers",**kwargs)
        self.warmup=warmup
        self.ndim=ndim 
        self.current_batch=None

        
    def setup(self):
        super().setup()
        self.ndim=self.context['fe_features']
        self.skip=self.context['skip']

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