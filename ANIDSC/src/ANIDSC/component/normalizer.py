from abc import abstractmethod
from .pipeline_component import PipelineComponent

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
    
    def process(self, X):
        if self.batch_processed<self.warmup:
            #update during warmup
            scaled=None
        else:
            scaled=self.process(X)
        self.batch_processed+=1
        self.update(X)
        return scaled