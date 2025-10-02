from abc import abstractmethod
from .pipeline_component import PipelineComponent

import copy 

class BaseOnlineNormalizer(PipelineComponent):
        
    @abstractmethod
    def update(self, X):
        pass 

    @abstractmethod
    def reset(self):
        pass 