# from ..models.sklearn_models import *
# from ..datasets.custom_dataset import *

from abc import ABC, abstractmethod
from ..utils import LazyInitializer

class BasePipeline(ABC, LazyInitializer):
    def __init__(self, 
                 allowed,
                 **kwargs):
        
        LazyInitializer.__init__(self, allowed)
        self.set_attr(**kwargs)
        self.entry_func = self.run_pipeline
    
    @abstractmethod
    def setup(self):
        pass
    
    @abstractmethod
    def teardown(self):
        pass
    
    def run_pipeline(self):
        self.setup()
        results = {}
        for step in self.steps:
            func = getattr(self, step)
            results[step] = func()
        self.teardown()
        return results
    
    def metric_names(self):
        return [metric.__name__ for metric in self.metrics]


