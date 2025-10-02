
from abc import ABC, abstractmethod

import inspect

import io
import pickle


from ..utils.helper import compare_dicts




class PipelineComponent(ABC):
    @property
    def component_type(self):
        module = self.__module__  # "pkg.A.Class"
        parts = module.split(".")
        return parts[1] if len(parts) > 1 else None
    
    def __init__(self):
        """A component in the pipeline that can be chained with | 

        Args:
            component_type (str, optional): type of component for saving. If '', the component is stateless and will not be saved. Defaults to "".
           
        """                
        self.parent_pipeline = None  # Reference to parent pipeline
        
        frame = inspect.currentframe().f_back  # the caller (child class __init__)
        arg_info = inspect.getargvalues(frame)
        # capture args (excluding "self")
        self._params = {k: arg_info.locals[k] for k in arg_info.args if k != "self"}
      
    @property
    def config_attr(self):
        
        return self._params
    
    
    def request_attr(self, attr, default=None):
        if self.parent_pipeline is not None:    
            return self.parent_pipeline.get_attr(self.index, attr, default)
        if self.component_type=="pipeline":
            return self.get_attr(0, attr, default)
        return default 
    
    @abstractmethod
    def setup(self):
        pass
    
    
    @abstractmethod
    def process(self, data):
        pass
 
    @abstractmethod
    def save(self):
        pass 
    
    @classmethod
    @abstractmethod
    def load(cls, path):
        pass 
    
    def __str__(self):
        return self.__class__.__name__
    
    @property
    def name(self):
        return self.__class__.__name__

    def to_dict(self):
        comp_dict = {
            "type":self.component_type,
            "class": self.__class__.__name__,
            "attrs": self.config_attr,
            "file": str(self.save_path)
        }
        return comp_dict

    @property
    @abstractmethod
    def save_path(self):
        pass 
    
    def __getstate__(self):
        state = {}
        for k, v in self.__dict__.items():
            if k == "parent_pipeline":
                continue

            try:
                pickle.dumps(v)  # test if pickleable
                state[k] = v
            except Exception:
                print(f"{k}:{v} of {self.__class__.__name__} is not pickleable. If it is created during setup, its fine")
                continue
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return compare_dicts(self.__getstate__(), other.__getstate__(), self.__class__)
        
              
