
from abc import ABC, abstractmethod

import inspect

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
        
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)

        # you can store them if you like
        self._params = {k: values[k] for k in args if k != "self"}
      
    @property
    def config_attr(self):
        # save_state={}
        # for i in self.save_attr:
        #     if i=="manifest": #special case for components
        #         value={k: v.to_dict() for k, v in self.__dict__["components"].items()}
        #     else:
        #         value=self.__dict__[i]
        #     save_state[i]=value
            
        return self._params
    
    
    def request_attr(self, component, attr, default=None):
        if self.parent_pipeline is None:
            return default
        else:
            return self.parent_pipeline.get_attr(component, attr, default)
    
    def request_action(self, component, action):
        return self.parent_pipeline.perform_action(component, action)
    
    
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
    
    def to_dict(self):
        comp_dict = {
            "class": self.__class__.__name__,
            "attrs": self.config_attr,
            "file": self.save_path
        }
        return comp_dict

    @property
    @abstractmethod
    def save_path(self):
        pass 
    
    def __getstate__(self):
        state = {}
        for k, v in self.__dict__.items():
            try:
                pickle.dumps(v)  # test if pickleable
                state[k] = v
            except Exception:
                # skip silently
                pass
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return compare_dicts(self.__dict__, other.__dict__, self.__class__)
        
              
