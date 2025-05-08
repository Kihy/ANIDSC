
from abc import ABC, abstractmethod

from pathlib import Path
import time
from typing import Callable, Dict, Any, List, Union

from ANIDSC.save_mixin.null import NullSaveMixin
from ANIDSC.save_mixin.pickle import PickleSaveMixin
from ANIDSC.save_mixin.torch import TorchSaveMixin
from ANIDSC.save_mixin.yaml import YamlSaveMixin
from tqdm import tqdm
import yaml
import importlib
from ..utils.helper import compare_dicts

class PipelineComponent(ABC):
    def __init__(self, component_type:str=""):
        """A component in the pipeline that can be chained with | 

        Args:
            component_type (str, optional): type of component for saving. If '', the component is stateless and will not be saved. Defaults to "".
           
        """                

        self.component_type=component_type
        
        self.parent_pipeline = None  # Reference to parent pipeline
        self.preprocessors=[]
        self.postprocessors=[]
        
        self.comparable=True
        
        # used for saving
        self.unpickleable=["parent_pipeline"]
        
        # used for saving config
        self.save_attr=[]
        
    def get_save_attr(self):
        save_state={}
        for i in self.save_attr:
            if i=="manifest": #special case for components
                value={k: v.to_dict() for k, v in self.__dict__["components"].items()}
            else:
                value=self.__dict__[i]
            save_state[i]=value
            
        return save_state
    
    def preprocess(self,X):
        """preprocesses the input with preprocessor

        Args:
            X (_type_): input data

        Returns:
            _type_: preprocessed X
        """
        if len(self.preprocessors) > 0:
            for p in self.preprocessors:
                X = p(X)
        return X
    
    def postprocess(self,X):
        """preprocesses the input with preprocessor

        Args:
            X (_type_): input data

        Returns:
            _type_: preprocessed X
        """
        if len(self.postprocessors) > 0:
            for p in self.postprocessors:
                X = p(X)
        return X
    
    def request_attr(self, component, attr, default=None):
        """finds the current component's context by recursively adding parent's context to self if it does not exist

        Returns:
            Dict[str, Any]: the overall context dictionary
        """
        return self.parent_pipeline.get_attr(component, attr, default)
    
    def request_action(self, component, action):
        return self.parent_pipeline.perform_action(component, action)
    
    def on_load(self):
        pass
    
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
            "attrs": self.get_save_attr(),
            "file": self.get_save_path()
        }
        return comp_dict

    
    def get_save_path(self):
        if isinstance(self, NullSaveMixin):
            return False
        elif isinstance(self, YamlSaveMixin):
            return self.get_save_path_template().format(self.component_type, str(self), "yaml")
        elif isinstance(self, PickleSaveMixin):
            return self.parent_pipeline.get_save_path_template().format(self.component_type, str(self), "pkl")
        elif isinstance(self, TorchSaveMixin):
            return self.parent_pipeline.get_save_path_template().format(self.component_type, str(self), "pth")
    
    def __getstate__(self):
        state = self.__dict__.copy()
        for i in self.unpickleable:            
            state.pop(i,None)
        return state
    
    def __eq__(self, other: 'PipelineComponent'):
        same_class=self.__class__==other.__class__ 
        if not same_class:
            return False
        
        if not self.comparable:
            return True
        # Create copies of the __dict__ to avoid modifying the original attributes
        self_attrs = self.__getstate__().copy()
        other_attrs = other.__getstate__().copy()

        return compare_dicts(self_attrs, other_attrs)        
              
