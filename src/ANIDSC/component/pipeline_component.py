
from abc import ABC, abstractmethod

from pathlib import Path
import time
from typing import Callable, Dict, Any, List, Union
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
        self.component_name = self.__class__.__name__
        self.component_type=component_type
        
        self.parent:Pipeline = self  # Reference to parent pipeline
        self.preprocessors=[]
        self.postprocessors=[]
        
        self.comparable=True
        
        # used for saving
        self.unpickleable=["parent"]
        
        # used for saving config
        self.save_attr=[]
        
    def get_save_attr(self):
        save_state={}
        for i in self.save_attr: 
            save_state[i]=self.__dict__[i]
            
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
    
    def request_attr(self, component, attr):
        """finds the current component's context by recursively adding parent's context to self if it does not exist

        Returns:
            Dict[str, Any]: the overall context dictionary
        """
        return self.parent.get_attr(component, attr)
        
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
    


    def __or__(self, other:'PipelineComponent')->'Pipeline':
        """attaches pipeline component together

        Args:
            other (PipelineComponent): another pipeline component to attach to

        Returns:
            Pipeline: pipeline
        """        
        return Pipeline([self, other])

    def __str__(self):
        return self.component_name
    
    def get_save_path(self, extension):
        return self.parent.get_save_path_template().format(self.component_type, self.component_name, extension)
    
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
              

class Pipeline(YamlSaveMixin, PipelineComponent):
    def __init__(self, components: Dict[str, PipelineComponent]):
        """A full pipeline that can be extended with |

        Args:
            components (PipelineComponent): the component of pipeline
        """        
        super().__init__(component_type="manifest")
        self.components = components
        self.component_name=str(self)
        self.start_time=None
        
    
    def setup(self):
        for _, component in self.components.items():
            component.parent=self
            component.setup()
    
    def get_attr(self, comp_type, attr):
        return getattr(self.components[comp_type], attr)

    
    
    def process(self, data=None):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.start_time=time.time()
        
        self.setup()
        
        pbar = tqdm()
        
        while True:
            for comp_type, component in self.components.items():
                data = component.preprocess(data)
                data = component.process(data)
                data = component.postprocess(data)
                if data is None: 
                    break
            pbar.update(1)
            if comp_type=="data_source":
                break # end of input
        
        self.save()
            
    def get_save_path_template(self):
        return f"{self.get_attr('data_source','dataset_name')}/{self.get_attr('feature_extractor', 'component_name')}/{{}}/{self.get_attr('data_source','file_name')}/{{}}.{{}}"

            
    def __eq__(self, other: 'Pipeline'):
        same_class=self.__class__==other.__class__ 
        if not same_class:
            return False
        
        return self.components==other.components

    def __str__(self):
        return "->".join([str(component) for _, component in self.components.items()])
    
