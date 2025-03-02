
from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Callable, Dict, Any, List, Union
import json

from ..utils.helper import compare_dicts

class PipelineComponent(ABC):
    def __init__(self, component_type:str="", component_name=None, call_back: Callable[[Any], Any] = None):
        """A component in the pipeline that can be chained with | 

        Args:
            component_type (str, optional): type of component for saving. If '', the component is stateless and will not be saved. Defaults to "".
            call_back (Callable[[Any], Any], optional): A callable function after process each batch of input. Defaults to None.
        """        
        
        if component_name is None:
            self.component_name = self.__class__.__name__
        else:
            self.component_name=component_name
            
        self.component_type=component_type
        self.call_back = call_back
        self.parent = None  # Reference to parent pipeline
        self.loaded_from_file=False
        self.suffix=[]
        self.preprocessors=[]
        self.postprocessors=[]
        self.ignore_attrs=["context", "parent", "call_back","suffix", "loaded_from_file","preprocessors","postprocessors"]# Remove the two excluded attributes from both copies
    
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
    
    def get_context(self)->Dict[str, Any]:
        """finds the current component's context by recursively adding parent's context to self if it does not exist

        Returns:
            Dict[str, Any]: the overall context dictionary
        """
        return self.parent.context
    

    def setup(self):
        self.context=self.get_context()
        
        

    @abstractmethod
    def process(self, data):
        pass
 
    @abstractmethod
    def save(self):
        pass 
    
    
    @classmethod
    @abstractmethod
    def load(cls, folder, dataset_name, fe_name, file_name, name, suffix=''):
        pass 
    
    def teardown(self):
        """saves the pipeline
        """        
        if self.component_type!="":
            self.save()

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
    
    def __eq__(self, other):
        same_class=self.__class__==other.__class__ 
        # Create copies of the __dict__ to avoid modifying the original attributes
        self_attrs = self.__dict__.copy()
        other_attrs = other.__dict__.copy()

        for i in self.ignore_attrs:            
            self_attrs.pop(i, None)       # Ignore KeyError if attribute is missing
            other_attrs.pop(i, None)

        diff_key=compare_dicts(self_attrs, other_attrs)        
        if  diff_key != True:
            print(f"different {diff_key}")
            return False
        else:
            return same_class            

class Pipeline:
    def __init__(self, components: List[PipelineComponent]):
        """A full pipeline that can be extended with |

        Args:
            components (PipelineComponent): the component of pipeline
        """        
        super().__init__()
        self.components = components
        

    def set_context(self, context):
        self.context=context
    
    def setup(self):
        for component in self.components:
            component.parent=self
            component.setup()
    
    def process(self, data):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.context["start_time"]=time.time()
        
        for component in self.components:
            data=component.preprocess(data)
            data = component.process(data)
            data = component.postprocess(data)
            if data is None: # break if buffer is none
                break 
        
        return data

    def teardown(self):
        """iteratively calls the teardown function of each pipeline
        """        
        for component in self.components:
            component.teardown()
            
        # save context
        context_file = Path(
                f"{self.context['dataset_name']}/{self.context['fe_name']}/contexts/{self.context['file_name']}.json"
            )
        
        #remove unnecessary stuff
        self.context.pop("scaler",None)
        
        context_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.context, open(context_file, "w"))

    def __or__(self, other: Union[PipelineComponent, 'Pipeline'])->'Pipeline':
        """extends the pipeline

        Args:
            other (PipelineComponent): if other is a pipeline, the two pipelines are merged, otherwise it is extended

        Returns:
            Pipeline: new pipeline
        """        
        if isinstance(other, Pipeline):
            return Pipeline(self.components + other.components)
        elif isinstance(other, PipelineComponent):
            return Pipeline(self.components + [other])
        else:
            raise ValueError("Unknown pipe type")

    

    def __str__(self):
        return "-".join([str(component) for component in self.components])
