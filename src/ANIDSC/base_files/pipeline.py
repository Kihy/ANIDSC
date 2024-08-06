from abc import ABC, abstractmethod
import time
from typing import Callable, Dict, Any, Union
from ..base_files.save_mixin import PickleSaveMixin

class PipelineComponent(ABC, PickleSaveMixin):

    def __init__(self, component_type:str="", call_back: Callable[[Any], Any] = None):
        """A component in the pipeline that can be chained with | 

        Args:
            component_type (str, optional): type of component for saving. If '', the component is stateless and will not be saved. Defaults to "".
            call_back (Callable[[Any], Any], optional): A callable function after process each batch of input. Defaults to None.
        """        
        self.call_back = call_back
        self.name = self.__class__.__name__
        self.component_type=component_type
        self.parent = None  # Reference to parent component
        self.loaded_from_file=False
        self.save=True

    def get_context(self)->Dict[str, Any]:
        """finds the current component's context by recursively adding parent's context to self if it does not exist

        Returns:
            Dict[str, Any]: the overall context dictionary
        """        
        parent=self.parent
        context={}
        while parent is not None:
            context.update({k:v for k,v in parent.context.items() if k not in context.keys()})
            parent=parent.parent
        return context
    
    def set_context(self, context: Dict[str, Any]):
        """sets the current context

        Args:
            context (Dict[str, Any]): context to set
        """
        self.context = context
    
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def process(self, data):
        pass
 
    def teardown(self):
        """saves the pipeline
        """        
        if self.save and self.component_type!="":
            context=self.get_context()
            self.save_pickle(self.component_type, suffix=context.get('protocol',''))

    def __or__(self, other:'PipelineComponent')->'Pipeline':
        """attaches pipeline component together

        Args:
            other (PipelineComponent): another pipeline component to attach to

        Returns:
            Pipeline: pipeline
        """        
        return Pipeline([self, other])

    def __str__(self):
        return self.name


class Pipeline(PipelineComponent):
    def __init__(self, components: PipelineComponent):
        """A full pipeline that can be extended with |

        Args:
            components (PipelineComponent): the component of pipeline
        """        
        super().__init__()
        self.components = components
        self.context = {}

    def setup(self):
        for component in self.components:
            component.parent=self
            component.setup()
    
    def set_save(self, save:bool):
        for component in self.components:
            component.save=save
        
    
    def process(self, data):
        """sequentially process data over each component

        Args:
            data (_type_): the input data

        Returns:
            _type_: output data
        """
        self.context["start_time"]=time.time()
        for component in self.components:
            data = component.process(data)
            if data is None:
                break
        return data

    

    def teardown(self):
        """iteratively calls the teardown function of each pipeline
        """        
        for component in self.components:
            component.teardown()

    def __or__(self, other: PipelineComponent)->'Pipeline':
        """extends the pipeline

        Args:
            other (PipelineComponent): if other is a pipeline, the two pipelines are merged, otherwise it is extended

        Returns:
            Pipeline: new pipeline
        """        
        if isinstance(other, Pipeline):
            return Pipeline(self.components + other.components)
        else:
            return Pipeline(self.components + [other])

    def __str__(self):
        return "-".join([str(component) for component in self.components])


class PipelineSource(ABC):
    """The data source of pipeline. This is attached to pipeline with >> """

    @abstractmethod
    def start(self, data):
        pass

    def __rshift__(self, other:Pipeline):
        """attaches itself with a pipeline

        Args:
            other (Pipeline): the pipeline to feed data to
        """        
        self.call_back = other.process
        self.on_end = other.teardown
        self.on_start = other.setup
        self.context['pipeline_name']=str(other)
        other.set_context(self.context)
        
        


class Processor(PipelineComponent):
    def __init__(self, process_func:Callable[[Any], Any]):
        """processor component that applies process_func to input data

        Args:
            process_func (Callable[[Any], Any]): arbitrary function
        """        
        self.process_func = process_func

    def process(self, data):
        """processes input

        Args:
            data (_type_): the input data

        Returns:
            _type_: the output data
        """        
        return self.process_func(data)

    def setup(self):
        pass

    def teardown(self):
        pass
    
class SplitterComponent(PipelineComponent):
    
    def __init__(self, pipeline: PipelineComponent, **kwargs):
        """A component that splits the input data into different outputs and attaches each output to separate pipelines.


        Args:
            pipeline (PipelineComponent): the base pipeline to attach
        """        
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.pipelines={}
        self.context={}

    

    @abstractmethod
    def split_function(self, data):
        pass 
    
    def process(self, data)->Dict[str, Any]:
        """splits the data into different partitions with split_function

        Args:
            data (_type_): the input data

        Returns:
            Dict[str, Any]: dictionary with partitioned data
        """        
        split_data = self.split_function(data)
        results = {}
        for key, data in split_data.items():
            results[key] = self.pipelines[key].process(data)
        return results

    # def teardown(self):
    #     self.save_pickle()

    def __str__(self):
        return f"SplitterComponent({self.pipeline})"

