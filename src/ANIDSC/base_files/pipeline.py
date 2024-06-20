from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Union
from ..base_files.save_mixin import PickleSaveMixin

class PipelineComponent(ABC, PickleSaveMixin):
    """A component in the pipeline that can be chained with |"""

    def __init__(self, component_type:str="", call_back: Callable[[Any], Any] = None):
        self.call_back = call_back
        # context = {}
        self.name = self.__class__.__name__
        self.component_type=component_type
        self.parent = None  # Reference to parent component

    def get_context(self):
        parent=self.parent
        context={}
        while parent is not None:
            
            context.update({k:v for k,v in parent.context.items() if k not in context.keys()})
            parent=parent.parent
        return context
    
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def process(self, data):
        pass

 
    def teardown(self):
        if self.component_type!="":
            context=self.get_context()
            self.save_pickle(self.component_type, suffix=context.get('protocol',''))

    def __or__(self, other):
        return Pipeline([self, other])

    def __str__(self):
        return self.name


class Pipeline(PipelineComponent):
    """A full pipeline that can be extended with |"""

    def __init__(self, components: PipelineComponent):
        super().__init__()
        self.components = components
        self.context = {}

    def setup(self):
        for component in self.components:
            component.parent=self
            component.setup()
            
    def process(self, data):
        for component in self.components:
            data = component.process(data)
            if data is None:
                break
        return data

    def set_context(self, context: Dict[str, Any]):
        self.context=context

    def teardown(self):
        for component in self.components:
            component.teardown()

    def __or__(self, other: PipelineComponent):
        # other.set_context(context)
        if isinstance(other, Pipeline):
            return Pipeline(self.components + other.components)
        else:
            return Pipeline(self.components + [other])

    def __str__(self):
        return "-".join([str(component) for component in self.components])


class PipelineSource(ABC):
    """The source of pipeline. This is attached to pipeline with >>"""

    @abstractmethod
    def start(self, data):
        pass

    def __rshift__(self, other:Pipeline):
        self.call_back = other.process
        self.on_end = other.teardown
        self.on_start = other.setup
        self.context['pipeline_name']=str(other)
        other.set_context(self.context)
        
        return self


class Processor(PipelineComponent):
    def __init__(self, process_func:Callable[[Any], Any]):
        self.process_func = process_func

    def process(self, data):
        return self.process_func(data)

    def setup(self):
        pass

    def teardown(self):
        pass
    
class SplitterComponent(PipelineComponent):
    """A component that splits the input data into different outputs and attaches each output to separate pipelines."""

    def __init__(self, pipeline: PipelineComponent, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.pipelines={}
        self.context={}

    def set_context(self, context: Dict[str, Any]):
        self.context=context

    @abstractmethod
    def split_function(self, data):
        pass 
    
    def process(self, data):
        split_data = self.split_function(data)
        results = {}
        for key, data in split_data.items():
            results[key] = self.pipelines[key].process(data)
        return results

    # def teardown(self):
    #     self.save_pickle()

    def __str__(self):
        return f"SplitterComponent({self.pipeline})"

