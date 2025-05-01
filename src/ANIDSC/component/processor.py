from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Union
from .pipeline_component import PipelineComponent


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
