from abc import abstractmethod
from typing import Any, Dict

from .pipeline_component import PipelineComponent


class SplitterComponent(PipelineComponent):

    def __init__(self, pipeline: PipelineComponent, **kwargs):
        """A component that splits the input data into different outputs and attaches each output to separate pipelines.

        Args:
            pipeline (PipelineComponent): the base pipeline to attach
        """
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.pipelines = {}
        self.context = {}

    @abstractmethod
    def split_function(self, data):
        pass

    def process(self, data) -> Dict[str, Any]:
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
