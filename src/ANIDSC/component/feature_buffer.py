from abc import abstractmethod
import os
from pathlib import Path

import numpy as np

from .pipeline_component import PipelineComponent

from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray


class OutputWriter(PipelineComponent):
    @property 
    @abstractmethod
    def file_type(self):
        pass
    

    @property
    @abstractmethod
    def folder_name(self):
        pass

    def setup(self):
        # setup files
        dataset_name = self.request_attr("dataset_name")
        file_name = self.request_attr("file_name")  
        fe_name = self.request_attr("fe_name")
        

        self.feature_path = Path(
            f"{dataset_name}/{fe_name}/{self.folder_name}/{file_name}.{self.file_type}"
        )
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)

        self.save_file = open(self.feature_path, "a")

    def teardown(self):
        self.save_file.close()


class BaseFeatureBuffer(OutputWriter):
    
    @property
    def folder_name(self):
        return self._folder_name
    
    def __init__(
        self,
        folder_name,
        buffer_size
    ):
        super().__init__()
        self._folder_name=folder_name
        self.save_file = None
        self.data_list = []
        self._buffer_size=buffer_size


    @property
    def buffer_size(self):
        return self._buffer_size
            

    def process(self, data: Tuple[List[Any], List[Any]]) -> Union[None, NDArray]:
        """process input data

        Args:
            data (Tuple[List[Any], List[Any]]): the input data, which must be a tuple of feature values and meta_data

        Returns:
            Union[None, NDArray]: returns buffered feature if buffer is full, other wise None
        """

        if data is None:
            return

        self.data_list.append(data)

        if len(self.data_list) >= self.buffer_size:
            return self.save_buffer()

    def teardown(self):
        self.save_buffer()
        super().teardown()

    @abstractmethod
    def save_buffer(self):
        pass

    

