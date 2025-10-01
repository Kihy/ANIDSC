from abc import abstractmethod
import os
from pathlib import Path

import numpy as np

from .pipeline_component import PipelineComponent

from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray






class BaseFeatureBuffer(PipelineComponent):
    
    @property 
    @abstractmethod
    def file_type(self):
        pass
    
    def __init__(
        self,
        folder_name,
    ):
        self.folder_name=folder_name
        self.save_file = None
        self.data_list = []


    @property
    def buffer_size(self):
        return 1024
    
    def setup(self):
        # setup files
        dataset_name = self.request_attr("data_source", "dataset_name")
        file_name = self.request_attr("data_source", "file_name")

        fe_name = self.request_action("feature_extractor", "__str__")
        if fe_name is None:
            fe_name = self.request_attr("data_source", "fe_name")
        

        feature_path = (
            f"{dataset_name}/{fe_name}/{self.folder_name}/{file_name}.{self.file_type}"
        )
        Path(feature_path).parent.mkdir(parents=True, exist_ok=True)

        self.save_file = open(feature_path, "a")

        # write header when file is empty
        headers = self.request_attr("data_source", "headers")
        
        if os.stat(feature_path).st_size == 0:
            self.save_file.write(",".join(headers) + "\n")

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

    @abstractmethod
    def save_buffer(self):
        pass

    

