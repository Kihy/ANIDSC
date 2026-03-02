from abc import abstractmethod
import os
from pathlib import Path

import numpy as np

from .pipeline_component import PipelineComponent

from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray
import fsspec
import subprocess

class OutputWriter(PipelineComponent):
    """Base class for output writers"""
    
    @property 
    @abstractmethod
    def file_type(self):
        pass
    
    @property
    @abstractmethod
    def output_file_name(self):
        pass
    
    
    @property
    def output_path(self):
        feature_path = Path(
            f"runs/{self.dataset_name}/{self.request_attr('run_identifier')}/{self.pipeline_name}/{self.file_name}/{self.output_file_name}.{self.file_type}"
        )
        return feature_path
    
    @property
    def feature_path(self):
        """Actual file path (may be modified by subclasses)"""
        return self.output_path
    
    def _open_file(self):
        """Override in subclasses to customize file opening"""
        return open(self.feature_path, "w")
    
    def _close_file(self):
        """Override in subclasses to customize file closing"""
        self.save_file.close()

    def setup(self):
        # Setup common attributes
        self.dataset_name = self.request_attr("dataset_name")
        self.file_name = self.request_attr("file_name")
        self.pipeline_name = self.request_attr("pipeline_name")
        self.run_identifier = self.request_attr('run_identifier')
        
        # Create directory and open file
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_file = self._open_file()

    def teardown(self):
        self._close_file()


class CompressedOutputWriter(OutputWriter):
    """Output writer with zstd compression"""
    
    @property
    def feature_path(self):
        """Add .zst extension to output path"""
        return self.output_path.with_name(self.output_path.name + ".zst")

    
    def _open_file(self):
        """Open with compression"""
        return fsspec.open(self.feature_path, "wt", compression="zstd").open()
    
    def _close_file(self):
        """Close and validate compressed file"""
        self.save_file.flush()
        self.save_file.close()
        
        # Validate compressed file
        subprocess.run(
            ["zstd", "-t", str(self.feature_path)],
            check=True,
            capture_output=True
        )
        
    

class BaseFeatureBuffer(CompressedOutputWriter):
    

    def __init__(
        self,
        buffer_size
    ):
        super().__init__()
        self.save_file = None
        self.data_list = []
        self._buffer_size=buffer_size

    @property
    def buffer_size(self):
        return self._buffer_size
            

    def setup(self):
        super().setup()

    def process(self, data: List[Any]) -> Union[None, NDArray]:
        """process input data

        Args:
            data (List[Any]): the input data, which must be a tuple of feature values and meta_data

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

    

