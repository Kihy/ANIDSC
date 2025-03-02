
from abc import abstractmethod
from pathlib import Path

import numpy as np

from ..components.pipeline_component import PipelineComponent
from ..save_mixin.null import NullSaveMixin
from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray
import networkx as nx

class BaseFeatureBuffer(PipelineComponent):
    def __init__(self, buffer_size:int=256, save_features:bool=True, save_meta:bool=True, *args, **kwargs):
        """feature buffer to buffer results in batches to speed up detection. It drops remaining features that does not make a batch

        Args:
            buffer_size (int, optional): number of features to buffer. Defaults to 256.
            save (bool, optional): whether to save the buffered features. Defaults to True.
        """        
        super().__init__(*args, **kwargs)
        self.buffer_size=buffer_size
        self.save_features=save_features
        self.save_meta=save_meta
        
    def setup(self):
        super().setup()
        if self.save_features:

            # setup files
            feature_file = Path(
                f"{self.context['dataset_name']}/{self.context['fe_name']}/features/{self.context['file_name']}.csv"
            )
            feature_file.parent.mkdir(parents=True, exist_ok=True)
            meta_file = Path(
                f"{self.context['dataset_name']}/{self.context['fe_name']}/metadata/{self.context['file_name']}.csv"
            )
            meta_file.parent.mkdir(parents=True, exist_ok=True)

            self.feature_file = open(feature_file, "w")
            self.meta_file = open(meta_file, "w")
            self.feature_file.write(",".join(self.context['feature_names']) + "\n")
            self.meta_file.write(",".join(self.context['meta_names']) + "\n")
            
            self.feature_list=[]
            self.meta_list=[]
            self.size=0
            
            self.parent.context["batch_size"]=self.buffer_size
    
    def process(self, data: Tuple[List[Any], List[Any]])->Union[None, NDArray]:
        """process input data

        Args:
            data (Tuple[List[Any], List[Any]]): the input data, which must be a tuple of feature values and meta_data

        Returns:
            Union[None, NDArray]: returns buffered feature if buffer is full, other wise None
        """        
        feature, meta_data=data
        
        if feature is None:
            return 
        
        self.feature_list.append(feature)
        self.meta_list.append(meta_data)
        
        if len(self.feature_list) >= self.buffer_size:
            return self.save_buffer()
    
    @abstractmethod
    def save_buffer(self)->NDArray:
        pass
    
    def teardown(self):
        # self.save_buffer()
        
        self.meta_file.close()
        self.feature_file.close()
        
        print("feature file saved at", self.feature_file.name)
        print("meta file saved at", self.meta_file.name)
    
    
        