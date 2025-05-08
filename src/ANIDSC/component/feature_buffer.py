
from abc import abstractmethod
import os
from pathlib import Path

import numpy as np

from .pipeline_component import PipelineComponent

from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray


class BaseFeatureBuffer(PipelineComponent):
    def __init__(self, buffer_size:int=256, save_features:bool=True, **kwargs):
        """feature buffer to buffer results in batches to speed up detection. It drops remaining features that does not make a batch

        Args:
            buffer_size (int, optional): number of features to buffer. Defaults to 256.
            save (bool, optional): whether to save the buffered features. Defaults to True.
        """        
        super().__init__(**kwargs)

        self.comparable=False 
        
        self.component_type="feature_buffer"
        self.buffer_size=buffer_size
        self.save_features=save_features
        
        
        
        self.feature_file=None 
        self.meta_file=None 
        self.feature_list=[]
        self.meta_list=[]
        
        self.size=0
        
        self.unpickleable.extend(['feature_list','meta_list'])
        self.save_attr.extend(['buffer_size','save_features'])
    
    def setup(self):
        super().setup()
        if self.save_features:
            # setup files
            dataset_name=self.request_attr('data_source','dataset_name')
            fe_name=self.request_action('feature_extractor','__str__')
            file_name=self.request_attr('data_source','file_name')
            
            feature_path =f"{dataset_name}/{fe_name}/features/{file_name}.csv"
            
            Path(feature_path).parent.mkdir(parents=True, exist_ok=True)
            meta_path = f"{dataset_name}/{fe_name}/metadata/{file_name}.csv"
            
            Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

            self.feature_file = open(feature_path, "a")
            self.meta_file = open(meta_path, "a")
            
            
            # write header when file is not empty
            if os.stat(feature_path).st_size==0:            
                self.feature_file.write(",".join(self.request_attr('feature_extractor',"feature_names")) + "\n")
                
            if os.stat(meta_path).st_size==0:            
                self.meta_file.write(",".join(self.request_attr('feature_extractor',"meta_names")) + "\n")
            
            
            
            
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
    
    def save(self):
        
        self.save_buffer()
        
        self.meta_file.close()
        self.feature_file.close()
        
        self.feature_file=self.feature_file.name 
        self.meta_file=self.meta_file.name

    
    
        