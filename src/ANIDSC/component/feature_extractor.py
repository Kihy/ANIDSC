from abc import abstractmethod

from ..save_mixin.null import NullSaveMixin
from .pipeline_component import PipelineComponent
from pathlib import Path
import numpy as np 
from scapy.all import Packet
from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray
import networkx as nx 
import json 


        
class BaseTrafficFeatureExtractor(PipelineComponent):
    def __init__(self, skip:int=0, offset_time:Union[int, str]="auto", **kwargs):
        """base interface for feature extractor

        Args:
            offset_time (Union[int, str], optional): time to offset, if set to "auto", automatically calculates offset so that it continues immediately from existing state. Defaults to "auto".
        """        
        super().__init__(component_type="feature_extractor",**kwargs)
        self.skipped=0
        self.processed=0
        self.offset_time=offset_time
        self.skip=skip
        
        self.save_attr.extend(['skip','offset_time'])
        
  
    def setup(self):
        super().setup()
        self.feature_names=self.get_headers()
        self.meta_names=self.get_meta_headers()
        self.output_features=len(self.get_headers())
        
    
    @abstractmethod
    def update(self, traffic_vector):
        """Updates the feature extractor with traffic_vector, and returns
        the features

        Args:
            traffic_vector (array): traffic vector extracted from the packets
        """
        pass

    @abstractmethod
    def peek(self, traffic_vectors):
        """applies fake update to the feature extractor, does not actually
        update the state of feature extractor. Not required but used for LM attack.
        returns a list of features corresonding to the traffic vectors

        Args:
            traffic_vectors (list of array): list of traffic vectors to be updated
        """
        pass

    @abstractmethod
    def get_traffic_vector(self, packet):
        """extracts traffic vectors from the raw packet,
        returns the extracted traffic vector.

        Args:
            packet (scapy packet): input packet
        """
        pass          

    @abstractmethod
    def get_headers(self):
        """returns the names of the features"""
        pass

    @abstractmethod
    def get_meta_headers(self):
        """returns the names of the traffic vectors"""
        pass
    
    def process(self, packet:Packet, peek=False)->Tuple[List[float],List[Any]]:
        """The main entry point of feature extractor, this function
        should define the process of extracting a single packet

        Returns:
            Tuple[List[float],List[Any]]: tuple of extracted features and metadata
        """        
            
        traffic_vector = self.get_traffic_vector(packet)
            
        if traffic_vector is None:
            self.skipped += 1
            return None 
        
        if self.offset_time=="auto":
            if self.state.last_timestamp is None:
                self.offset_time=0
            else:
                self.offset_time=traffic_vector["timestamp"]-self.state.last_timestamp
        
        traffic_vector["timestamp"] -= self.offset_time

        if peek:
            feature=self.peek(traffic_vector)
        else:
            feature = self.update(traffic_vector)
            if feature is None: 
                return feature, traffic_vector
            
            self.processed += 1
            
        return feature, traffic_vector
    

    
