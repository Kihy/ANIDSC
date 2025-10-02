from abc import abstractmethod

from ..save_mixin.pickle import PickleSaveMixin

from ..save_mixin.null import NullSaveMixin
from .pipeline_component import PipelineComponent
from pathlib import Path
import numpy as np
from scapy.all import Packet
from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray
import networkx as nx
import json




class BaseMetaExtractor(PipelineComponent):
    def __init__(self):
        """base interface for feature extractor

        """
        super().__init__()
        self.skipped = 0
        self.processed = 0

        self.last_timestamp = None
        self.result_folder_name="metadata"
        
    

    @abstractmethod
    def get_meta_vector(self, packet):
        """extracts meta vectors from the raw packet,
        returns the extracted traffic vector.

        Args:
            packet (scapy packet): input packet
        """
        pass
    
    @property
    @abstractmethod
    def headers(self):
        """returns the names of the meta data"""
        pass
    
    

    def process(self, packet: Packet, peek=False) -> Tuple[List[float], List[Any]]:
        """The main entry point of feature extractor, this function
        should define the process of extracting a single packet

        Returns:
            Tuple[List[float],List[Any]]: tuple of extracted features and metadata
        """

        meta_vector = self.get_meta_vector(packet)

        if meta_vector is None:
            self.skipped += 1
            return None
        
        self.processed+=1

        self.last_timestamp = meta_vector["timestamp"]
        return meta_vector


    
            




class BaseFeatureExtractor(PipelineComponent):
    def __init__(self,**kwargs):
        """base interface for feature extractor

        """
        super().__init__( **kwargs)
        self.result_folder_name="features"


    @property 
    def fe_name(self):
        return self.name 
    
    @property
    @abstractmethod
    def headers(self):
        """returns the names of the features"""
        pass

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
    


    def process(self, meta_vector) -> Tuple[List[float], List[Any]]:
        features=self.update(meta_vector)
        return features
            
