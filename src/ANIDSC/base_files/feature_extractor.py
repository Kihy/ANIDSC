from abc import abstractmethod
from .pipeline import PipelineComponent
from pathlib import Path
import numpy as np 
from scapy.all import Packet
import pickle
from typing import Dict, Any, List, Union


class FeatureBuffer(PipelineComponent):
    #drops remaining features
    def __init__(self, buffer_size:int=1e4, save:bool=True):
        super().__init__()
        self.buffer_size=buffer_size
        self.save=save
        
    def setup(self):
        if self.save:
            context=self.get_context()
            # setup files
            feature_file = Path(
                f"{context['dataset_name']}/{context['fe_name']}/features/{context['file_name']}.csv"
            )
            feature_file.parent.mkdir(parents=True, exist_ok=True)
            meta_file = Path(
                f"{context['dataset_name']}/{context['fe_name']}/metadata/{context['file_name']}.csv"
            )
            meta_file.parent.mkdir(parents=True, exist_ok=True)

            self.feature_file = open(feature_file, "w")
            self.meta_file = open(meta_file, "w")
            self.feature_file.write(",".join(context['feature_names']) + "\n")
            self.meta_file.write(",".join(context['meta_names']) + "\n")
            
            self.feature_list=[]
            self.meta_list=[]
            self.size=0
    
    def process(self, data):
        feature, meta_data=data
        
        self.feature_list.append(feature)
        self.meta_list.append(meta_data)
        
        self.size+=feature.shape[0]
        if self.size >= self.buffer_size:
            return self.save_buffer()
            
    def save_buffer(self):
        if len(self.feature_list)==0:
            return 
        
        batch_data=np.vstack(self.feature_list)
        
        if self.save:
            np.savetxt(
                    self.feature_file,
                    batch_data,
                    delimiter=",",
                    fmt="%s",
                )
            np.savetxt(
                self.meta_file, np.vstack(self.meta_list), delimiter=",", fmt="%s"
            )
        
        self.feature_list = []
        self.meta_list = []
        self.size=0
        return batch_data
    
    def teardown(self):
        # self.save_buffer()
        
        self.meta_file.close()
        self.feature_file.close()
        
        print("feature file saved at", self.feature_file.name)
        print("meta file saved at", self.meta_file.name)
        
        
        
class BaseTrafficFeatureExtractor(PipelineComponent):
    def __init__(self, load_state=False, offset_time:Union[int, str]="auto"):
        super().__init__()
        self.skipped=0
        self.processed=0
        self.offset_time=offset_time
        self.load_state=load_state
    
    
    def setup(self):
        
        self.parent.context["feature_names"]=self.get_headers()
        self.parent.context["meta_names"]=self.get_meta_headers()
        self.parent.context["fe_name"]=self.name
        self.parent.context["n_features"]=len(self.get_headers())
        self.init_state()
        
    @abstractmethod
    def init_state(self):
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
    
    def process(self, packet:Packet):
        """The main entry point of feature extractor, this function
        should define the process of extracting a single packet
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

        feature = self.update(traffic_vector)
        
        self.processed += feature.shape[0]
        
        return feature, traffic_vector
    
    def teardown(self):
        print(
            f"skipped: {self.skipped} processed: {self.processed+self.skipped} written: {self.processed}"
        )
        self.save(folder="feature_extractors")
    
