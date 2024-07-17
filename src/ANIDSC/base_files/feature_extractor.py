from abc import abstractmethod
from .pipeline import PipelineComponent
from pathlib import Path
import numpy as np 
from scapy.all import Packet
from typing import Dict, Any, List, Tuple, Union
from numpy.typing import NDArray

class FeatureBuffer(PipelineComponent):
    def __init__(self, buffer_size:int=1e4, save_features:bool=True):
        """feature buffer to buffer results in batches to speed up detection. It drops remaining features that does not make a batch

        Args:
            buffer_size (int, optional): number of features to buffer. Defaults to 1e4.
            save (bool, optional): whether to save the buffered features. Defaults to True.
        """        
        super().__init__()
        self.buffer_size=buffer_size
        self.save_features=save_features
        
    def setup(self):
        if self.save_features:
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
            
            self.parent.context["batch_size"]=self.buffer_size
    
    def process(self, data: Tuple[List[Any], List[Any]])->Union[None, NDArray]:
        """process input data

        Args:
            data (Tuple[List[Any], List[Any]]): the input data, which must be a tuple of feature values and meta_data

        Returns:
            Union[None, NDArray]: returns buffered feature if buffer is full, other wise None
        """        
        feature, meta_data=data
        
        self.feature_list.append(feature)
        self.meta_list.append(meta_data)
        
        self.size+=feature.shape[0]
        if self.size >= self.buffer_size:
            return self.save_buffer()
            
    def save_buffer(self)->NDArray:
        """saves buffer

        Returns:
            NDArray: the buffered features as numpy array
        """        
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
    
    def __str__(self):
        return f"FeatureBuffer({self.buffer_size})"
        
        
class BaseTrafficFeatureExtractor(PipelineComponent):
    def __init__(self, offset_time:Union[int, str]="auto", skip:int=0, **kwargs):
        """base interface for feature extractor

        Args:
            offset_time (Union[int, str], optional): time to offset, if set to "auto", automatically calculates offset so that it continues immediately from existing state. Defaults to "auto".
        """        
        super().__init__(component_type="feature_extractors",**kwargs)
        self.skipped=0
        self.processed=0
        self.offset_time=offset_time
        self.skip=skip
  
    def setup(self):
        self.parent.context["feature_names"]=self.get_headers()
        self.parent.context["meta_names"]=self.get_meta_headers()
        self.parent.context["fe_name"]=self.name
        self.parent.context["fe_features"]=len(self.get_headers())
        self.parent.context["output_features"]=len(self.get_headers())
        self.parent.context['skip']=self.skip
        
        if not self.loaded_from_file:
            self.init_state()
        
    @abstractmethod
    def init_state(self):
        """initialize state information
        """        
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
            
            self.processed += feature.shape[0]
            
        return feature, traffic_vector
    
    def teardown(self):
        print(
            f"skipped: {self.skipped} processed: {self.processed+self.skipped} written: {self.processed}"
        )
        super().teardown()
    
