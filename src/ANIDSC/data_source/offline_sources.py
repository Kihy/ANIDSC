import json
import os
from typing import Any, Dict
import numpy as np
from scapy.all import PcapReader
from ..component.pipeline_source import PipelineSource
from pathlib import Path
import pandas as pd
from ..save_mixin.null import NullSaveMixin

class PacketReader(NullSaveMixin,PipelineSource):
    def __init__(self, **kwargs):
        """reads data from pcap file

        Args:
            dataset_name (str): name of dataset
            file_name (str): name of file
            max_pkts (int, optional): maximum number of packets to read. Defaults to float("inf").
        """
        super().__init__(**kwargs)

        # Try both extensions
        base_path = f"{self.dataset_name}/pcap/{self.file_name}"
        pcap_path = f"{base_path}.pcap"
        pcapng_path = f"{base_path}.pcapng"

        if os.path.exists(pcap_path):
            self.path = pcap_path
        elif os.path.exists(pcapng_path):
            self.path = pcapng_path
        else:
            raise FileNotFoundError(f"Neither {pcap_path} nor {pcapng_path} found.")
        
        self._iter = PcapReader(self.path)
        
    def get_timestamp(self, data):
        return data.time
    
    @property
    def batch_size(self):
        return 1
        
        
        

class CSVReader(NullSaveMixin, PipelineSource):
    def __init__(self, fe_name: str, fe_attrs:Dict[str, Any]={}, **kwargs):
        """reads data from CSV file

        Args:
            fe_cls (str): class of the feature extractor
            fe_name (str): name of feature extractor

        """
        super().__init__(**kwargs)

        # set fe_attrs to self 
        self.fe_name=fe_name
        
        # set fe_attrs to self
        self.fe_attrs=fe_attrs
        
        self.path = f"{self.dataset_name}/{self.fe_name}/features/{self.file_name}.csv"
        self._iter = pd.read_csv(
            self.path, chunksize=self.batch_size, nrows=self.max_records, header=0
        )
    
    def get_timestamp(self, data):
        return data["timestamp"]
    
    def __getattr__(self, name):
        return self.fe_attrs[name]
    
    @property
    def batch_size(self):
        return 1024
    

class JsonGraphReader(NullSaveMixin, PipelineSource):
    def __init__(self, fe_name: str, fe_attrs:Dict[str, Any]={}, **kwargs):
        """reads data from JSON file

        Args:
            fe_cls (str): class of the feature extractor
            fe_name (str): name of feature extractor

        """
        super().__init__(**kwargs)

        # set fe_attrs to self 
        self.fe_name=fe_name
 
        
        # set fe_attrs to self
        self.fe_attrs=fe_attrs

        
        self.path = f"{self.dataset_name}/{self.fe_name}/features/{self.file_name}.ndjson"
        
        self._iter=self.get_json_obj()
        
    def get_json_obj(self):
        with open(self.path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                yield obj
    
    def get_timestamp(self, data):
        return data["graph"]["time_stamp"]
        
    @property
    def batch_size(self):
        return 1