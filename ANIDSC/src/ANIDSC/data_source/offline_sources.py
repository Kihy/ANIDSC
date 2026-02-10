import json
import os
from typing import Any, Dict
import networkx as nx
import fsspec
from ..save_mixin.pickle import PickleSaveMixin
from scapy.all import PcapReader
from ..component.pipeline_source import PipelineSource
import pandas as pd


class PacketReader(PickleSaveMixin,PipelineSource):
    def __init__(self, **kwargs):
        """reads data from pcap file
        """
        super().__init__(**kwargs)

    def setup(self):
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
    
    @property
    def output_dim(self):
        return None
        
        
        

class CSVReader(PickleSaveMixin, PipelineSource):
    def __init__(self, fe_name: str, **kwargs):
        """reads data from CSV file

        Args:
            fe_cls (str): class of the feature extractor
            fe_name (str): name of feature extractor

        """
        super().__init__(**kwargs)
        self.fe_name=fe_name 

    
    def setup(self):
        self.path = f"{self.dataset_name}/features/{self.fe_name}/{self.file_name}.csv.zst"
        
        self._file = fsspec.open(self.path, "rt", compression="zstd").open()

        self._iter = pd.read_csv(
            self._file, chunksize=self.batch_size, nrows=self.max_records, header=0
        )
    
    def get_timestamp(self, data):
        if "timestamp" in data.columns:            
            return data["timestamp"]
        else:
            return None
    
    @property
    def batch_size(self):
        return 1024
    
    @property
    def output_dim(self):
        return len(self.iter._engine.names)
    
    def teardown(self):
        self._file.close()

class JsonGraphReader(PickleSaveMixin, PipelineSource):
    def __init__(self, fe_name: str, **kwargs):
        """reads data from JSON file

        Args:
            fe_cls (str): class of the feature extractor
            fe_name (str): name of feature extractor

        """
        super().__init__(**kwargs)

        # set fe_attrs to self 
        self.fe_name=fe_name

    def setup(self):
        self.path = f"{self.dataset_name}/features/{self.fe_name}/{self.file_name}.ndjson.zst"
        self._file = fsspec.open(self.path, "rt", compression="zstd").open()
        
        self._iter=self.get_json_obj()
        
    def get_json_obj(self):
        for line in self._file:
            yield [nx.readwrite.json_graph.node_link_graph(json.loads(line))]

    def teardown(self):
        self._file.close()
        
    
    def get_timestamp(self, data):
        return [g.graph["time_stamp"] for g in data]
    

    @property
    def batch_size(self):
        return 1
    
    @property
    def output_dim(self):
        return None