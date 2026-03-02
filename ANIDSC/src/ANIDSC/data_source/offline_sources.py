from abc import abstractmethod
import json
import os
from typing import Any, Dict, Iterator
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
        base_path = f"datasets/{self.dataset_name}/{self.file_name}"
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
        
        
class PipelineReader(PickleSaveMixin, PipelineSource):
    """Base class for readers that read from a previous pipeline's output."""

    def __init__(self, prev_pipeline: str, **kwargs):
        super().__init__(**kwargs)
        self.prev_pipeline = prev_pipeline

    @property
    @abstractmethod
    def input_file_name(self):
        pass

    @property
    def path(self) -> str:
        return f"runs/{self.dataset_name}/{self.prev_pipeline}/{self.file_name}/{self.input_file_name}"
    
    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass 
    
    @property
    @abstractmethod
    def output_dim(self):
        pass 
    
    @abstractmethod
    def get_iterator(self, file) -> Iterator:
        pass

    @abstractmethod
    def get_timestamp(self, data):
        pass 

    def setup(self):
        self._file = fsspec.open(self.path, "rt", compression="zstd").open()
        self._iter = self.get_iterator(self._file)

    def teardown(self):
        self._file.close()


class CSVReader(PipelineReader):
    @property 
    def input_file_name(self):
        return "features.csv.zst"

    @property
    def batch_size(self) -> int:
        return 1024

    @property
    def output_dim(self):
        return len(self._iter._engine.names)

    def get_iterator(self, file) -> Iterator:
        return pd.read_csv(file, chunksize=self.batch_size, nrows=self.max_records, header=0)

    def get_timestamp(self, data):
        timestamp = data["timestamp"].tolist() 
        
        return timestamp


class JsonGraphReader(PipelineReader):
    @property 
    def input_file_name(self):
        return "input_graph_features.ndjson.zst"
    
    @property
    def batch_size(self) -> int:
        return 1

    @property
    def output_dim(self):
        return None

    def get_iterator(self, file) -> Iterator:
        for line in file:
            yield [nx.readwrite.json_graph.node_link_graph(json.loads(line))]

    def get_timestamp(self, data):
        return [g.graph["time_stamp"] for g in data]