import json
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

        # Remove 'self' so we only keep real input arguments
        
        self.path = f"{self.dataset_name}/pcap/{self.file_name}.pcap"
        self.iter = PcapReader(self.path)
        

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
        self.save_attr.append("fe_name")
        
        # set fe_attrs to self
        self.fe_attrs=fe_attrs
        self.save_attr.append("fe_attrs")
        
        self.path = f"{self.dataset_name}/{self.fe_name}/features/{self.file_name}.csv"
        self.iter = pd.read_csv(
            self.path, chunksize=self.batch_size, nrows=self.max_records, header=0
        )
        self.feature_names=self.iter._engine.names
        self.ndim=len(self.feature_names)
        
    def __getattr__(self, name):
        return self.fe_attrs[name]