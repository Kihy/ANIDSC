import json
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
    def __init__(self, fe_name: str, **kwargs):
        """reads data from CSV file

        Args:
            fe_cls (str): class of the feature extractor
            fe_name (str): name of feature extractor

        """
        super().__init__(**kwargs)

        self.fe_name = fe_name
        self.path = f"{self.dataset_name}/{self.fe_name}/features/{self.file_name}.csv"
        self.iter = pd.read_csv(
            self.path, chunksize=self.batch_size, nrows=self.max_records, skiprows=1
        )