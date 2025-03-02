import json
import numpy as np
from scapy.all import PcapReader
from .pipeline_source import PipelineSource
from pathlib import Path
import pandas as pd


class PacketReader(PipelineSource):
    def __init__(self, *args, **kwargs):
        """reads data from pcap file

        Args:
            dataset_name (str): name of dataset
            file_name (str): name of file
            max_pkts (int, optional): maximum number of packets to read. Defaults to float("inf").
        """
        super().__init__(*args, **kwargs)

    def on_start(self):
        self.path = Path(f"{self.dataset_name}/pcap/{self.file_name}.pcap")
        self.iter = PcapReader(str(self.path))

        super().on_start()


class CSVReader(PipelineSource):
    def __init__(self, fe_cls: str, fe_name: str, load_context=True, *args, **kwargs):
        """reads data from CSV file

        Args:
            fe_cls (str): class of the feature extractor
            fe_name (str): name of feature extractor

        """
        super().__init__(*args, **kwargs)

        self.fe_name = fe_name

        if load_context:
            context_file = Path(
                f"{self.dataset_name}/{self.fe_name}/contexts/{self.file_name}.json"
            )
            self.context.update(json.load(open(context_file)))

        self.context.update(
            {
                "fe_name": fe_name,
                "fe_cls": fe_cls,
            }
        )
        
    
    def on_start(self):
        self.path = f"{self.dataset_name}/{self.fe_name}/features/{self.file_name}.csv"
        self.iter = pd.read_csv(
            self.path, chunksize=self.batch_size, nrows=self.max_records, skiprows=1
        )
        super().on_start()
