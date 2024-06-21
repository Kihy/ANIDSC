from scapy.all import PcapReader
from ..base_files import PipelineSource
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, List


class PacketReader(PipelineSource):
    def __init__(self, dataset_name: str, file_name: str, max_pkts: int = float("inf")):
        """reads data from pcap file

        Args:
            dataset_name (str): name of dataset
            file_name (str): name of file
            max_pkts (int, optional): maximum number of packets to read. Defaults to float("inf").
        """        
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.context = {"dataset_name": dataset_name, "file_name": file_name}
        self.max_pkts = max_pkts
        self.count = 0

    def start(self):
        """start reading from pcap file
        """        
        self.path = Path(f"{self.dataset_name}/pcap/{self.file_name}.pcap")
        self.input_pcap = PcapReader(str(self.path))

        self.on_start()

        for packet in tqdm(self.input_pcap):
            self.call_back(packet)
            self.count += 1

            if self.count == self.max_pkts:
                break

        self.on_end()


class CSVReader(PipelineSource):
    def __init__(
        self,
        dataset_name: str,
        fe_name: str,
        file_name: str,
        fe_features: int,
        max_pkts: int = float("inf"),
        protocols:Dict[str,int]={},
        batch_size=256,
        skip=0
    ):
        """reads data from CSV file

        Args:
            dataset_name (str): name of dataset
            fe_name (str): name of feature extractor
            file_name (str): file name
            fe_features (int): number of features extracted
            max_pkts (int, optional): maximum number of packets to be read. Defaults to float("inf").
            protocols (Dict[str,int], optional): procotol information from feature extractor. Defaults to {}.
            batch_size (int, optional): batch size . Defaults to 256.
            skip (int, optional): number of feature to skip for normalization. Defaults to 0.
        """        
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.fe_name = fe_name
        self.context = {
            "dataset_name": dataset_name,
            "file_name": file_name,
            "fe_name": fe_name,
            "fe_features": fe_features,
            "output_features":fe_features,
            "protocols":protocols,
            "skip":skip
        }
        self.max_pkts = max_pkts
        self.count = 0
        self.batch_size = batch_size

    def start(self):
        """starts to read from csv file
        """        
        self.path = f"{self.dataset_name}/{self.fe_name}/features/{self.file_name}.csv"
        self.data = pd.read_csv(
            self.path, chunksize=self.batch_size, nrows=self.max_pkts, skiprows=1
        )

        self.on_start()

        for features in tqdm(self.data):
            self.call_back(features.to_numpy())
            self.count += features.shape[0]

            if self.count == self.max_pkts:
                break

        self.on_end()
