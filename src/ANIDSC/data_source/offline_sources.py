from scapy.all import PcapReader
from ..base_files import PipelineSource
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import Dict, List


class PacketReader(PipelineSource):
    def __init__(self, dataset_name: str, file_name: str, max_pkts: int = float("inf")):
        self.dataset_name = dataset_name
        self.file_name = file_name
        self.context = {"dataset_name": dataset_name, "file_name": file_name}
        self.max_pkts = max_pkts
        self.count = 0

    def start(self):
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
