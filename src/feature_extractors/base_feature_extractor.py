from abc import ABC, abstractmethod
from pathlib import Path
from utils import LazyInitializer, load_dataset_info, save_dataset_info
from scapy.all import *
import pickle
from tqdm import tqdm 
import numpy as np

class BaseTrafficFeatureExtractor(ABC, LazyInitializer):
    def __init__(self, state=None, max_pkt=float("inf"), **kwargs):
        """base feature extractor. By default three variables must be
        set: dataset_name, file_name, and state.
        """
        LazyInitializer.__init__(["dataset_name", "file_name", "state", "save_state"])
        self.set_attr(**kwargs)

        
        self.state=state
        self.max_pkt=max_pkt
        self.entry_func=self.extract_features
        
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

    def setup(self):
        """set up the feature extractor. By default it opens the input pcap
        file, the output feature csv file, output meta data (AKA traffic vector) csv file,
        and sets the state of the extractor.
        It also initializes count (number of packets processed), skipped (number of
        packets skipped), and written (number of features extracted)
        """
        self.path = Path(
            f"../../datasets/{self.dataset_name}/pcap/{self.file_name}.pcap"
        )

        feature_file = Path(
            f"../../datasets/{self.dataset_name}/{self.name}/features/{self.file_name}.csv"
        )
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file = Path(
            f"../../datasets/{self.dataset_name}/{self.name}/metadata/{self.file_name}.csv"
        )
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        self.feature_file = open(feature_file, "w")
        self.meta_file = open(meta_file, "w")
        self.feature_file.write(",".join(self.get_headers()) + "\n")
        self.meta_file.write(",".join(self.get_meta_headers()) + "\n")

        self.count = 0
        self.skipped = 0


        self.input_pcap = PcapReader(str(self.path))

        if self.state is not None:
            self.reset_state = False
            
            self.offset_timestamp = True
        else:
            self.reset_state = True
            
            self.offset_timestamp = False
        self.offset_time = None

    @abstractmethod
    def get_headers(self):
        """returns the names of the features"""
        pass

    @abstractmethod
    def get_meta_headers(self):
        """returns the names of the traffic vectors"""
        pass

    def teardown(self):
        """closes the opened files, write the files in the dataset_info json
        file, save the state.
        """
        self.meta_file.close()
        self.feature_file.close()
        self.input_pcap.close()
        # save file information
        data_info = load_dataset_info()

        if self.dataset_name not in data_info.keys():
            data_info[self.dataset_name] = {}

        if self.name not in data_info[self.dataset_name].keys():
            data_info[self.dataset_name][self.name] = {}

        data_info[self.dataset_name][self.name][self.file_name] = {
            "pcap_path": self.path,
            "feature_path": self.feature_file,
            "meta_path": self.meta_file,
            "num_rows": self.count,
        }

        save_dataset_info(data_info)
        print(
            f"skipped: {self.skipped} processed: {self.count+self.skipped} written: {self.count}"
        )

        if self.save_state:
            state_path = Path(
                f"../../datasets/{self.dataset_name}/{self.name}/state/{self.file_name}.pkl"
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "wb") as pf:
                pickle.dump(self.state, pf)

    
    def extract_features(self):
        """The main entry point of feature extractor, this function
        should define the process of extracting the features from input
        pcap file
        """
        self.setup()

        features_list = []
        meta_list = []
        for packet in tqdm(self.input_pcap, desc=f"parsing {self.file_name}"):
            if self.count>self.max_pkt:
                break
            
            traffic_vector = self.get_traffic_vector(packet)
            
            if traffic_vector is None:
                self.skipped += 1
                continue

            if self.offset_time is None and self.offset_timestamp:
                self.offset_time = traffic_vector[-2] - self.state.last_timestamp
            else:
                self.offset_time = 0
            traffic_vector[-2] -= self.offset_time

            feature = self.update(traffic_vector)
            features_list.append(feature)
            
            meta_list.append([self.count]+[i for i in traffic_vector.values()])
            
            self.count += feature.shape[0]
            chunk_size+=feature.shape[0]
            
            if chunk_size > 1e4:
                np.savetxt(
                    self.feature_file,
                    np.vstack(features_list),
                    delimiter=",",
                    fmt="%s",
                )
                np.savetxt(
                    self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s"
                )
                features_list = []
                meta_list = []
                chunk_size=0

        # save remaining
        np.savetxt(
            self.feature_file, np.vstack(features_list), delimiter=",", fmt="%s"
        )
        np.savetxt(self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s")

        self.teardown()