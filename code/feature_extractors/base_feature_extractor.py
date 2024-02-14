from abc import ABC, abstractmethod
from pathlib import Path
from utils import *
from scapy.all import *
import pickle


class BaseTrafficFeatureExtractor(ABC, LazyInitializationMixin):
    def __init__(self, **kwargs):
        """base feature extractor. By default three variables must be
        set: dataset_name, file_name, and state.
        """
        self.allowed = ["dataset_name", "file_name", "state"]
        self.lazy_init(**kwargs)
        self.entry = self.extract_features

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
            f"../../datasets/{self.dataset_name}/{self.name}/{self.file_name}.csv"
        )
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file = Path(
            f"../../datasets/{self.dataset_name}/{self.name}/{self.file_name}_meta.csv"
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
            self.save_state = False
            self.offset_timestamp = True
        else:
            self.reset_state = True
            self.save_state = True
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
                f"../../datasets/{self.dataset_name}/{self.name}/state.pkl"
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(state_path, "wb") as pf:
                pickle.dump(self.state, pf)

    @abstractmethod
    def extract_features(self):
        """The main entry point of feature extractor, this function
        should define the process of extracting the features from input
        pcap file
        """
        pass
