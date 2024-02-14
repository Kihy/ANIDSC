import numpy as np
from abc import ABC, abstractmethod
import argparse
import itertools
import os
import subprocess
import sys
from scapy.all import *
from collections import OrderedDict
from itertools import product
from .base_feature_extractor import BaseTrafficFeatureExtractor
import pyshark
from tqdm import tqdm
from pathlib import Path
from utils import LazyInitializationMixin
from collections import defaultdict


class IncStats:
    def __init__(self):
        """
        m (float): cumulative mean.
        m2 (float): cumulative variance.
        m3 (float): cumulative skew.
        m4 (float): cumulative kurtosis.
        n (integer): number of samples seen so far.

        Returns:
            None:

        """
        self.m = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0
        self.n = 0
        self.min = float("Inf")
        self.max = -float("Inf")
        self.eps = 1e-6

    def update(self, x, verbose=False):
        """
        calculates high order statistics(mean, variance, skewness and kurtosis) using
        online algorithm (so we dont get any issues with memory overflow).
        Note the returned values needs to be transformed into actual statistics.
        Args:
            x (float): the current sample value.

        Returns:
            None.

        """
        delta = x - self.m
        delta_n = delta / (self.n + 1)
        delta_n2 = delta_n**2
        term1 = delta * delta_n * self.n
        self.m = self.m + delta_n
        self.m4 = (
            self.m4
            + term1 * delta_n2 * ((self.n + 1) ** 2 - 3 * (self.n + 1) + 3)
            + 6 * delta_n2 * self.m2
            - 4 * delta_n * self.m3
        )
        self.m3 = self.m3 + term1 * delta_n * (self.n - 1) - 3 * delta_n * self.m2
        self.m2 = self.m2 + term1
        self.n += 1
        if verbose:
            print(self.n)

        if self.min > x:
            self.min = x
        if x > self.max:
            self.max = x

    def get_statistics(self):
        """
        calculates statistics based on currently observed data.
        Note the sknew of sample with 0 is 0, and kurtosis of samples with
        same value(e.g. bunch of 1s) is -3

        Returns:
            mean, std, skewness, kurtosis, min and max:

        """

        if self.n == 0:
            return 0, 0, 0, -3, 0, 0

        mean = self.m

        std = np.sqrt(self.m2 / self.n)

        if self.m2 == 0:
            kurtosis = -3.0
            skew = 0
        else:
            kurtosis = (self.n * self.m4) / (self.m2 * self.m2) - 3
            skew = np.sqrt(self.n) * self.m3 / self.m2**1.5
        return mean, std, skew, kurtosis, self.min, self.max


class StreamingInterface(ABC):
    """
    An interface for packet input. It acts as a observable in observer pattern,
    other than standard attach, detach, _notify function it also needs a start
    function to start capturing packet.
    """

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def attach(self):
        pass

    @abstractmethod
    def detach(self):
        pass

    @abstractmethod
    def _notify(self):
        pass

    @abstractmethod
    def _end_signal(self):
        pass


def cartesian_product(*args, seperator="_"):
    """
    creates fieldname as product of args, joined by _.
    e.g. args=["a","b"],["c","d"] will return ["a_c","a_d","b_c","b_d"]

    Args:
        *args (args): iterables to be joined.

    Returns:
        list: list of strings.

    """
    return_list = []
    for field in product(*args):
        return_list.append(seperator.join(field))
    return return_list


class PyFlowMeter(BaseTrafficFeatureExtractor):
    """
    an offline flow meter which extracts various features from flow. Offline
    because it relies on wireshark's stream index as key.

    Args:
        output_path (string): path to output file.
        timeout (int): timeout to determine a connection has finished. Defaults to 600.
        check_interval (int): period in packets to check for timed out flows. Defaults to 1.
        check_range (int): checks the oldest number of flows. Defaults to 100.

    Attributes:
        output_file (file): the output file that can be used write directly.
        flows (dict): a dictionary of current flows.
        timeout (int): duration to consider flow as being finished.
        feature_names (string list): list of feature names.
        check_interval
        check_range

    """

    def __init__(
        self,
        timeout=600,
        check_interval=1,
        check_range=100,
        **kwargs
    ):
        super().__init__(self,**kwargs)
        self.timeout = timeout
        self.check_interval = check_interval
        self.check_range = check_range
        self.name="PyFlowMeter"
        self.flags = ["F", "S", "R", "P", "A", "U", "E", "C"]
        
    def peek(self, traffic_vectors):
        raise NotImplementedError()
    
    def decode_flags(self, flag):
        pointer1=0
        pointer2=0
        flag_array=np.zeros(8, dtype="int32")
        while pointer2!=len(flag):
            if flag[pointer2] == self.flags[pointer1]:
                flag_array[pointer1]=1
                pointer2+=1
            pointer1+=1
        return flag_array


    def get_headers(self):
        directions = ["fwd", "bwd"]
        
        type = ["pkt", "byte"]
        dist_features = ["mean", "std", "skewness", "kurtosis", "min", "max"]
        
        # quickly create field names by using cartesian product of strings
        return (
            ["duration", "protocol"]
            + cartesian_product(directions, ["tot"], type)
            + cartesian_product(directions, ["pkt_size"], dist_features)
            + cartesian_product(directions, ["iat"], dist_features)
            + cartesian_product(directions, self.flags, ["cnt"])
        )

    def setup(self):
        super().setup()
        
        if self.reset_state:
            self.state = {}
            self.state["last_timestamp"]=None
            self.state["flows"]=OrderedDict()
        
        

    def get_traffic_vector(self, packet):
        packet_info = defaultdict(lambda :None)
        
        packet_info["arrival_time"] = packet.time
        packet_info["packet_size"] = len(packet)
        
        if packet.haslayer(IP):  # IPv4
            packet_info["src_IP"] = packet[IP].src
            packet_info["dst_IP"] = packet[IP].dst
            packet_info["protocol"]=packet[IP].proto   
        elif packet.haslayer(ARP):  # IPv4
            packet_info["src_IP"]=packet[ARP].psrc
            packet_info["dst_IP"]=packet[ARP].pdst
            packet_info["protocol"]="ARP"    
        
        if packet.haslayer(TCP):  # IPv4
            packet_info["dst_port"]=packet[TCP].dport
            packet_info["src_port"]=packet[TCP].sport
            packet_info["protocol"]="TCP"
            packet_info["flags"]=self.decode_flags(list(packet[TCP].flags))
        elif packet.haslayer(UDP):  # IPv4
            packet_info["dst_port"]=packet[UDP].dport
            packet_info["src_port"]=packet[UDP].sport
            packet_info["protocol"]="UDP"
         
        return packet_info

    def get_meta_headers(self):
        return ["arrival_time","packet_size","src_IP",
                "dst_IP","dst_port","src_port", "protocol","flags"]    

    def update(self, traffic_vector):
        
        stream_id = frozenset([traffic_vector["protocol"],traffic_vector["src_IP"],
                               traffic_vector["dst_IP"],
                               traffic_vector["src_port"],
                               traffic_vector["dst_port"]])

        if stream_id not in self.state["flows"].keys():
            self._init_stream(stream_id, traffic_vector)
        self._update_stream(stream_id, traffic_vector)

        
        # to speed things up it checks timeout once every certain number of packets.
        # note that self._subject is assigned by streaming interface.
        if self.count % self.check_interval == 0:
            self._check_timeout(traffic_vector["arrival_time"])

    def extract_features(self):
        """
        main cycle of flow meter, called once a packet is generated.

        Args:
            packet (dict): packet being generated, already in dict format

        Returns:
            None

        """
        self.setup()
        meta_list=[]
        
        for packet in tqdm(self.input_pcap,desc=f"parsing {self.file_name}"):
            if not (packet.haslayer(IP) or packet.haslayer(IPv6) or packet.haslayer(ARP)):
                self.skipped+=1
                continue
            
            
            traffic_vector = self.get_traffic_vector(packet)
            
            if self.offset_time is None and self.offset_timestamp:
                self.offset_time=traffic_vector["arrival_time"] - self.state["last_timestamp"]
            else:
                self.offset_time=0
            
            traffic_vector["arrival_time"]-=self.offset_time
            
            meta_list.append([str(traffic_vector[i]) for i in self.get_meta_headers()])
            
            if self.count % 1e4 == 0:
                np.savetxt(
                    self.meta_file, np.vstack(meta_list), delimiter=",", fmt="%s"
                )
                meta_list = []
                
            self.update(traffic_vector)

            self.count += 1
        self.teardown()

    def _check_timeout(self, arrival_time):
        """
        checks the oldest self.check_range number of flows to see if they have exceeded
        self.timeout
        Args:
            arrival_time (type): Description of parameter `arrival_time`.

        Returns:
            type: Description of returned object.

        """
        timed_out_stream = []
        for stream in itertools.islice(self.state["flows"], self.check_range):
            if arrival_time - self.state["flows"][stream]["last_time"] > self.timeout:
                timed_out_stream.append(stream)
        if len(timed_out_stream) > 0:
            self._save_batch_flow(timed_out_stream)

    def _save_batch_flow(self, stream_ids, delete=True):
        """
        saves a batch of flows to csv file, deletes them from self.flows afterwards
        if delete is set to True

        Args:
            stream_ids (int list): list of stream_id to save.
            delete (boolean): whether to delete after saving. Defaults to True.

        Returns:
            None

        """
        feature_names = self.get_headers()
        for index in sorted(list(stream_ids), key=lambda x: self.state["flows"][x]["init_time"]):
            stream = self.state["flows"][index]
            values = [stream[x] for x in feature_names[:6]]

            for i in range(4):
                values += [
                    x for x in stream[feature_names[6 + i * 6][:-5]].get_statistics()
                ]

            values += [x for x in stream["fwd_flags"]]
            values += [x for x in stream["bwd_flags"]]
            self.feature_file.write(",".join(str(x) for x in values))
            self.feature_file.write("\n")
            self.written+=1
            self.feature_file.flush()
            if delete:
                del self.state["flows"][index]

    def _update_stream(self, stream_id, packet_info):
        """
        updates the stream/connection/flow with extracted packet_info.
        Once updated the stream_id is moved to the end of self.flows

        Args:
            packet_info (dictionary): information extracted with tcp_extractor(packet).
            stream_id (int): index of stream stored.

        Returns:
            None: The information is updated directly in stream.

        """
        self.state["last_timestamp"]=packet_info["arrival_time"]
        stream = self.state["flows"][stream_id]

        # determine direction
        if packet_info["src_port"] == stream["src_port"]:
            # packet sent from src port, so it is in the forwarrd direction
            direction = "fwd"
        else:
            direction = "bwd"

        packet_len = int(packet_info["packet_size"])
        time_delta = packet_info["arrival_time"] - stream["last_time"]
        if time_delta<0:
            print("Warining negative iat")
        stream["last_time"] = packet_info["arrival_time"]
        stream["duration"] += time_delta
        stream[direction + "_tot_pkt"] += 1
        stream[direction + "_tot_byte"] += packet_len
        stream[direction + "_pkt_size"].update(packet_len)

        stream[direction + "_iat"].update(time_delta)

        if packet_info["flags"] is not None:
            stream[direction + "_flags"] += packet_info["flags"]

        # move to end
        self.state["flows"].move_to_end(stream_id)

    def _init_stream(self, stream_id, packet_info):
        """
        initializes the stream with default values
        Args:
            stream_id (int): id of stream calculated by wireshark
            packet_info (dict): information extracted from packet.
            arrival_time (float): timestamp of arrival time

        Returns:
            dict: initialized packet information

        """
        init_dict = {}
        features = ["protocol", "src_port", "dst_port"]
        for feature in features:
            init_dict[feature] = packet_info[feature]
        init_dict["duration"] = 0
        init_dict["last_time"] = packet_info["arrival_time"]
        init_dict["init_time"] = packet_info["arrival_time"]

        directions = ["fwd", "bwd"]
        type = ["pkt", "byte"]
        for i in cartesian_product(directions, ["tot"], type):
            init_dict[i] = 0
        for i in cartesian_product(directions, ["iat", "pkt_size"]):
            init_dict[i] = IncStats()
        for i in cartesian_product(directions, ["flags"]):
            init_dict[i] = np.zeros(8, dtype="int32")

        self.state["flows"][stream_id] = init_dict

    def teardown(self):
        """
        writes all remaining flows to file and closes the file.

        Returns:
            None

        """
        
        self._save_batch_flow(self.state["flows"].keys())
        
        super().teardown()
