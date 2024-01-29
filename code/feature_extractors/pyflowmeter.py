import numpy as np
from abc import ABC, abstractmethod
import argparse
import itertools
import os
import subprocess
import sys
from collections import OrderedDict
from itertools import product
from .base import BaseTrafficFeatureExtractor
import pyshark
from tqdm import tqdm
from pathlib import Path
from utils import LazyInitializationMixin


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


class Observer(ABC):
    """
    Define an updating interface for objects that should be notified of
    changes in a subject(streaming interface).
    """

    @abstractmethod
    def update(self, packet):
        pass

    @abstractmethod
    def close(self):
        pass


def decode_flags(flag):
    """
    decodes the flag field into a integer array of flag counts.

    Args:
        flag (hexadecimal): the flag field in TCP packet.

    Returns:
        integer array: array of length 8 indicating corresponding flags.

    """
    if flag == "":
        flag = "0x00"
    str_rep = "{:08b}".format(eval(flag))
    return np.array([i for i in str_rep], dtype="int32")


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


class PyFlowMeter(BaseTrafficFeatureExtractor, LazyInitializationMixin):
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
        accepted_protocols=["TCP", "UDP"],
        **kwargs
    ):
        self.timeout = timeout
        self.check_interval = check_interval
        self.check_range = check_range
        self.accepted_protocols = accepted_protocols
        self.column_format = 'gui.column.format:"No.", "%m","Time", "%t","Source", "%s","Destination", "%d","Length", "%L","Stream_index_tcp", "%Cus:tcp.stream:0:R", "Stream_index_udp", "%Cus:udp.stream:0:R","Protocol", "%Cus:ip.proto:0:R","Flags", "%Cus:tcp.flags:0:R","Src_port_tcp", "%Cus:tcp.srcport:0:R","Dst_port_tcp", "%Cus:tcp.dstport:0:R","Src_port_udp", "%Cus:udp.srcport:0:R","Dst_port_udp", "%Cus:udp.dstport:0:R","Info", "%i"'

        self.allowed = "path"
        self.lazy_init(**kwargs)
        self.entry = self.extract_features

    def peek(self, traffic_vectors):
        raise NotImplementedError()

    def get_headers(self):
        directions = ["fwd", "bwd"]
        ports = ["src", "dst"]
        type = ["pkt", "byte"]
        dist_features = ["mean", "std", "skewness", "kurtosis", "min", "max"]
        flags = ["FIN", "SYN", "RST", "PUSH", "ACK", "URG", "CWE", "ECE"]
        # quickly create field names by using cartesian product of strings
        return (
            ["duration", "protocol"]
            + cartesian_product(ports, ["port"])
            + cartesian_product(directions, ["tot"], type)
            + cartesian_product(directions, ["pkt_size"], dist_features)
            + cartesian_product(directions, ["iat"], dist_features)
            + cartesian_product(directions, flags, ["cnt"])
        )

    def setup(self):
        self.num_pkt = 0

        self.basename = Path(self.path).stem
        feature_file = (
            Path(self.path).parents[1] / "features" / (self.basename + ".csv")
        )
        feature_file.parent.mkdir(parents=True, exist_ok=True)

        self.feature_file = open(feature_file, "w")
        self.flows = OrderedDict()
        self.input_pcap = pyshark.FileCapture(
            self.path,
            keep_packets=False,
            #   only_summaries=True,
            custom_parameters={"-o": self.column_format},
        )

        self.feature_file.write(",".join(self.get_headers()) + "\n")

    def get_traffic_vector(self, packet):
        packet_info = {}
        merge_fields = ["dst_port", "src_port", "stream_index"]
        same_fields = ["highest_layer", "flags", "length", "time"]

        for i in same_fields:
            packet_info[i] = getattr(packet, i)
        for i in merge_fields:
            packet_info[i] = getattr(
                packet, "{}_{}".format(i, packet.highest_layer.lower())
            )
        return packet_info

    def update(self, traffic_vector):
        arrival_time = float(traffic_vector["time"])
        stream_id = "{}{}".format(
            traffic_vector["protocol"], traffic_vector["stream_index"]
        )

        if stream_id not in self.flows.keys():
            self._init_stream(stream_id, traffic_vector, arrival_time)
        self._update_stream(stream_id, traffic_vector, arrival_time)

        self.num_pkt += 1
        # to speed things up it checks timeout once every certain number of packets.
        # note that self._subject is assigned by streaming interface.
        if self.num_pkt % self.check_interval == 0:
            self._check_timeout(arrival_time)

    def extract_features(self):
        """
        main cycle of flow meter, called once a packet is generated.

        Args:
            packet (dict): packet being generated, already in dict format

        Returns:
            None

        """
        self.setup()

        for packet in tqdm(self.input_pcap):
            if packet.highest_layer not in self.accepted_protocols:
                continue
            traffic_vector = self.get_traffic_vector(packet)

            self.update(traffic_vector)

            self.num_pkt += 1
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
        for stream in itertools.islice(self.flows, self.check_range):
            if arrival_time - self.flows[stream]["last_time"] > self.timeout:
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
        for index in sorted(list(stream_ids), key=lambda x: self.flows[x]["init_time"]):
            stream = self.flows[index]
            values = [stream[x] for x in feature_names[:8]]

            for i in range(4):
                values += [
                    x for x in stream[feature_names[8 + i * 6][:-5]].get_statistics()
                ]

            values += [x for x in stream["fwd_flags"]]
            values += [x for x in stream["bwd_flags"]]
            self.feature_file.write(",".join(str(x) for x in values))
            self.feature_file.write("\n")
            self.feature_file.flush()
            if delete:
                del self.flows[index]

    def _update_stream(self, stream_id, packet_info, arrival_time):
        """
        updates the stream/connection/flow with extracted packet_info.
        Once updated the stream_id is moved to the end of self.flows

        Args:
            packet_info (dictionary): information extracted with tcp_extractor(packet).
            stream_id (int): index of stream stored.
            arrival_time(float): time of arrival.

        Returns:
            None: The information is updated directly in stream.

        """
        stream = self.flows[stream_id]

        # determine direction
        if packet_info["src_port"] == stream["src_port"]:
            # packet sent from src port, so it is in the forwarrd direction
            direction = "fwd"
        else:
            direction = "bwd"

        packet_len = int(packet_info["length"])
        time_delta = arrival_time - stream["last_time"]
        stream["last_time"] = arrival_time
        stream["duration"] = arrival_time - stream["init_time"]
        stream[direction + "_tot_pkt"] += 1
        stream[direction + "_tot_byte"] += packet_len
        stream[direction + "_pkt_size"].update(packet_len)

        stream[direction + "_iat"].update(time_delta)

        flags = decode_flags(packet_info["flags"])
        if len(flags > 8):
            flags = flags[-8:]
        stream[direction + "_flags"] += flags

        # move to end
        self.flows.move_to_end(stream_id)

    def _init_stream(self, stream_id, packet_info, arrival_time):
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
        init_dict["last_time"] = arrival_time
        init_dict["init_time"] = arrival_time

        directions = ["fwd", "bwd"]
        type = ["pkt", "byte"]
        for i in cartesian_product(directions, ["tot"], type):
            init_dict[i] = 0
        for i in cartesian_product(directions, ["iat", "pkt_size"]):
            init_dict[i] = IncStats()
        for i in cartesian_product(directions, ["flags"]):
            init_dict[i] = np.zeros(8, dtype="int32")

        self.flows[stream_id] = init_dict

    def teardown(self):
        """
        writes all remaining flows to file and closes the file.

        Returns:
            None

        """
        print("saved at", self.feature_file.name)
        self._save_batch_flow(self.flows.keys())
        self.feature_file.close()
        self.input_pcap.close()
