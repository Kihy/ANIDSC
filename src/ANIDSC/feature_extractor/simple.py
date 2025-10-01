from ANIDSC.feature_buffer.json import JsonFeatureBuffer
from ANIDSC.feature_buffer.tabular import TabularFeatureBuffer
import numpy as np
from networkx.readwrite import json_graph
from ..save_mixin.pickle import PickleSaveMixin
from ..component.feature_extractor import BaseFeatureExtractor
from scapy.all import (
    Ether,
    ARP,
    IP,
    IPv6,
    TCP,
    UDP,
    ICMP,
    Raw,
    DNS,
    DNSQR,
    DNSRR,  # for DNS info if needed
    rdpcap,
)
import networkx as nx


class NetworkAccessGraphExtractor(PickleSaveMixin, BaseFeatureExtractor):
    def __init__(self, granularity=1, **kwargs):
        """"""
        super().__init__(**kwargs)

        self.granularity = granularity
        self.time_stamp = None
        self.G = nx.DiGraph()

        self.feature_buffer = JsonFeatureBuffer(buffer_size=256)
        self.feature_buffer.attach_to(self)

    def peek(self, traffic_vectors):
        pass

    def export_graph(self):
        self.G.graph["time_stamp"] = self.time_stamp
        data = nx.readwrite.json_graph.node_link_data(self.G)
        self.G = nx.DiGraph()
        return data

    def get_headers(self):
        """No header for graphs"""
        return ["size", "count"]

    def update(self, traffic_vector):

        time_stamp = traffic_vector["timestamp"]
        length = traffic_vector["packet_size"]
        src_mac = traffic_vector["srcMAC"]
        dst_mac = traffic_vector["dstMAC"]

        if self.time_stamp is None:
            self.time_stamp = time_stamp
            
        if time_stamp - self.time_stamp > self.granularity:
            self.time_stamp = time_stamp
            graph = self.export_graph()
            
        else:
            graph = None

        if not self.G.has_node(src_mac):
            self.G.add_node(src_mac, count=0, size=0)

        if not self.G.has_node(dst_mac):
            self.G.add_node(dst_mac, count=0, size=0)

        self.G.nodes[src_mac]["count"] -= 1
        self.G.nodes[src_mac]["size"] -= length

        self.G.nodes[dst_mac]["count"] += 1
        self.G.nodes[dst_mac]["size"] += length

        self.G.add_edge(src_mac, dst_mac)

        return graph
